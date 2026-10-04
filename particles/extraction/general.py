# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""General-purpose LLM extractor (§C.4, §9.2).

Implements the ExtractorInterface from §14.3:
  extract(snapshot, config, prior) -> ExtractionResult
  accepts(snapshot) -> {accepted, reason}

Uses the Anthropic API with claude-sonnet-4-6 to extract claim-granularity
particles from any source type. Particles produced by this extractor carry
calibration_source = EXTRACTOR_DIRECT by default; when a calibration record
exists for the extractor, the pipeline threads it into
``candidate_to_particle`` and the particle is graduated to
CALIBRATED_BENCHMARK.
"""

from __future__ import annotations

import contextlib
import functools
import inspect
import json
import logging
import math
import re
import time
from collections.abc import Iterator, Mapping, Sequence
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal

from particles.config import get_config
from particles.core.schema import (
    SCHEMA_VERSION,
    ApplicabilityClause,
    AssertionModality,
    CanonicalForm,
    Confidence,
    ContributorRef,
    ExternalRef,
    ExtractorCalibration,
    ExtractorRef,
    Particle,
    ParticleType,
    ProvenanceRef,
    ProvenanceRefType,
    RelationType,
    Snapshot,
    StructuredClaim,
    UncertaintyNature,
)
from particles.core.scoring.confidence import CalibrationSource
from particles.extraction import polarity as _polarity_module
from particles.extraction import scope as _scope_module
from particles.extraction import session_state as _session_state_module
from particles.extraction import structure as _structure_module
from particles.extraction.components import (
    ComponentTable,
    PromptComponent,
    assemble_prompt,
    component_digest,
    declare_components,
    record_component,
    record_prompt_components,
)
from particles.extraction.polarity import (
    NON_ASSERTED_POLARITIES,
    POLARITY_ASSERTED,
    POLARITY_KEY,
)
from particles.extraction.scope import (
    SCOPE_ACTION_KEY,
    SCOPE_ACTION_OBSERVE,
    SCOPE_DOCUMENT_META,
    SCOPE_KEY,
)
from particles.extraction.session_state import drop_session_state
from particles.extraction.structure import bind_subject_id, parse_structured_claim_payload
from particles.extraction.tool_turns import TOOL_TURN_RULE, mark_tool_turns

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from particles.extraction.incremental import ChunkUnit
    from particles.llm import CompletionPool, CompletionRequest, RequestFailure, VisionImage

log = logging.getLogger(__name__)

EXTRACTOR_ID = "general-extractor"
EXTRACTOR_VERSION = "0.16.0"
# 0.4.0: prompt now preserves URL schemes.
# 0.5.0: HTML chunking moves to paragraph boundaries with hash-input
#        normalisation and routes through extract_with_carry_forward.
#        Chunk boundaries differ from 0.4.0, so the
#        version-mismatch rule forces a one-time full reindex.
# 0.6.0: prompt now classifies each candidate's document scope
#        (WORLD vs DOCUMENT_META) and the parser applies the configured
#        mode. Output differs from 0.5.0, so the
#        version-mismatch rule forces a one-time full reindex.
# 0.7.0: prompt now also classifies each candidate's assertion_modality
#        (FALSIFIABLE / EVALUATIVE / EXPERIENTIAL / CONSTITUTIVE),
#        populating the field. Output differs from 0.6.0, so the
#        version-mismatch rule forces a one-time full reindex.
# 0.9.0: PDF path is modality-aware. When
#        extraction_vision.enabled, an image-bearing / low-text page is sent to
#        the vision-capable provider as one multimodal call (page text + a
#        rendered page image) instead of the text-only call, recovering claims
#        from diagrams / charts / scanned pages; vision-derived candidates carry
#        properties["extraction:source_modality"] = "vision". With vision
#        disabled (the default) the output is byte-for-byte the 0.8.0 behaviour,
#        so the version bump only forces a reindex for operators who enable it.
# 0.10.0: standalone image deposits (IMAGE source type).
#        An image blob (PNG/JPEG/GIF/WebP) is sent to the vision-capable provider
#        in one multimodal call (no rasterizer — the bytes are already an image),
#        marked source_modality=vision. Adds a new code path only for IMAGE
#        content, so non-image extraction output is unchanged.
# 0.11.0: prompt-injection hardening (security F3). The trusted rules/schema move
#        to the ``system`` turn and the untrusted source is wrapped in a per-call
#        nonce fence in the user turn (was: rules + "SOURCE TEXT:" + raw source
#        concatenated into one user message). The semantic instructions and the
#        JSON-array output contract are unchanged, so benign sources extract the
#        same particles; a *malicious* source extracts differently (injected
#        instructions are now resisted), which is why the version bumps — an
#        operator can ``reindex --extractor-version 0.10.0`` to re-extract a
#        previously-poisoned corpus under the hardened prompt.
# 0.12.0: event-anchored validity extraction. When
#        extraction_validity.enabled (default), the prompt asks the model to emit
#        a ``valid_until`` boundary + ``validity_confidence`` + ``validity_basis``
#        ONLY for a claim genuinely bounded in time, and the parser gates emission
#        on an explicit cue + a confidence floor + a future resolved date (a
#        born-expired past bound is dropped). A date-bounded source now extracts a
#        ``valid_until`` it did not before, so an operator ``reindex``es to apply
#        it; sources with no bounded claims — and the ``enabled: false`` case —
#        are byte-for-byte the 0.11.0 output.
# 0.13.0: derived structured-claim annotation. When
#        structured_claim.enabled (default), the prompt asks for one S-P-O triple
#        per claim as an *annotation* on ``content`` — no extra LLM call, and
#        nothing about the claim itself changes: same content, same confidence,
#        same provenance. Only the derived annotation is new, so a reindex is
#        optional (``particles structure`` annotates in place, far cheaper than
#        re-extraction). The ``enabled: false`` case is byte-for-byte 0.12.0.
# 0.14.0: the annotation keys move under the prefix rule
#        ``extraction:`` prefix — ``polarity`` -> ``extraction:polarity``,
#        ``scope`` -> ``extraction:scope``, ``scope_action`` ->
#        ``extraction:scope_action``, joining the ``extraction:validity_*`` /
#        ``extraction:source_modality`` keys this same path already wrote. The
#        claims are identical; only the ``properties`` key spelling differs, and
#        Alembic 035 puts existing particles in exactly this state, so **no
#        reindex is required** — the version bumps because the same source now
#        yields different ``properties``, which is what the
#        version-mismatch rule keys on.
# 0.15.0: confidence elicitation reworked and the parse clamp tightened to
#        [0.01, 0.99]. The same source now yields a different
#        ``confidence_value``, so this **does** require a reindex to apply
#        retroactively. ``particles reindex --extractor-version 0.14.0``
#        names the 0.14.0 tranche; it is NOT the whole stale set — every
#        earlier version predates this rubric too, and on the author's store
#        0.14.0 was 3,921 particles against 33,574 uncalibrated in total.
# 0.16.0: reference resolution. The prompt now asks the model to replace a
#        deictic or relative reference ("my flat", "last month", "my current
#        job") with what the same source says it is, to emit the locating claim
#        a source implies ("X is in <place>"), and to keep relative wording when
#        the source does not resolve it. A claim that read "a ten-minute walk
#        from the user's flat" silently changed referent when the user moved.
#        The reference date now rides every prompt, not only the validity rule.
#        Output differs from 0.15.0; ``reindex --extractor-version 0.15.0``
#        re-extracts the prior tranche.
#: The interval every emitted ``confidence_value`` is clamped into
#: (D1). Strict interiority is the whole point: ``calibration.is_saturated``
#: treats a value within ``_EPS = 1e-7`` of an endpoint as immovable, so
#: anything strictly inside is fittable and the exact bound is not
#: load-bearing. These are the widest such bounds that survive a JSON float
#: round-trip unambiguously, chosen to minimise how far a stated value moves.
_CONFIDENCE_FLOOR = 0.01
_CONFIDENCE_CEILING = 0.99

DEFAULT_TRUST_WEIGHT = 0.70

# Extension-side audit crumbs recorded whenever the extractor sets a
# ``valid_until`` — because a validity boundary can flip a particle out of ACTIVE
# (via the §9.3 staleness lint), an operator reviewing a VALIDITY_EXPIRED
# retirement must be able to see *why* the extractor thought it expires. Core
# never branches on these; they ride through interchange as ordinary
# ``properties`` entries. ``valid_until`` itself is the first-class field.
VALIDITY_BASIS_KEY = "extraction:validity_basis"
VALIDITY_CONFIDENCE_KEY = "extraction:validity_confidence"

# marker recording that a candidate was extracted via the vision pass
# (a figure / scanned page), for the audit trail. A ``properties`` entry
# (Extension), not a schema field — no spec impact.
VISION_SOURCE_MODALITY_KEY = "extraction:source_modality"
VISION_SOURCE_MODALITY = "vision"
APPLICABILITY = [
    ApplicabilityClause(
        keyword="MAY",
        domain_uri="http://www.wikidata.org/entity/Q35127",
        domain_label="website",
        source_types=["WEB_PAGE"],
    ),
    ApplicabilityClause(
        keyword="MAY",
        domain_uri="http://www.wikidata.org/entity/Q49848",
        domain_label="document",
        source_types=["PDF"],
    ),
    ApplicabilityClause(
        keyword="MAY",
        domain_uri="http://www.wikidata.org/entity/Q478798",
        domain_label="image",
        source_types=["IMAGE"],
    ),
    # No MUST_NOT — GeneralExtractor is the unconditional fallback.
]

# standalone-image detection by magic bytes, mapped to the IANA media
# type the vision provider needs. Limited to the Anthropic-supported set.
_IMAGE_MAGIC: tuple[tuple[bytes, str], ...] = (
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
)


def sniff_image_media_type(content: bytes) -> str | None:
    """Return the IANA image media type for ``content``, or None if not an image.

    Detects PNG / JPEG / GIF by leading magic bytes and WebP by the
    ``RIFF....WEBP`` container header. Used both at deposit (to stamp the
    ``IMAGE`` source type) and at extraction (to build the ``VisionImage``).
    """
    for magic, media_type in _IMAGE_MAGIC:
        if content.startswith(magic):
            return media_type
    if content[:4] == b"RIFF" and content[8:12] == b"WEBP":
        return "image/webp"
    return None


_EXTRACT_RULES = """\
You are a particle extractor. Extract every individual, falsifiable claim (one that could in
principle be shown false) from the source text. The source text is supplied in the user
message, wrapped in a data fence (see the SECURITY note below).

Rules:
- One particle = one falsifiable assertion (a single claim that could in principle be shown false)
- Do not include opinions, questions, instructions, or non-falsifiable statements (those that
  could never be shown false even in principle)
- Use only information from the source text; do not add external knowledge
- confidence_value: how clearly and definitively THIS source states the claim.
  **Never emit exactly 1.0 or exactly 0.0.** A source can state something
  plainly without putting it beyond revision, and those two values are fixed
  points that no downstream calibration can correct — an extractor that emits
  them cannot be calibrated at all. Keep every value in [0.05, 0.97].
  - 0.90–0.97: stated flatly and unconditionally, in the source's own voice
  - 0.75–0.89: stated as fact but attributed, approximate, or carrying one
    qualifier ("about 40 people", "according to the filing")
  - 0.60–0.74: hedged, provisional, or framed as expectation ("we expect",
    "should", "is on track to")
  - 0.40–0.59: speculative, second-hand, or contested elsewhere in the source
  - 0.05–0.39: raised only as a possibility, or explicitly doubted
  Use the spread. Within one document some claims are stated more firmly than
  others; returning the same value for most claims means that difference was
  not read.
- uncertainty_nature: EPISTEMIC (knowledge could in principle resolve it) or
  ALEATORY (inherently random/irreducible, e.g. future events, quantum phenomena)
- subjects: the real-world entities this claim is primarily about. Use the most
  specific name from the source text (e.g. "1 Pfennig (1960-1990) GDR" rather
  than "coin"). Use an empty list [] for general claims not about a specific entity.
  For relational claims (X relates to Y), include both entities.
- Preserve URLs verbatim, including the scheme (https:// or http://). Do not
  shorten, paraphrase, or strip the scheme — downstream renderers rely on the
  scheme to auto-link them."""

# Reference resolution: a claim is stored alone and read later beside claims
# from other sources, so a deictic or relative reference ("my flat", "last
# month") silently changes referent when the speaker's situation changes. Shared
# with the journal extractor, so it names no date line of its own: the general
# prompt supplies the reference date through ``_REFERENCE_DATE_RULE`` (or the
# validity rule, which already carries it). The examples deliberately name
# nothing a test fixture uses, so a passing test is resolution, not copying.
REFERENCE_RULE = """
- Resolve references before you write a claim. Each claim is stored on its own
  and read later beside claims from other sources, so it must still mean the
  same thing then. Words whose meaning depends on who is speaking, where, or
  when ("my flat", "home", "here", "this city", "my current job", "my manager",
  "last month", "yesterday", "now", "this year") are replaced with what they
  refer to whenever THIS source says what that is:
  - a place: in a source where Maria says she lives in Kreuzberg, Berlin,
    "Rosa's Bakery is five minutes from my flat" becomes "Rosa's Bakery is five
    minutes from Maria's flat in Kreuzberg, Berlin";
  - a time: in a source dated 2026-03-14, "I started the course last month"
    becomes "Maria started the course in February 2026". Resolve against a date
    the source states for itself (a dated heading, a timestamp, a "Date:" line)
    first, and otherwise against the REFERENCE DATE when one is given;
  - a role or relation: "my current job" in a source that names the employer
    becomes the employer's name.
  Resolving a reference adds words to the claim; it does not change what the
  claim is about. Keep in "subjects" every entity you would have listed before
  resolving it, including the speaker wherever the subjects rule lists the
  speaker: "Maria moved to Hamburg in February 2026" is about Maria and
  Hamburg, never Hamburg alone.
- When the source places a named thing only by reference to such an anchor
  ("five minutes from my flat", "the café downstairs"), also emit the locating
  claim it implies as its own particle ("Rosa's Bakery is in Kreuzberg,
  Berlin"), at a confidence no higher than the claim it comes from.
- When the source does not say what a reference points to, keep its own
  relative wording. Never invent a place, a date, an employer, or any other
  anchor the source does not give."""

# The reference date for REFERENCE_RULE when the validity rule (which carries
# the same placeholder) is off, so relative time always has an anchor.
_REFERENCE_DATE_RULE = """
- REFERENCE DATE: the source was published/captured on {reference_date}."""

# optional document-scope classification clause. Appended to the
# rules and the JSON schema only when ``extraction_scope.enabled`` is set, so
# a disabled feature leaves the prompt (and the extractor output) unchanged.
_SCOPE_RULE = """
- scope: WORLD if the claim is about an entity, event, fact, or relationship in
  the world. DOCUMENT_META if the claim is instead about THIS source document's
  own structure or editorial apparatus — its sections, numbering, headings,
  tables, figures, cross-references, or the document as an artifact (e.g.
  "Section 10.4 defines the wiki exporter", "Table 3 lists the results", "this
  specification supersedes the previous version"). When unsure, use WORLD."""

_SCOPE_SCHEMA_FIELD = ',\n  "scope": "WORLD" or "DOCUMENT_META"'

# optional assertion_modality classification clause. Woven in only
# when ``extraction_modality.enabled`` is set; default-safe — the parser falls
# back to FALSIFIABLE on any missing / unknown value, so a classifier miss can
# only ever leave a claim in the truth engine, never wrongly exempt one.
_MODALITY_RULE = """
- assertion_modality: FALSIFIABLE if the claim is an observer-independent
  statement that could in principle be shown true or false (the default —
  facts, events, measurements, relationships). EVALUATIVE if it is a value
  judgement, preference, or ranking with no fact of the matter ("X is the best
  language", "this design is elegant"). EXPERIENTIAL if it is a first-person
  report of an inner state ("I felt anxious", "I was excited"). CONSTITUTIVE if
  it is a rule, requirement, or definition the source document *establishes*
  rather than an empirical claim it reports ("a Particle MUST carry a
  confidence value", "conformance REQUIRES …"). When unsure, use FALSIFIABLE."""

_MODALITY_SCHEMA_FIELD = (
    ',\n  "assertion_modality": "FALSIFIABLE", "EVALUATIVE", "EXPERIENTIAL", or "CONSTITUTIVE"'
)

# capability 1: optional claim-polarity clause. Woven in only when
# ``extraction_polarity.enabled`` is set; default-safe — the parser falls back
# to ASSERTED on any missing / unknown value, so a classifier miss can only
# ever leave a claim on the factual surface, never wrongly hide a real decision
# (the dangerous direction). Classify by how the document FRAMES the
# proposition, never whether it is true — a rejected design may be truly
# rejected (confidence is never the lever).
_POLARITY_RULE = """
- polarity: how THIS source document presents the proposition. ASSERTED (the
  default) if the document puts it forward as a decision it makes or a claim it
  holds. DECLINED if the document presents it as rejected, superseded, deferred,
  or out-of-scope ("we rejected X", "X was superseded by Y", "X is deferred",
  "X is out of scope; do Y for now", a Rejected-Alternatives entry). HYPOTHETICAL
  if the document presents it as a counterfactual, conditional, motivational
  supposition, future projection, or worked example ("without a single source of
  truth, audit trails will be unreliable", "suppose X", "for example, if Y").
  Judge by how the document FRAMES the proposition, not whether it is true — a
  rejected design may be truly rejected. When unsure, use ASSERTED."""

_POLARITY_SCHEMA_FIELD = ',\n  "polarity": "ASSERTED", "DECLINED", or "HYPOTHETICAL"'

# optional endorsement-stance clause. Woven in only when
# ``extraction_stance.enabled`` is set. Default-safe toward UNDER-emission
# (M3): a stance is emitted only when the source explicitly positions
# itself toward another claim *in this same array* — the dangerous direction is
# emission (a spurious stance is permanent substrate that distorts the §4
# agreement view), so "when unsure, omit the stance fields".
_STANCE_RULE = """
- stance: set the stance fields ONLY when the source itself **explicitly takes a
  position toward another claim that also appears in this array** — endorsing,
  disputing, rebutting, or concurring with it ("I disagree that …", "@bob is
  wrong about …", "the authors concur with Smith that …", "+1, exactly right").
  Set "stance_kind" to "ENDORSES" or "DISPUTES"; set "stance_target" to the
  0-based index, within THIS array, of the claim being endorsed / disputed
  (which MUST be another element you also emit); set "stance_magnitude" to the
  strength the source expressed (float 0.0–1.0) or null if unqualified. A mere
  assertion, question, hedge, or bare re-share is NOT a stance; do NOT emit a
  stance for an author trivially agreeing with their own assertion. **When in
  any doubt, leave stance_kind null** — a missing stance is harmless, a spurious
  one is not."""

_STANCE_SCHEMA_FIELD = (
    ',\n  "stance_kind": "ENDORSES" | "DISPUTES" | null'
    ',\n  "stance_target": <0-based index of the target claim in this array, or null>'
    ',\n  "stance_magnitude": <float 0.0–1.0, or null>'
)

# optional event-anchored validity clause. Woven in only when
# ``extraction_validity.enabled``. Default-safe toward UNDER-emission — the
# dangerous direction is emission (a spurious ``valid_until`` silently retires a
# durable fact via the §9.3 staleness lint), so the rule draws the
# mention-vs-boundary line explicitly and says "when unsure, omit it". The
# {reference_date} placeholder is filled per-call with the source's publication
# instant (or the extraction wall-clock) so relative expressions resolve to
# absolute dates in the model, never in the parser.
_VALIDITY_RULE = """
- valid_until: set this ONLY when the claim asserts a state or arrangement that
  CEASES to be true at a specific future time — a genuine validity boundary.
  Emit an ISO-8601 date (or datetime) naming when the claim stops holding.
  - Emit for: an arrangement with a built-in end ("the contract runs through
    2026", "she is the interim CEO until March", "the offer expires Friday",
    "the visa is valid until 2027-06-30"), or a scheduled event whose occurrence
    ends the claim ("the exam is tomorrow", "the summit is next week").
  - Do NOT emit for a durable fact that merely MENTIONS a date: a completed past
    event ("I met her in 2019", "the treaty was signed in 1919"), an origin
    ("founded in 1998", "born in 1980"), or any claim that stays true
    indefinitely. The date being present is not a boundary.
  - The test: does the claim STOP being true after the date? Only then is there
    a valid_until. When unsure, OMIT it — a missing boundary is harmless, a wrong
    one silently retires a true fact.
  - REFERENCE DATE: the source was published/captured on {reference_date}.
    Resolve relative expressions ("tomorrow", "next week", "until Q3") against it.
    Only emit valid_until if you can name an absolute calendar date.
- validity_confidence: when you set valid_until, your self-assessed certainty
  [0.0–1.0] that this is a real validity boundary AND that you resolved the date
  correctly. null when you set no boundary.
- validity_basis: when you set valid_until, a short verbatim quote of the cue
  phrase that bounds validity ("runs through 2026", "exam tomorrow"). null
  otherwise."""

_VALIDITY_SCHEMA_FIELD = (
    ',\n  "valid_until": "<ISO-8601 date or datetime, or null>"'
    ',\n  "validity_confidence": <float 0.0–1.0, or null>'
    ',\n  "validity_basis": "<short verbatim cue quote, or null>"'
)

_STRUCTURE_RULE = """
- structured_claim: re-express the claim's relational core as ONE
  subject-predicate-object triple. This is an ANNOTATION on the claim, never a
  replacement for it — "content" remains the assertion.
  - subject: the entity the claim is about. Use one of the names you put in
    "subjects", verbatim, so the triple and the claim point at the same entity.
  - predicate: the relation, as a short lowercase verb phrase ("was minted at",
    "has weight", "reports to"). Use an ontology URI only when the source
    itself names one.
  - object: the value or entity the relation points at.
  - Emit null when the claim has no single clean triple — a compound
    statement, a hedge, an evaluation, a first-person report, or anything whose
    relational core you would have to invent. A missing triple is harmless and
    permanently fine; a wrong one is a false statement this system will publish
    and act on. When in doubt, emit null.
"""

_STRUCTURE_SCHEMA_FIELD = (
    ',\n  "structured_claim": {"subject": "<entity>", "predicate": "<relation>",'
    ' "object": "<value>"} or null'
)

# The output-schema frame. The optional fields of each enabled rule are woven
# in between its head and its tail, which is why the frame is one component
# placed last in the table (see ``assemble_prompt``): its head closes the rules
# block and its tail closes the fields block.
_EXTRACT_SCHEMA_HEAD = '''

Return ONLY a JSON array. No prose before or after. Each element:
{
  "content": "<the claim as a complete sentence>",
  "subjects": ["<entity name>"],
  "confidence_value": <float 0.0–1.0>,
  "uncertainty_nature": "EPISTEMIC" or "ALEATORY"'''
_EXTRACT_SCHEMA_TAIL = """
}"""


#: The rule a delta read carries when its prompt holds a context section
#:. In the prompt proper, never inlined in the chunk text, so the
#: instruction is a trusted rule rather than words inside the fenced data.
APPEND_CONTEXT_RULE = """
- The user message holds two fenced sections. The one labelled "context" is
  earlier text of this same source, already extracted. Use it ONLY to work out
  what the "source" section refers to (a person, a file, a failure described
  earlier, "that", "it", "the fix"). Extract NO claim from the context section,
  even one it states plainly. Every claim you return must be stated in the
  "source" section, with its references resolved from the context where the
  source section alone leaves them unclear.
"""


#: Component names. Stable identifiers: a stored record names a
#: component by these, so renaming one reads as "removed and added" to every
#: snapshot stamped before the rename. Change the text freely; keep the name.
PROMPT_RULES = "prompt.general.rules"
PROMPT_REFERENCE = "prompt.reference"
PROMPT_SCOPE = "prompt.scope"
PROMPT_MODALITY = "prompt.modality"
PROMPT_POLARITY = "prompt.polarity"
PROMPT_STANCE = "prompt.stance"
PROMPT_VALIDITY = "prompt.validity"
PROMPT_REFERENCE_DATE = "prompt.reference_date"
PROMPT_STRUCTURE = "prompt.structure"
PROMPT_TOOL_TURNS = "prompt.tool_turns"
PROMPT_APPEND_CONTEXT = "prompt.append_context"
PROMPT_SCHEMA = "prompt.general.schema"
PATH_SINGLE_PASS = "path.single_pass"
PATH_CHUNKED = "path.chunked"
PATH_APPEND_DELTA = "path.append_delta"
PATH_PDF_PAGES = "path.pdf_pages"
PATH_IMAGE = "path.image"
CHANNEL_VISION = "channel.vision"
CODE_PARSE = "code.general.parse"
ROUTING = "routing.general"


def modality_prompt_component() -> PromptComponent:
    """The assertion-modality rule as one extraction component.

    The classifier identity a modality stamp names is this component's
    ``name@digest``, and the standalone regeneration classifier
    runs this same component, so a regenerated claim and a freshly extracted
    one carry one identity while the rule text is unchanged.
    """
    return PromptComponent(PROMPT_MODALITY, _MODALITY_RULE, _MODALITY_SCHEMA_FIELD)


def extract_prompt_components(
    *,
    scope_enabled: bool,
    modality_enabled: bool,
    polarity_enabled: bool = False,
    stance_enabled: bool = False,
    validity_enabled: bool = False,
    structure_enabled: bool = False,
    tool_turns_present: bool = False,
    context_present: bool = False,
) -> list[PromptComponent]:
    """The components of one extraction prompt, in prompt order.

    The single table the prompt is assembled from (:func:`_build_extract_prompt`)
    and the extraction records as exercised (:func:`_build_llm_request`), so a
    component's name cannot drift from its text. Each optional rule is one
    component carrying both its rule text and its JSON-schema field; the output
    schema is the last component (see :func:`assemble_prompt`).
    """
    table: list[PromptComponent | None] = [
        PromptComponent(PROMPT_RULES, _EXTRACT_RULES),
        PromptComponent(PROMPT_REFERENCE, REFERENCE_RULE),
        PromptComponent(PROMPT_SCOPE, _SCOPE_RULE, _SCOPE_SCHEMA_FIELD) if scope_enabled else None,
        modality_prompt_component() if modality_enabled else None,
        PromptComponent(PROMPT_POLARITY, _POLARITY_RULE, _POLARITY_SCHEMA_FIELD)
        if polarity_enabled
        else None,
        PromptComponent(PROMPT_STANCE, _STANCE_RULE, _STANCE_SCHEMA_FIELD)
        if stance_enabled
        else None,
        PromptComponent(PROMPT_VALIDITY, _VALIDITY_RULE, _VALIDITY_SCHEMA_FIELD)
        if validity_enabled
        else PromptComponent(PROMPT_REFERENCE_DATE, _REFERENCE_DATE_RULE),
        PromptComponent(PROMPT_STRUCTURE, _STRUCTURE_RULE, _STRUCTURE_SCHEMA_FIELD)
        if structure_enabled
        else None,
        # only when the source actually carries a tool turn, so
        # every other prompt is byte-for-byte unchanged.
        PromptComponent(PROMPT_TOOL_TURNS, TOOL_TURN_RULE) if tool_turns_present else None,
        # only on a delta read with a context section.
        PromptComponent(PROMPT_APPEND_CONTEXT, APPEND_CONTEXT_RULE) if context_present else None,
        PromptComponent(PROMPT_SCHEMA, _EXTRACT_SCHEMA_HEAD, _EXTRACT_SCHEMA_TAIL),
    ]
    return [component for component in table if component is not None]


def _build_extract_prompt(
    *,
    scope_enabled: bool,
    modality_enabled: bool,
    polarity_enabled: bool = False,
    stance_enabled: bool = False,
    validity_enabled: bool = False,
    structure_enabled: bool = False,
    tool_turns_present: bool = False,
    context_present: bool = False,
) -> str:
    """Assemble the extraction prompt from :func:`extract_prompt_components`.

    When ``scope_enabled``, the document-scope rule and ``scope``
    JSON field are woven in; when ``modality_enabled``, the
    assertion-modality rule and field; when ``polarity_enabled`` (
    cap. 1), the claim-polarity rule and field; when ``stance_enabled``,
    the endorsement-stance rule and fields; when
    ``validity_enabled``, the event-anchored validity rule and the
    ``valid_until`` / ``validity_confidence`` / ``validity_basis`` fields; when
    ``structure_enabled``, the S-P-O annotation rule and the
    ``structured_claim`` field. Disabling any feature removes exactly its rule
    and field. The reference-resolution rule (0.16.0) is always present.

    The prompt always carries a ``{reference_date}`` placeholder, in the
    validity rule when it is on and in ``_REFERENCE_DATE_RULE`` otherwise, so
    relative time in a claim can be resolved either way. :func:`_call_llm`
    substitutes it per call (the source's publication instant or the extraction
    wall-clock); the substitution uses ``str.replace`` — never ``str.format`` —
    because the JSON schema block contains literal braces.
    """
    return assemble_prompt(
        extract_prompt_components(
            scope_enabled=scope_enabled,
            modality_enabled=modality_enabled,
            polarity_enabled=polarity_enabled,
            stance_enabled=stance_enabled,
            validity_enabled=validity_enabled,
            structure_enabled=structure_enabled,
            tool_turns_present=tool_turns_present,
            context_present=context_present,
        )
    )


def _configured_prompt_components(
    *, tool_turns_present: bool = False, context_present: bool = False
) -> list[PromptComponent]:
    """:func:`extract_prompt_components` under the current config."""
    cfg = get_config()
    return extract_prompt_components(
        scope_enabled=cfg.extraction_scope.enabled,
        modality_enabled=cfg.extraction_modality.enabled,
        polarity_enabled=cfg.extraction_polarity.enabled,
        stance_enabled=cfg.extraction_stance.enabled,
        validity_enabled=cfg.extraction_validity.enabled,
        structure_enabled=cfg.structured_claim.enabled,
        tool_turns_present=tool_turns_present,
        context_present=context_present,
    )


def path_component_digests() -> dict[str, str]:
    """The digest of each extraction path and of the vision channel, under current config.

    A path's text is the settings that decide what the model is shown on it
    (where chunks break, how much context a delta carries, which PDF pages are
    read); the vision channel's is the settings that decide which pages go to
    the vision provider and at what resolution. A change to one re-digests
    that path only.
    """
    ext = get_config().extraction
    vision = get_config().extraction_vision
    return {
        PATH_SINGLE_PASS: component_digest("single_pass", f"html_chunk_size={ext.html_chunk_size}"),
        PATH_CHUNKED: component_digest(
            "paragraph_chunks",
            f"html_chunk_size={ext.html_chunk_size}",
            f"html_chunk_overlap_lines={ext.html_chunk_overlap_lines}",
            f"max_llm_calls_per_source={ext.max_llm_calls_per_source}",
        ),
        PATH_APPEND_DELTA: component_digest(
            "append_delta",
            f"append_context_chars={ext.append_context_chars}",
            f"append_chunk_chars={ext.append_chunk_chars}",
            f"max_llm_calls_per_source={ext.max_llm_calls_per_source}",
        ),
        PATH_PDF_PAGES: component_digest(
            "pdf_pages",
            f"pdf_page_overlap_lines={ext.pdf_page_overlap_lines}",
            f"max_pdf_pages={ext.max_pdf_pages}",
            f"max_pdf_page_chars={ext.max_pdf_page_chars}",
        ),
        PATH_IMAGE: component_digest("image", f"max_image_bytes={ext.max_image_bytes}"),
        CHANNEL_VISION: component_digest(
            "vision",
            f"trigger={vision.trigger}",
            f"low_text_threshold={vision.low_text_threshold}",
            f"render_dpi={vision.render_dpi}",
            f"max_pages={vision.max_pages}",
        ),
    }


def source_text(obj: object) -> str:
    """The source of a function or module, for a code component's digest.

    Falls back to a text that changes with every ``EXTRACTOR_VERSION`` when no
    source is shipped, so the component then reads as changed on every bump:
    the safe direction.
    """
    try:
        return inspect.getsource(obj)  # type: ignore[arg-type]
    except (OSError, TypeError):
        return f"no-source:{EXTRACTOR_VERSION}"


@functools.cache
def _parse_code_texts() -> tuple[str, ...]:
    """The source of everything between the model's reply and the stored claim.

    Read once per process: source does not change while it runs. Named
    functions rather than this whole module, because the module also holds the
    prompt text, and a prompt edit must change only its own component.
    """
    functions = (
        extraction_text,
        content_to_text,
        _strip_obsidian_frontmatter,
        mark_tool_turns,
        _finish_llm_reply,
        _parse_extraction_response,
        _parse_extraction_reply,
        _gate_valid_until,
        _drop_session_state_candidates,
        candidate_to_particle,
    )
    modules = (_polarity_module, _scope_module, _session_state_module, _structure_module)
    return tuple(source_text(f) for f in (*functions, *modules))


def parse_component_digest() -> str:
    """The digest of the general extractor's reply-to-claim code and its config.

    Exercised by every extraction call, and on every extraction's ``always``
    list, so a change to how a reply becomes a claim (a parser rule, the
    confidence clamp, a filter, a classifier's config) selects every snapshot
    the general extractor made. Over-selecting is the price of not having to
    name which snapshots a parser change reaches.
    """
    cfg = get_config()
    return component_digest(
        *_parse_code_texts(),
        f"confidence_clamp={_CONFIDENCE_FLOOR},{_CONFIDENCE_CEILING}",
        *(
            section.model_dump_json()
            for section in (
                cfg.extraction_scope,
                cfg.extraction_modality,
                cfg.extraction_polarity,
                cfg.extraction_stance,
                cfg.extraction_validity,
                cfg.structured_claim,
            )
        ),
    )


def routing_component_digest() -> str:
    """The digest of the settings that decide which source-dependent components a source gets.

    Which source types get the tool-turn marker or the working-state filter,
    whether an append-only snapshot is read as a delta, where the chunking
    threshold sits, whether a PDF page may go to the vision provider. A record
    cannot say whether a source it never routed somewhere would go there now,
    so a change here selects every snapshot.
    """
    ext = get_config().extraction
    vision = get_config().extraction_vision
    return component_digest(
        "tool_turn_source_types=" + ",".join(sorted(ext.tool_turn_source_types)),
        "session_state_source_types=" + ",".join(sorted(ext.session_state_source_types)),
        f"append_only_delta={ext.append_only_delta}",
        f"html_chunk_size={ext.html_chunk_size}",
        f"vision_enabled={vision.enabled}",
        f"vision_trigger={vision.trigger}",
    )


def _record_path(name: str) -> None:
    """Record the extraction path (or the vision channel) this extraction took."""
    record_component(name, path_component_digests()[name])


def general_component_table() -> ComponentTable:
    """Every component the general extractor could exercise under the current config.

    ``always`` is the prompt every extraction sends (the core rules, the
    reference rule, each config-enabled rule, the output schema), the
    reply-to-claim code (:func:`parse_component_digest`) and the routing
    settings (:func:`routing_component_digest`). The tool-turn and
    append-context rules, the paths and the vision channel depend on the source
    and are in the table but not in ``always``.
    """
    base = _configured_prompt_components()
    gated = [
        c
        for c in _configured_prompt_components(tool_turns_present=True, context_present=True)
        if c.name not in {b.name for b in base}
    ]
    digests = {c.name: c.digest for c in [*base, *gated]}
    digests.update(path_component_digests())
    digests[CODE_PARSE] = parse_component_digest()
    digests[ROUTING] = routing_component_digest()
    always = {c.name for c in base} | {CODE_PARSE, ROUTING}
    return ComponentTable(digests=digests, always=frozenset(always))


def _extraction_response_schema(
    *,
    scope_enabled: bool,
    modality_enabled: bool,
    polarity_enabled: bool = False,
    stance_enabled: bool = False,
    validity_enabled: bool = False,
    structure_enabled: bool = False,
) -> dict[str, Any]:
    """The JSON Schema of the extraction reply — the candidate array.

    The machine-checkable half of :data:`_EXTRACT_SCHEMA`, passed as
    ``response_schema`` so a schema-enforcing adapter (the
    ``LocalProvider`` under ``llm.local.structured_output: auto``) can pin the
    array shape instead of hoping for it; :func:`_parse_extraction_response`
    remains the tolerant backstop. Optional fields mirror the enabled prompt
    clauses so the schema and prompt can never disagree about the dialect.
    """
    properties: dict[str, Any] = {
        "content": {"type": "string"},
        "subjects": {"type": "array", "items": {"type": "string"}},
        "confidence_value": {"type": "number"},
        "uncertainty_nature": {"type": "string", "enum": ["EPISTEMIC", "ALEATORY"]},
    }
    if scope_enabled:
        properties["scope"] = {"type": "string", "enum": ["WORLD", SCOPE_DOCUMENT_META]}
    if modality_enabled:
        properties["assertion_modality"] = {
            "type": "string",
            "enum": [m.value for m in AssertionModality],
        }
    if polarity_enabled:
        properties["polarity"] = {
            "type": "string",
            "enum": [POLARITY_ASSERTED, *sorted(NON_ASSERTED_POLARITIES)],
        }
    if stance_enabled:
        properties["stance_kind"] = {
            "type": ["string", "null"],
            "enum": ["ENDORSES", "DISPUTES", None],
        }
        properties["stance_target"] = {"type": ["integer", "null"]}
        properties["stance_magnitude"] = {"type": ["number", "null"]}
    if validity_enabled:
        properties["valid_until"] = {"type": ["string", "null"]}
        properties["validity_confidence"] = {"type": ["number", "null"]}
        properties["validity_basis"] = {"type": ["string", "null"]}
    if structure_enabled:
        properties["structured_claim"] = {
            "type": ["object", "null"],
            "properties": {
                "subject": {"type": "string"},
                "predicate": {"type": "string"},
                "object": {"type": "string"},
            },
            "required": ["subject", "predicate", "object"],
            "additionalProperties": False,
        }
    return {
        "type": "array",
        "items": {
            "type": "object",
            "properties": properties,
            "required": ["content", "subjects", "confidence_value", "uncertainty_nature"],
            "additionalProperties": False,
        },
    }


@dataclass
class CandidateParticle:
    """An extractor's proposed particle before conflict resolution and storage.

    Attributes:
        content: The claim text.
        confidence_value: Self-assessed confidence in [0, 1].
        uncertainty_nature: EPISTEMIC or ALEATORY.
        subjects: Subject names/QIDs this claim is about; resolved to UUIDs by the pipeline.
        properties: Nomisma ontology-keyed structured data; None for free-text.
            Also carries the document-scope tag (``scope`` /
            ``scope_action``) for DOCUMENT_META candidates.
        subject_classes: Maps subject name → Nomisma class applied after subject resolution.
    """

    content: str
    confidence_value: float
    uncertainty_nature: UncertaintyNature
    subjects: list[str] = field(default_factory=list)
    # truth-aptness, classified at extraction. Default FALSIFIABLE so
    # structured extractors and the disabled case are unchanged.
    assertion_modality: AssertionModality = AssertionModality.FALSIFIABLE
    # stance reification. When ``stance_kind`` is set the candidate
    # is a stance — the pipeline binds it to the sibling candidate at
    # ``stance_target_index`` via an ENDORSES/DISPUTES edge and stamps
    # ``stance:holder`` (the source author) + ``stance:magnitude``. ``None`` ⇒
    # not a stance (the default and the disabled case).
    stance_kind: RelationType | None = None
    stance_target_index: int | None = None
    stance_magnitude: float | None = None
    # journal-aware NARRATIVE emission. ``particle_type`` lets a
    # candidate be the entry-level NARRATIVE (connective tissue); the
    # journal extractor emits exactly one. ``narrative_index`` is a constituent
    # claim's 0-based position in entry order — set on each claim that belongs
    # to the narrative, ``None`` on the NARRATIVE container and on every
    # non-journal candidate. The pipeline writes PART_OF (constituent →
    # narrative) and SEQUENCE_IN (predecessor → successor) from these once ids
    # exist, modelled on the stance edge-writer.
    particle_type: ParticleType = ParticleType.CLAIM
    narrative_index: int | None = None
    # event-anchored validity boundary, populated by the parser's
    # three-condition gate (explicit cue + confidence floor + future date) when
    # ``extraction_validity.enabled``. ``None`` (the default and the disabled
    # case) means the claim carries no boundary and keeps its decay treatment.
    # Passed straight through ``candidate_to_particle`` to ``Particle.valid_until``.
    valid_until: datetime | None = None
    # the derived S-P-O annotation, parsed from the same reply and
    # already stamped with the producing extractor's id + version. Its terms are
    # still *unresolved* — the Engine binds the subject term to a Subject UUID
    # in ``ingest.pipeline`` once subject resolution has run. ``None`` (the
    # default, the disabled case, and any parse failure) leaves the particle
    # un-annotated, which is a legal permanent state.
    structured_claim: StructuredClaim | None = None
    # which half of the prose/structure pair is the assertion.
    # ``PROSE`` (the default) for every LLM-driven extractor — ``content`` is
    # asserted and the triple annotates it. ``STRUCTURED`` is set by a
    # structure-native parser (the RDF extractor) whose triple *is* the
    # assertion and whose ``content`` is a derived verbalization.
    canonical_form: CanonicalForm = CanonicalForm.PROSE
    properties: dict[str, object] | None = None
    subject_classes: dict[str, str] = field(default_factory=dict)
    external_refs: dict[str, ExternalRef] = field(default_factory=dict)
    # subject name → non-entity gate class, for the names the gate
    # kept as *qualified* (scoped by the source's project) rather than
    # suppressed. Written by the gate in the Engine pipeline, never by an
    # extractor; empty on every candidate the gate did not qualify.
    qualified_subjects: dict[str, str] = field(default_factory=dict)
    # SHA-256 of the LLM-prompt text that produced this candidate.
    # Set by ``extract_with_carry_forward``; passed through to the
    # ``SOURCE`` ProvenanceRef so the next re-extraction can carry the
    # particle forward when the chunk's text hasn't changed.
    chunk_hash: str | None = None
    # SHA-256 fingerprint of the ACTIVE-particle baseline at the start of
    # the extraction run that produced this candidate (Extension C).
    # Stamped by ``extract_snapshot()`` after the extractor returns; left
    # ``None`` by extractors themselves.
    context_fingerprint: str | None = None
    # The ``"<provider>:<model>"`` pairing that produced this candidate.
    # Stamped by the completion seam itself — ``_call_llm`` here
    # and ``journal._call_journal_llm`` — not by the extractor and not by the
    # pipeline, so a deterministic extractor (rdf, numista, wikidata, …) never
    # sets it and its particles are correctly unstamped. Same "written by the
    # machinery, not the extractor" class as ``chunk_hash`` above.
    provider_model: str | None = None
    # Four fields a *migration* extractor needs and an LLM extractor
    # never sets, so every existing producer is unaffected by their defaults:
    #   tags                — Extension C tags to stamp on the particle. The
    #                         reference-memory migration uses them to carry the
    #                         store vocabulary, so a migrated record
    #                         and a façade-written one are the same record.
    #   contributors        — the attributed act: who imported
    #                         this, with role ``importer``.
    #   calibration_source  — declared by the extractor when the confidence is
    #                         not a model output. Set ⇒ ``candidate_to_particle``
    #                         uses it verbatim and does **not** apply temperature
    #                         scaling, which would be meaningless over a value no
    #                         model produced.
    #   provenance_location — the record's position inside the deposited blob,
    #                         carried onto the SOURCE ``ProvenanceRef.location``.
    #                         This is what makes a migrated claim's provenance
    #                         checkable rather than merely asserted.
    tags: list[str] | None = None
    contributors: list[ContributorRef] | None = None
    calibration_source: CalibrationSource | None = None
    provenance_location: str | None = None


@dataclass
class PageStat:
    """Extraction statistics for a single PDF page or HTML chunk."""

    page_number: int
    candidate_count: int


ChunkStatus = Literal["carried", "answered", "failed", "capped"]


@dataclass
class ChunkOutcome:
    """What happened to one chunk of a chunked read, in chunk order.

    ``status`` is ``carried`` (skipped by carry-forward, its claims already
    exist), ``answered`` (the model replied), ``failed`` (a transient failure)
    or ``capped`` (skipped at ``max_llm_calls_per_source``). ``start`` is the
    chunk's position in the text the read cut it from: the extraction text for
    a delta chunk, the normalised text for a whole read's. The pipeline uses
    the outcomes to decide what a partly failed read keeps.
    """

    chunk_id: str
    prompt_hash: str
    status: ChunkStatus
    start: int | None = None


@dataclass
class ExtractionResult:
    """The complete output of one extractor run.

    Attributes:
        candidates: Proposed particles ready for conflict resolution.
        quality_notes: Human-readable notes about extraction quality or errors.
        page_stats: Per-page/chunk statistics (PDF and HTML chunked extractors).
    """

    candidates: list[CandidateParticle] = field(default_factory=list)
    quality_notes: list[str] = field(default_factory=list)
    page_stats: list[PageStat] = field(default_factory=list)
    # Count of LLM calls (single-pass, per-chunk, or per-page) that failed with
    # a transient API error — rate limit, billing, 5xx, network. Aggregated
    # across every chunk/page so the pipeline can decide retry-vs-COMPLETE
    # *structurally*, never by string-matching ``quality_notes`` (whose
    # "API error: …" text the chunked/PDF paths prefix with chunk/page labels —
    # the F4.1 silent-loss bug). >0 ⇒ ``extract_snapshot`` resets the snapshot
    # to PENDING for ``extract --all-pending`` retry and writes nothing from the
    # pass, so the retry sends every call again: carry-forward cannot
    # skip a chunk whose claims were never written.
    transient_error_count: int = 0
    # Count of LLM calls (per chunk or per page) that answered, on the paths
    # that make more than one call. When the pass also had a transient failure,
    # these are the calls a retry sends, and pays for, a second time.
    answered_calls: int = 0
    # IDs of existing ACTIVE particles eligible for carry-forward.
    # Populated by ``extract_with_carry_forward`` when a chunk's text hashes
    # match an existing particle's recorded ``chunk_hash``. The reindex
    # operation reads this list and excludes the named particles from
    # supersession.
    carry_forward_ids: list[str] = field(default_factory=list)
    # ``chunk_id`` of every chunk ``extract_with_carry_forward`` skipped at the
    # ``max_llm_calls`` cap (``CHUNK_TRUNCATION``), in chunk order.
    unread_chunk_ids: list[str] = field(default_factory=list)
    # where the read stopped, in characters of the extraction
    # text (:func:`extraction_text`), when it stopped short of the end: the
    # start of the first chunk the call cap skipped. ``None`` when the read
    # reached the end, the common case. The pipeline turns it into the
    # snapshot's ``extracted_through`` byte offset.
    read_through: int | None = None
    # the snapshot was read as a delta after the prefix the
    # pipeline handed over (``append_prefix``).
    append_delta: bool = False
    # why a handed-over prefix was not used, when it was not.
    # The extraction then ran as a whole read.
    append_fallback: str | None = None
    # one outcome per chunk of a chunked read, in chunk order,
    # with each chunk's start once the extractor fills it in. Empty for a
    # single-call read and for every path that does not chunk through
    # ``extract_with_carry_forward``.
    chunk_outcomes: list[ChunkOutcome] = field(default_factory=list)


def extraction_text(content: bytes, *, is_markdown: bool, mark_tools: bool) -> tuple[str, int]:
    """The text the general extractor reads from ``content``, and its tool-turn count.

    ``content_to_text``, then the Obsidian frontmatter strip for a Markdown
    source, then tool-turn marking for a transcript. One
    function so the extractor and the pipeline decode alike: the delta read
    compares a prefix decoded here with the whole snapshot decoded
    here, and maps a stop point in this text back to raw bytes.
    """
    text = content_to_text(content)
    if is_markdown:
        _, text = _strip_obsidian_frontmatter(text)
    tool_turns = 0
    if mark_tools:
        text, tool_turns = mark_tool_turns(text)
    return text, tool_turns


def source_text_flags(source_type: object) -> tuple[bool, bool]:
    """``(is_markdown, mark_tools)`` for a source type, as the general extractor reads it."""
    is_markdown = isinstance(source_type, str) and source_type == "LOCAL_MARKDOWN"
    mark_tools = isinstance(source_type, str) and (
        source_type in get_config().extraction.tool_turn_source_types
    )
    return is_markdown, mark_tools


@dataclass
class NormalizedDocument:
    """Output of the source-format-parsing stage of an LLM-driven extractor.

    The convention splits the ``extract()`` of an
    LLM-driven extractor into two private methods:

    * ``_normalise(content, snapshot) -> NormalizedDocument`` — source-format
      parsing only. Walks domain JSON / HTML, builds the prose chunk units
      that the LLM will see, and surfaces the metadata downstream code needs
      to know about the source (author, publication time, quality issues,
      domain-injected subjects).
    * ``_extract_claims(doc, **kwargs) -> ExtractionResult`` — LLM claim
      extraction only. The default body is one line:
      ``extract_with_carry_forward(doc.chunks, ...)``.

    ``extract()`` becomes a two-line composition that calls them in order.

    This is a **pattern**, not a Protocol extension. ``ExtractorPlugin``
    stays exactly as it was defined. Structured extractors (Wikidata,
    Nomisma, the three Numista variants) have no prose stage and override
    ``extract()`` directly without going through this convention.
    §"Decision" item 3 for the full rationale.

    Attributes:
        chunks: Ordered ``ChunkUnit``s ready for ``extract_with_carry_forward``.
        author_id: Author identity for UGC sources (e.g. ``"github:karpathy"``,
            ``"reddit:u/throwaway_investor"``); ``None`` for non-UGC sources.
            Typically already on the ``Snapshot`` (set by the importer); the
            field is repeated here so an ``_extract_claims`` implementation
            can read it without round-tripping through the corpus store.
        content_published_at: Source publication timestamp; drives
            content age decay at query time. ``None`` if the source does not
            expose a publication date.
        quality_notes: Human-readable warnings about the parsed document
            (truncations, malformed sections, missing metadata) that the
            extractor wants to surface alongside any particles it produces.
        injected_subjects: Subject names that domain extractors inject so the
            LLM's particle content can reference them by short form (e.g.
            ``["gh/karpathy"]`` for a GitHub gist, normalised back to
            ``"@karpathy"`` post-extraction).
    """

    chunks: list[ChunkUnit] = field(default_factory=list)
    author_id: str | None = None
    content_published_at: datetime | None = None
    quality_notes: list[str] = field(default_factory=list)
    injected_subjects: list[str] = field(default_factory=list)


class GeneralExtractor:
    """General-purpose LLM extractor. Accepts any source type as a fallback.

    PDFs are extracted page-by-page. HTML is converted to Markdown
    and processed in 15 K-character chunks. All other content types
    are processed in a single LLM call. Requires ``ANTHROPIC_API_KEY``.
    """

    EXTRACTOR_ID: str = EXTRACTOR_ID
    EXTRACTOR_VERSION: str = EXTRACTOR_VERSION
    DEFAULT_TRUST_WEIGHT: float = DEFAULT_TRUST_WEIGHT
    APPLICABILITY = APPLICABILITY

    def accepts(self, source_type: str) -> bool:  # noqa: ARG002
        return True

    async def extract(
        self,
        snapshot: Snapshot,
        content: bytes,
        **kwargs: object,
    ) -> ExtractionResult:
        """Extract claim-granularity particles from snapshot content.

        PDF sources use per-page extraction; all other sources use
        a single-pass call after HTML-to-Markdown preprocessing.
        ``LOCAL_MARKDOWN`` sources have their Obsidian YAML frontmatter
        stripped before extraction so the LLM does not extract
        metadata-key claims from it.

        When the decoded text exceeds the chunked-extraction threshold, the
        HTML path routes through ``extract_with_carry_forward``.
        That helper consults the particle store for prior chunks whose hash
        matches the current chunk's text, so ``session`` and
        ``corpus_entry_id`` are read from kwargs (the pipeline already
        passes them).
        """
        from sqlalchemy.ext.asyncio import AsyncSession as _Session

        # the pipeline threads a CompletionPool only for a
        # latency-tolerant caller (the consolidation extract pass); its
        # presence routes the text paths through pooled batch dispatch.
        # Deferred import: lazy-init of the LLM stack (AGENTS.md case 2),
        # mirroring ``_call_llm``.
        from particles.llm import CompletionPool as _Pool

        pool_obj = kwargs.get("completion_pool")
        completion_pool: CompletionPool | None = pool_obj if isinstance(pool_obj, _Pool) else None

        # the anchor for resolving relative validity boundaries
        # ("tomorrow", "until Q3") is the source's publication instant when the
        # source exposes one, else the extraction wall-clock (applied in
        # ``_call_llm``). Threaded down so every LLM path resolves relative dates
        # against the same instant.
        reference_published_at = snapshot.content_published_at
        # the raw prefix of an APPEND_ONLY snapshot the store has
        # already extracted, handed over by the pipeline. Only the text paths
        # read a delta; a binary source says why it did not.
        prefix_obj = kwargs.get("append_prefix")
        append_prefix: bytes | None = prefix_obj if isinstance(prefix_obj, bytes) else None
        # what this extractor could exercise, so a later table can
        # tell a component this snapshot skipped from one it never knew.
        declare_components(general_component_table())
        record_component(ROUTING, routing_component_digest())
        if content.startswith(b"%PDF"):
            _record_path(PATH_PDF_PAGES)
            result = await self._extract_pdf_paged(
                content,
                reference_published_at=reference_published_at,
                completion_pool=completion_pool,
            )
            return _binary_fallback(result, append_prefix)
        image_media_type = sniff_image_media_type(content)
        if image_media_type is not None:
            _record_path(PATH_IMAGE)
            result = await self._extract_image(
                content, image_media_type, reference_published_at=reference_published_at
            )
            return _binary_fallback(result, append_prefix)
        source_type = kwargs.get("source_type")
        # a transcript's tool turns are relabelled before the model
        # sees them, so tool output is never extracted as the speaker's fact.
        is_markdown, mark_tools = source_text_flags(source_type)
        session_obj = kwargs.get("session")
        session: AsyncSession | None = session_obj if isinstance(session_obj, _Session) else None
        entry_obj = kwargs.get("corpus_entry_id")
        corpus_entry_id: str | None = entry_obj if isinstance(entry_obj, str) else None
        # Reindex threads its supersede set so carry-forward treats the
        # marked particles as absent (see extract_with_carry_forward).
        sup_obj = kwargs.get("supersede_ids")
        supersede_ids: frozenset[str] = sup_obj if isinstance(sup_obj, frozenset) else frozenset()
        # on a partial read's retry, the chunk hashes this
        # snapshot's own claims carry, which carry-forward skips whatever
        # extractor version wrote them.
        resume_obj = kwargs.get("resume_carried")
        resume_carried: Mapping[str, Sequence[str]] | None = (
            resume_obj if isinstance(resume_obj, Mapping) else None
        )
        result = await self._extract_single_pass(
            content,
            is_markdown=is_markdown,
            mark_tools=mark_tools,
            session=session,
            corpus_entry_id=corpus_entry_id,
            reference_published_at=reference_published_at,
            completion_pool=completion_pool,
            resume_carried=resume_carried,
            supersede_ids=supersede_ids,
            append_prefix=append_prefix,
        )
        # An agent session's worktree / cwd / branch is true for minutes, never
        # a durable belief. Filtered here, after every text path returns, so
        # the single-pass, chunked, and pooled routes are all covered.
        if isinstance(source_type, str) and (
            source_type in get_config().extraction.session_state_source_types
        ):
            _drop_session_state_candidates(result, content)
        return result

    async def _extract_single_pass(
        self,
        content: bytes,
        *,
        is_markdown: bool = False,
        mark_tools: bool = False,
        session: AsyncSession | None = None,
        corpus_entry_id: str | None = None,
        reference_published_at: datetime | None = None,
        completion_pool: CompletionPool | None = None,
        supersede_ids: frozenset[str] = frozenset(),
        append_prefix: bytes | None = None,
        resume_carried: Mapping[str, Sequence[str]] | None = None,
    ) -> ExtractionResult:
        """Single LLM call for non-PDF sources.

        Chunks if content exceeds HTML_CHUNK_SIZE.
        When ``is_markdown`` is True, any leading Obsidian YAML frontmatter
        (``---\\nkey: value\\n---\\n``) is stripped from the decoded text
        before extraction so the LLM doesn't manufacture "the source has
        a `tags` field" claims from metadata noise.

        With ``append_prefix`` the text after the decoded prefix
        is read as a delta, on either side of the chunk threshold; a prefix
        whose decoding does not prefix this text falls back to the whole read.
        """
        try:
            text, tool_turns = extraction_text(
                content, is_markdown=is_markdown, mark_tools=mark_tools
            )
        except Exception as exc:
            return ExtractionResult(quality_notes=[f"Decode error: {exc}"])

        fallback: str | None = None
        if append_prefix is not None:
            from particles.extraction.append_delta import decoded_prefix_length

            prefix_len, fallback = decoded_prefix_length(
                append_prefix, text, is_markdown=is_markdown, mark_tools=mark_tools
            )
            if prefix_len is not None:
                return await self._extract_append_delta(
                    text,
                    prefix_len,
                    tool_turns=tool_turns,
                    session=session,
                    corpus_entry_id=corpus_entry_id,
                    reference_published_at=reference_published_at,
                    completion_pool=completion_pool,
                    supersede_ids=supersede_ids,
                    resume_carried=resume_carried,
                )

        if not text.strip():
            return ExtractionResult(quality_notes=["Empty content"], append_fallback=fallback)

        result = await self._extract_whole(
            text,
            tool_turns=tool_turns,
            session=session,
            corpus_entry_id=corpus_entry_id,
            reference_published_at=reference_published_at,
            completion_pool=completion_pool,
            supersede_ids=supersede_ids,
            resume_carried=resume_carried,
        )
        result.append_fallback = fallback
        return result

    async def _extract_whole(
        self,
        text: str,
        *,
        tool_turns: int,
        session: AsyncSession | None,
        corpus_entry_id: str | None,
        reference_published_at: datetime | None,
        completion_pool: CompletionPool | None,
        supersede_ids: frozenset[str],
        resume_carried: Mapping[str, Sequence[str]] | None = None,
    ) -> ExtractionResult:
        """Today's whole read of ``text``: one call, or chunks above the threshold."""
        cfg = get_config().extraction

        if len(text) > cfg.html_chunk_size:
            _record_path(PATH_CHUNKED)
            return await self._extract_html_chunked(
                text,
                session=session,
                corpus_entry_id=corpus_entry_id,
                reference_published_at=reference_published_at,
                completion_pool=completion_pool,
                supersede_ids=supersede_ids,
                tool_turns=tool_turns,
                resume_carried=resume_carried,
            )

        _record_path(PATH_SINGLE_PASS)
        log.info("Sending %d characters to extractor model", len(text))
        log.debug("Source text (first 500 chars):\n%s", text[:500])
        if completion_pool is not None:
            # a single-chunk source rides the merged nightly
            # batch too — a one-request group is what lets it clear the
            # ``min_requests`` gate the pooled set is tested against.
            planned = _build_llm_request(
                text,
                reference_published_at=reference_published_at,
                tool_turns_present=tool_turns > 0,
            )
            [(candidates, notes, transient)] = await _pooled_extract(completion_pool, [planned])
        # pass the anchor only when set, so the no-anchor call is
        # byte-identical to the pre-0197 signature (see _extract_html_chunked).
        elif reference_published_at is not None:
            candidates, notes, transient = await _call_llm(
                text,
                reference_published_at=reference_published_at,
                tool_turns_present=tool_turns > 0,
            )
        else:
            candidates, notes, transient = await _call_llm(text, tool_turns_present=tool_turns > 0)
        if tool_turns:
            # Never silent (the disclosure habit): the operator can
            # see that the source carried tool output and how much.
            notes = [*notes, f"tool output: {tool_turns} turn(s) marked unverified"]
        return ExtractionResult(
            candidates=candidates,
            quality_notes=notes,
            transient_error_count=1 if transient else 0,
        )

    async def _extract_html_chunked(
        self,
        text: str,
        *,
        session: AsyncSession | None = None,
        corpus_entry_id: str | None = None,
        reference_published_at: datetime | None = None,
        completion_pool: CompletionPool | None = None,
        supersede_ids: frozenset[str] = frozenset(),
        tool_turns: int = 0,
        resume_carried: Mapping[str, Sequence[str]] | None = None,
    ) -> ExtractionResult:
        """Paragraph-chunked HTML extraction with carry-forward.

        The decoded text is normalised to absorb cosmetic noise that
        otherwise drifts between re-deposits (Wikipedia ``[edit]`` markers,
        page footers), split on paragraph boundaries — falling back to line
        breaks inside long paragraphs and to a hard cut at chunk size for
        pathological one-line inputs — and routed through
        :func:`extract_with_carry_forward`. Chunks whose
        SHA-256 hash matches a prior particle's ``chunk_hash`` for the same
        corpus entry and extractor identity skip the LLM call entirely.

        ``session`` and ``corpus_entry_id`` are threaded through from the
        pipeline. When either is None (unit tests driving the extractor
        directly), the helper degrades to a plain per-chunk LLM loop.

        ``tool_turns`` is the count :func:`mark_tool_turns` relabelled in
        ``text``; when non-zero each chunk carrying a relabelled turn gets the
        tool-turn rule, as the single-pass call does.
        """
        # Deferred import: incremental imports CandidateParticle from this
        # module; hoisting would create a cycle (legitimate case 1).
        # case.
        from particles.extraction.incremental import ChunkUnit, extract_with_carry_forward

        cfg = get_config().extraction
        normalised = _normalise_for_hashing(text)
        chunk_texts = _split_into_paragraph_chunks(normalised, cfg.html_chunk_size)
        chunks = [
            ChunkUnit(chunk_id=f"chunk_{i}", chunk_text=t)
            for i, t in enumerate(chunk_texts, start=1)
        ]
        log.info(
            "GeneralExtractor: %d HTML chunk(s) after paragraph splitting"
            " (input %d chars, normalised %d chars)",
            len(chunks),
            len(text),
            len(normalised),
        )
        # thread the reference anchor so relative validity boundaries
        # resolve against the source's publication instant on the chunked path
        # too. The helper binds it onto its default ``_call_llm`` (keeping that
        # call patchable as the ``incremental._call_llm`` test seam); an injected
        # ``call_llm`` (the journal extractor) is used unchanged.
        result = await extract_with_carry_forward(
            session=session,
            chunks=chunks,
            corpus_entry_id=corpus_entry_id,
            extractor_id=EXTRACTOR_ID,
            extractor_version=EXTRACTOR_VERSION,
            max_llm_calls=cfg.max_llm_calls_per_source,
            reference_published_at=reference_published_at,
            completion_pool=completion_pool,
            supersede_ids=supersede_ids,
            tool_turns_present=tool_turns > 0,
            resume_carried=resume_carried,
        )
        if tool_turns:
            # Same disclosure as the single-pass branch.
            result.quality_notes.append(f"tool output: {tool_turns} turn(s) marked unverified")
        # a read cut at the call cap stopped where the first
        # skipped chunk starts. The chunks are cut from the normalised text,
        # which only ever deletes characters from ``text``, so an offset in it
        # is a lower bound on the same point in ``text``: the next read may
        # re-read a few characters, and never skips any.
        spans = paragraph_spans(normalised, cfg.html_chunk_size)
        if result.unread_chunk_ids:
            first = next(i for i, c in enumerate(chunks) if c.chunk_id in result.unread_chunk_ids)
            result.read_through = spans[first][0]
        # each chunk's start, in the normalised text it was cut
        # from, so the pipeline can place a partly failed read's stop.
        _set_outcome_starts(result, [start for start, _ in spans])
        _finish_chunk_stats(chunks, result)
        return result

    async def _extract_append_delta(
        self,
        text: str,
        prefix_len: int,
        *,
        tool_turns: int,
        session: AsyncSession | None,
        corpus_entry_id: str | None,
        reference_published_at: datetime | None,
        completion_pool: CompletionPool | None,
        supersede_ids: frozenset[str],
        resume_carried: Mapping[str, Sequence[str]] | None = None,
    ) -> ExtractionResult:
        """Read only ``text[prefix_len:]``, the delta, with context before it.

        The delta is split on paragraph boundaries into chunks of at most
        ``extraction.append_chunk_chars``. Each chunk carries a labelled context
        section: the last ``extraction.append_context_chars`` of the text
        before the delta for the first chunk, the end of the chunk before it
        for every later one, cut back to a paragraph boundary. The chunks go
        through :func:`extract_with_carry_forward`, so the call cap, the
        pooled path and carry-forward within the delta behave as on
        a whole read; each chunk hash covers its context too.
        """
        from particles.extraction.append_delta import plan_delta_chunks
        from particles.extraction.incremental import ChunkUnit, extract_with_carry_forward

        _record_path(PATH_APPEND_DELTA)

        cfg = get_config().extraction
        planned = plan_delta_chunks(
            text,
            prefix_len,
            chunk_chars=cfg.append_chunk_chars,
            context_chars=cfg.append_context_chars,
        )
        chunks = [
            ChunkUnit(chunk_id=f"delta_{i}", chunk_text=p.text, context=p.context)
            for i, p in enumerate(planned, start=1)
        ]
        log.info(
            "GeneralExtractor: append-only delta of %d chars after %d already read, %d chunk(s)",
            len(text) - prefix_len,
            prefix_len,
            len(chunks),
        )
        if not chunks:
            return ExtractionResult(
                quality_notes=["append-only delta: no new text since the base"],
                append_delta=True,
            )
        # Tool turns are counted over the whole text; a chunk gets the rule only
        # when its own text or context carries a relabelled turn.
        result = await extract_with_carry_forward(
            session=session,
            chunks=chunks,
            corpus_entry_id=corpus_entry_id,
            extractor_id=EXTRACTOR_ID,
            extractor_version=EXTRACTOR_VERSION,
            max_llm_calls=cfg.max_llm_calls_per_source,
            reference_published_at=reference_published_at,
            completion_pool=completion_pool,
            supersede_ids=supersede_ids,
            tool_turns_present=tool_turns > 0,
            resume_carried=resume_carried,
        )
        result.append_delta = True
        if result.unread_chunk_ids:
            first = next(i for i, c in enumerate(chunks) if c.chunk_id in result.unread_chunk_ids)
            result.read_through = planned[first].start
        # each chunk's start, in the extraction text.
        _set_outcome_starts(result, [p.start for p in planned])
        _finish_chunk_stats(chunks, result)
        return result

    async def _extract_pdf_paged(
        self,
        content: bytes,
        *,
        reference_published_at: datetime | None = None,
        completion_pool: CompletionPool | None = None,
    ) -> ExtractionResult:
        """One LLM call per PDF page.

        Pages are processed sequentially. The last PDF_PAGE_OVERLAP_LINES lines
        of each page are prepended to the next page to handle entries that straddle
        a page boundary.

        When ``extraction_vision.enabled``, a page judged *visual*
        (embedded images, low/no text, or under the ``always`` trigger) is sent
        to the vision-capable provider as one multimodal call — the page text
        plus a rendered image of the page — instead of the text-only call, so
        diagrams / charts / scanned pages are extracted. Vision tokens are paid
        only on visual pages, capped by ``extraction_vision.max_pages``.

        With a ``completion_pool`` the pooled variant runs instead;
        this sequential loop is deliberately untouched by that path so the
        interactive behaviour — including the wall-clock budget covering LLM
        time — stays byte-for-byte what it was.
        """
        if completion_pool is not None:
            return await self._extract_pdf_paged_pooled(
                content,
                reference_published_at=reference_published_at,
                completion_pool=completion_pool,
            )
        import io

        from pypdf import PdfReader

        from particles.llm import VisionImage

        cfg = get_config().extraction
        vcfg = get_config().extraction_vision
        pdf_page_overlap_lines = cfg.pdf_page_overlap_lines
        max_pages = cfg.max_pdf_pages
        max_page_chars = cfg.max_pdf_page_chars
        max_seconds = cfg.max_pdf_seconds

        # Open the rasterizer once when vision is on; raises an actionable
        # ImportError if the [vision] extra is missing. None when
        # vision is off, leaving the text-only path byte-for-byte unchanged.
        pdfium_doc = _open_pdfium_for_vision(content) if vcfg.enabled else None
        vision_pages_used = 0

        reader = PdfReader(io.BytesIO(content))
        page_count = len(reader.pages)
        # Security: a hostile PDF can carry an enormous page count. Cap the
        # pages we process; record the truncation as a quality note.
        total = min(page_count, max_pages)

        all_candidates: list[CandidateParticle] = []
        page_stats: list[PageStat] = []
        all_notes: list[str] = []
        if page_count > max_pages:
            all_notes.append(
                f"PDF_PAGE_CAP: document has {page_count} pages; "
                f"only the first {max_pages} were extracted (extraction.max_pdf_pages)"
            )
            log.warning(
                "PDF has %d pages; capping extraction at %d (max_pdf_pages)",
                page_count,
                max_pages,
            )
        transient_errors = 0
        answered_calls = 0
        prev_tail: list[str] = []
        started = time.monotonic()

        for idx, page in enumerate(reader.pages):
            page_num = idx + 1
            if page_num > max_pages:
                break
            # Security: a wall-clock budget bounds total work a malicious or
            # pathological PDF can force (slow per-page parsing, many LLM
            # calls). Stop cleanly and note how far we got.
            if time.monotonic() - started > max_seconds:
                all_notes.append(
                    f"PDF_TIME_CAP: extraction exceeded {max_seconds:.0f}s wall-clock "
                    f"budget after page {idx}; remaining pages skipped "
                    "(extraction.max_pdf_seconds)"
                )
                log.warning(
                    "PDF extraction exceeded %.0fs budget after page %d; stopping",
                    max_seconds,
                    idx,
                )
                break
            page_text = page.extract_text() or ""
            # Security: bound a single page's extracted text — a crafted page
            # can decompress to enormous strings.
            if len(page_text) > max_page_chars:
                all_notes.append(
                    f"PDF_PAGE_CHARS_CAP: page {page_num} extracted text truncated "
                    f"to {max_page_chars} chars (extraction.max_pdf_page_chars)"
                )
                page_text = page_text[:max_page_chars]

            context = (
                ("\n".join(prev_tail) + "\n" + page_text).strip()
                if prev_tail
                else page_text.strip()
            )

            # decide this page's modality. A visual page (embedded
            # images, low/no text, or ``always``) gets a rendered page image so
            # a multimodal call can read its figures — including scanned pages
            # whose text is empty, which the text-only path would skip below.
            images: list[VisionImage] | None = None
            if vcfg.enabled and _page_is_visual(
                page,
                page_text,
                trigger=vcfg.trigger,
                low_text_threshold=vcfg.low_text_threshold,
            ):
                if vision_pages_used < vcfg.max_pages:
                    try:
                        png = _render_page_png(pdfium_doc, idx, vcfg.render_dpi)
                        images = [VisionImage(media_type="image/png", data=png)]
                        vision_pages_used += 1
                    except Exception as exc:
                        all_notes.append(f"VISION_RENDER_FAILED: page {page_num}: {exc}")
                        log.warning("Vision render failed for page %d: %s", page_num, exc)
                else:
                    all_notes.append(
                        f"VISION_PAGE_CAP: page {page_num} is image-bearing but the "
                        f"per-document vision-page budget ({vcfg.max_pages}) is exhausted; "
                        "extracted text-only (extraction_vision.max_pages)"
                    )

            if not context and not images:
                page_stats.append(PageStat(page_number=page_num, candidate_count=0))
                all_notes.append(f"ZERO_PAGE_YIELD: page {page_num} had no extractable text")
                prev_tail = []
                continue

            log.info(
                "Extracting PDF page %d/%d (%d chars%s)",
                page_num,
                total,
                len(context),
                ", +vision" if images else "",
            )

            if reference_published_at is not None:
                candidates, notes, transient = await _call_llm(
                    context, images=images, reference_published_at=reference_published_at
                )
            else:
                candidates, notes, transient = await _call_llm(context, images=images)
            count = len(candidates)

            if transient:
                transient_errors += 1
            else:
                answered_calls += 1
            if count == 0:
                all_notes.append(f"ZERO_PAGE_YIELD: page {page_num} produced 0 particles")
            if notes:
                all_notes.extend(f"Page {page_num}: {n}" for n in notes)

            all_candidates.extend(candidates)
            page_stats.append(PageStat(page_number=page_num, candidate_count=count))
            log.info("PDF page %d/%d: %d particles", page_num, total, count)

            lines = page_text.split("\n")
            prev_tail = lines[-pdf_page_overlap_lines:] if lines else []

        if pdfium_doc is not None:
            with contextlib.suppress(Exception):
                pdfium_doc.close()

        return ExtractionResult(
            candidates=all_candidates,
            quality_notes=all_notes,
            page_stats=page_stats,
            transient_error_count=transient_errors,
            answered_calls=answered_calls,
        )

    async def _extract_pdf_paged_pooled(
        self,
        content: bytes,
        *,
        reference_published_at: datetime | None,
        completion_pool: CompletionPool,
    ) -> ExtractionResult:
        """Pooled-batch variant of ``_extract_pdf_paged``.

        Phase 1 parses every page exactly as the sequential loop does — same
        caps, same notes, same ``prev_tail`` threading — which is possible
        because the only inter-page coupling is parsed page text, never an LLM
        result. The ``max_pdf_seconds`` budget bounds this parse phase; the
        batch leg is bounded by ``llm.batch.max_wait_seconds``.
        Phase 2 submits every text page as one pooled group; vision pages are
        multimodal — outside the batch shape — and run
        sequentially after the group returns. Candidates, notes, and page
        stats are assembled in page order, matching the sequential output.
        """
        import io

        from pypdf import PdfReader

        from particles.llm import VisionImage

        cfg = get_config().extraction
        vcfg = get_config().extraction_vision
        pdf_page_overlap_lines = cfg.pdf_page_overlap_lines
        max_pages = cfg.max_pdf_pages
        max_page_chars = cfg.max_pdf_page_chars
        max_seconds = cfg.max_pdf_seconds

        pdfium_doc = _open_pdfium_for_vision(content) if vcfg.enabled else None
        vision_pages_used = 0

        reader = PdfReader(io.BytesIO(content))
        page_count = len(reader.pages)
        total = min(page_count, max_pages)

        head_notes: list[str] = []
        tail_notes: list[str] = []
        if page_count > max_pages:
            head_notes.append(
                f"PDF_PAGE_CAP: document has {page_count} pages; "
                f"only the first {max_pages} were extracted (extraction.max_pdf_pages)"
            )
            log.warning(
                "PDF has %d pages; capping extraction at %d (max_pdf_pages)",
                page_count,
                max_pages,
            )

        plans: list[_PdfPagePlan] = []
        prev_tail: list[str] = []
        started = time.monotonic()
        for idx, page in enumerate(reader.pages):
            page_num = idx + 1
            if page_num > max_pages:
                break
            # The wall-clock budget bounds the parse phase here (the LLM leg
            # is one pooled batch, bounded by llm.batch.max_wait_seconds).
            if time.monotonic() - started > max_seconds:
                tail_notes.append(
                    f"PDF_TIME_CAP: extraction exceeded {max_seconds:.0f}s wall-clock "
                    f"budget after page {idx}; remaining pages skipped "
                    "(extraction.max_pdf_seconds)"
                )
                log.warning(
                    "PDF extraction exceeded %.0fs budget after page %d; stopping",
                    max_seconds,
                    idx,
                )
                break
            parse_notes: list[str] = []
            page_text = page.extract_text() or ""
            if len(page_text) > max_page_chars:
                parse_notes.append(
                    f"PDF_PAGE_CHARS_CAP: page {page_num} extracted text truncated "
                    f"to {max_page_chars} chars (extraction.max_pdf_page_chars)"
                )
                page_text = page_text[:max_page_chars]

            context = (
                ("\n".join(prev_tail) + "\n" + page_text).strip()
                if prev_tail
                else page_text.strip()
            )

            images: list[VisionImage] | None = None
            if vcfg.enabled and _page_is_visual(
                page,
                page_text,
                trigger=vcfg.trigger,
                low_text_threshold=vcfg.low_text_threshold,
            ):
                if vision_pages_used < vcfg.max_pages:
                    try:
                        png = _render_page_png(pdfium_doc, idx, vcfg.render_dpi)
                        images = [VisionImage(media_type="image/png", data=png)]
                        vision_pages_used += 1
                    except Exception as exc:
                        parse_notes.append(f"VISION_RENDER_FAILED: page {page_num}: {exc}")
                        log.warning("Vision render failed for page %d: %s", page_num, exc)
                else:
                    parse_notes.append(
                        f"VISION_PAGE_CAP: page {page_num} is image-bearing but the "
                        f"per-document vision-page budget ({vcfg.max_pages}) is exhausted; "
                        "extracted text-only (extraction_vision.max_pages)"
                    )

            if not context and not images:
                parse_notes.append(f"ZERO_PAGE_YIELD: page {page_num} had no extractable text")
                plans.append(
                    _PdfPagePlan(
                        page_number=page_num,
                        context="",
                        images=None,
                        parse_notes=parse_notes,
                        skipped=True,
                    )
                )
                prev_tail = []
                continue

            log.info(
                "Extracting PDF page %d/%d (%d chars%s)",
                page_num,
                total,
                len(context),
                ", +vision" if images else "",
            )
            plans.append(
                _PdfPagePlan(
                    page_number=page_num,
                    context=context,
                    images=images,
                    parse_notes=parse_notes,
                    skipped=False,
                )
            )
            lines = page_text.split("\n")
            prev_tail = lines[-pdf_page_overlap_lines:] if lines else []

        if pdfium_doc is not None:
            with contextlib.suppress(Exception):
                pdfium_doc.close()

        # Phase 2: one pooled group for the text pages, then the (capped,
        # rare) vision pages sequentially — after the group, so this task
        # parks promptly and never stalls the wave for its siblings.
        text_plans = [p for p in plans if not p.skipped and p.images is None]
        planned_calls = [
            _build_llm_request(p.context, reference_published_at=reference_published_at)
            for p in text_plans
        ]
        settled = await _pooled_extract(completion_pool, planned_calls)
        outcome: dict[int, tuple[list[CandidateParticle], list[str], bool]] = {
            plan.page_number: reply for plan, reply in zip(text_plans, settled, strict=True)
        }
        for plan in (p for p in plans if not p.skipped and p.images is not None):
            if reference_published_at is not None:
                outcome[plan.page_number] = await _call_llm(
                    plan.context,
                    images=plan.images,
                    reference_published_at=reference_published_at,
                )
            else:
                outcome[plan.page_number] = await _call_llm(plan.context, images=plan.images)

        all_candidates: list[CandidateParticle] = []
        page_stats: list[PageStat] = []
        all_notes: list[str] = list(head_notes)
        transient_errors = 0
        answered_calls = 0
        for plan in plans:
            all_notes.extend(plan.parse_notes)
            if plan.skipped:
                page_stats.append(PageStat(page_number=plan.page_number, candidate_count=0))
                continue
            candidates, notes, transient = outcome[plan.page_number]
            count = len(candidates)
            if transient:
                transient_errors += 1
            else:
                answered_calls += 1
            if count == 0:
                all_notes.append(f"ZERO_PAGE_YIELD: page {plan.page_number} produced 0 particles")
            if notes:
                all_notes.extend(f"Page {plan.page_number}: {n}" for n in notes)
            all_candidates.extend(candidates)
            page_stats.append(PageStat(page_number=plan.page_number, candidate_count=count))
            log.info("PDF page %d/%d: %d particles", plan.page_number, total, count)
        all_notes.extend(tail_notes)

        return ExtractionResult(
            candidates=all_candidates,
            quality_notes=all_notes,
            page_stats=page_stats,
            transient_error_count=transient_errors,
            answered_calls=answered_calls,
        )

    async def _extract_image(
        self,
        content: bytes,
        media_type: str,
        *,
        reference_published_at: datetime | None = None,
    ) -> ExtractionResult:
        """Extract claims from a standalone image via one multimodal call.

        The blob is already an image (no rasterization), so it goes straight
        into the ``images`` channel. Requires a vision-capable
        extraction model (Claude 3+/4 are); a non-multimodal model surfaces the
        mismatch as a provider error recorded in ``quality_notes`` (the blob is
        retained, so a ``reindex`` after configuring a vision model recovers the
        claims). Not gated by ``extraction_vision.enabled`` — that knob trades off
        the PDF-page render heuristic, which does not apply to a bare image.
        """
        from particles.llm import VisionImage

        cfg = get_config().extraction
        if len(content) > cfg.max_image_bytes:
            return ExtractionResult(
                quality_notes=[
                    f"IMAGE_BYTES_CAP: image is {len(content)} bytes; exceeds "
                    f"extraction.max_image_bytes ({cfg.max_image_bytes}); skipped"
                ]
            )
        image = VisionImage(media_type=media_type, data=content)
        prompt = "(The source is the attached image; extract claims from it.)"
        if reference_published_at is not None:
            candidates, notes, transient = await _call_llm(
                prompt, images=[image], reference_published_at=reference_published_at
            )
        else:
            candidates, notes, transient = await _call_llm(prompt, images=[image])
        if not candidates and not transient:
            notes.append("ZERO_PAGE_YIELD: image produced 0 particles")
        return ExtractionResult(
            candidates=candidates,
            quality_notes=notes,
            transient_error_count=1 if transient else 0,
        )


def _drop_session_state_candidates(result: ExtractionResult, content: bytes) -> None:
    """Remove momentary session-state claims from ``result`` in place.

    Worktree names are read from the raw source bytes, so a kept claim loses a
    subject named after a worktree even when no dropped claim spelled its path.
    """
    filtered = drop_session_state(result.candidates, content.decode("utf-8", errors="replace"))
    result.candidates = filtered.candidates
    if filtered.dropped or filtered.subjects_stripped:
        # Never silent (the disclosure habit).
        result.quality_notes.append(
            f"session state: {filtered.dropped} claim(s) dropped,"
            f" {filtered.subjects_stripped} worktree subject(s) stripped"
        )


def _split_into_chunks(text: str, size: int, overlap_lines: int) -> list[str]:
    """Split text into size-character chunks, breaking at line boundaries.

    Retained for the github-pages extractor (§ Deferred);
    no longer used by the general extractor's HTML path, which routes
    through :func:`_split_into_paragraph_chunks` for carry-forward stability.
    """
    chunks: list[str] = []
    prev_tail: list[str] = []
    start = 0
    while start < len(text):
        end = start + size
        if end < len(text):
            nl = text.rfind("\n", start, end)
            if nl > start:
                end = nl
        segment = text[start:end]
        chunk = ("\n".join(prev_tail) + "\n" + segment).strip() if prev_tail else segment.strip()
        if chunk:
            chunks.append(chunk)
        lines = segment.split("\n")
        prev_tail = lines[-overlap_lines:] if len(lines) >= overlap_lines else lines
        start = end
    return chunks


def _split_into_paragraph_chunks(text: str, size: int) -> list[str]:
    """Split text into ≤``size``-character chunks, preferring paragraph breaks.

    Boundary ladder, checked per chunk in order:

    1. Last ``\\n\\n`` (paragraph break) in the ``size``-character window.
    2. Last ``\\n`` (line break) in the window, for paragraphs longer than
       the budget. This is the same fallback line chunker used.
    3. Hard cut at exactly ``size`` characters, for inputs that are one
       unbroken line.

    Empty / whitespace-only chunks are dropped. Unlike the line-based
    chunker, no overlap is prepended: paragraphs are the unit of extraction,
    and overlap would double-hash the same text and defeat carry-forward.
    """
    return [text[start:end] for start, end in paragraph_spans(text, size)]


def paragraph_spans(text: str, size: int) -> list[tuple[int, int]]:
    """The ``[start, end)`` spans :func:`_split_into_paragraph_chunks` cuts.

    Each span is its chunk's position in ``text``, whitespace already trimmed,
    so ``text[start:end]`` is the chunk byte for byte. The delta read
    needs the positions, not only the texts, to know where a read stopped.
    """
    spans: list[tuple[int, int]] = []
    start = 0
    n = len(text)
    while start < n:
        remaining = n - start
        if remaining <= size:
            _append_trimmed_span(spans, text, start, n)
            break
        window_end = start + size
        # Prefer the last paragraph break inside the window.
        cut = text.rfind("\n\n", start, window_end)
        advance: int
        if cut > start:
            advance = (cut + 2) - start  # skip the "\n\n" separator
            end = cut
        else:
            # Fall back to the last line break.
            nl = text.rfind("\n", start, window_end)
            if nl > start:
                advance = (nl + 1) - start
                end = nl
            else:
                # Hard cut.
                advance = size
                end = window_end
        _append_trimmed_span(spans, text, start, end)
        start += advance
    return spans


def _append_trimmed_span(spans: list[tuple[int, int]], text: str, start: int, end: int) -> None:
    """Append ``text[start:end]`` with its surrounding whitespace trimmed, if any is left."""
    segment = text[start:end]
    body = segment.strip()
    if body:
        lead = len(segment) - len(segment.lstrip())
        spans.append((start + lead, start + lead + len(body)))


# Patterns intentionally narrow: only remove what is provably cosmetic and
# has been observed to drift across re-deposits. New rules land additively.
_EDIT_MARKER_RE = re.compile(r"\[\s*edit\s*\]", re.IGNORECASE)
_WIKI_FOOTER_LINE_RE = re.compile(
    r"^[ \t]*"
    r"(?:This page was last (?:edited|modified)|Retrieved from|Last updated)"
    r"\b[^\n]*$",
    re.IGNORECASE | re.MULTILINE,
)


def _normalise_for_hashing(text: str) -> str:
    """Strip cosmetic noise that drifts across re-deposits.

    Removes Wikipedia-style ``[edit]`` markers and whole-line wiki footers
    that carry no claim signal but change every fetch. Idempotent: running
    the function twice produces the same output as running it once.

    The return value is the text the chunker, hasher, and LLM all see —
    the three share one normalised representation by design.
    """
    out = _EDIT_MARKER_RE.sub("", text)
    out = _WIKI_FOOTER_LINE_RE.sub("", out)
    return out


def _synthesise_html_page_stats(
    chunks: list[ChunkUnit], result: ExtractionResult
) -> list[PageStat]:
    """Build per-chunk ``PageStat`` rows for CLI compatibility.

    ``extract_with_carry_forward`` returns ``candidates`` only for the
    chunks that actually called the LLM; carry-forward chunks contribute
    to ``carry_forward_ids`` instead. The CLI's "Chunks:" output expects
    one row per chunk, so we synthesise the stats post-hoc by attributing
    each candidate back to the chunk whose hash it carries.
    """
    by_hash: dict[str, int] = {}
    for c in result.candidates:
        if c.chunk_hash:
            by_hash[c.chunk_hash] = by_hash.get(c.chunk_hash, 0) + 1
    stats: list[PageStat] = []
    for i, chunk in enumerate(chunks, start=1):
        stats.append(PageStat(page_number=i, candidate_count=by_hash.get(chunk.prompt_hash, 0)))
    return stats


def _set_outcome_starts(result: ExtractionResult, starts: list[int]) -> None:
    """Fill each chunk outcome's start, in chunk order.

    The outcomes come from ``extract_with_carry_forward`` in the order of the
    chunks it was handed, so they align with ``starts`` one to one. A count
    that does not match (an injected caller that chunked differently) leaves
    the starts unset, and the pipeline then keeps nothing from a failed read.
    """
    if len(result.chunk_outcomes) != len(starts):
        return
    for outcome, start in zip(result.chunk_outcomes, starts, strict=True):
        outcome.start = start


def _finish_chunk_stats(chunks: list[ChunkUnit], result: ExtractionResult) -> None:
    """Attach per-chunk stats to a chunked result and note every chunk that yielded nothing."""
    result.page_stats = _synthesise_html_page_stats(chunks, result)
    for stat in result.page_stats:
        if stat.candidate_count == 0:
            result.quality_notes.append(
                f"ZERO_PAGE_YIELD: chunk {stat.page_number} produced 0 particles"
            )


def _binary_fallback(result: ExtractionResult, append_prefix: bytes | None) -> ExtractionResult:
    """Say why a binary source ignored a handed-over prefix (text paths only)."""
    if append_prefix is not None:
        result.append_fallback = "not a text source"
    return result


@dataclass
class _PlannedLLMCall:
    """One extraction request, built but not yet dispatched.

    The build half of ``_call_llm``: everything before the network — the
    fenced prompt pair, the response schema, the token cap — so a pooled
    caller can enumerate a whole request set up front and submit it as one
    batch. ``images`` marks a multimodal call, which the batch shape
    cannot carry (``CompletionRequest`` is text-only); pooled
    callers must dispatch image-bearing plans through ``_call_llm``.
    """

    request: CompletionRequest
    response_schema: dict[str, Any]
    max_tokens: int
    images: list[VisionImage] | None = None


@dataclass
class _PdfPagePlan:
    """One parsed PDF page awaiting dispatch on the pooled path."""

    page_number: int
    context: str
    images: list[VisionImage] | None
    parse_notes: list[str]
    skipped: bool


def _build_llm_request(
    text: str,
    images: list[VisionImage] | None = None,
    *,
    reference_published_at: datetime | None = None,
    tool_turns_present: bool = False,
    context: str | None = None,
) -> _PlannedLLMCall:
    """Build one extraction request — the pre-network half of ``_call_llm``.

    ``context`` is already-extracted text shown before ``text``
    on a delta read: it rides the user turn as its own fenced section, and the
    system turn gains :data:`APPEND_CONTEXT_RULE`. ``None`` leaves the request
    exactly as before.
    """
    from particles.llm import (
        CompletionRequest,
        data_fence_instruction,
        fence,
        fenced_prompt,
        make_nonce,
    )

    cfg = get_config().extraction
    _cfg = get_config()
    validity_enabled = _cfg.extraction_validity.enabled
    components = _configured_prompt_components(
        tool_turns_present=tool_turns_present, context_present=context is not None
    )
    template = assemble_prompt(components)
    # the one place the prompt is assembled is the one place its
    # components are recorded, so the stamp names exactly what was sent.
    record_prompt_components(components)
    record_component(CODE_PARSE, parse_component_digest())
    if images:
        record_component(CHANNEL_VISION, path_component_digests()[CHANNEL_VISION])
    # Fill the {reference_date} placeholder (the validity rule's, or
    # _REFERENCE_DATE_RULE's when validity is off) with the source's publication
    # instant (or the extraction wall-clock) so the model resolves relative time
    # ("tomorrow", "last month") to absolute dates. ``str.replace`` (not
    # ``str.format``) — the JSON schema block carries literal braces.
    #
    # prompt caching: the ~1.3k+-token instruction block repeats on
    # every extraction call in a run, so mark its invariant prefix as a cache
    # boundary. The reference_date is the ONE per-source-varying byte in the
    # instructions, so the boundary sits *before* it — everything after (the
    # date, the remaining rules, the schema) stays uncached, so a cache hit
    # survives across sources. ``cache_split`` is the offset into the
    # (identical-up-to-here) system string where caching stops.
    cache_split = template.index("{reference_date}")
    anchor = reference_published_at or datetime.now(UTC)
    instructions = template.replace("{reference_date}", anchor.date().isoformat())
    # F3 hardening: the trusted rules/schema go in the ``system`` turn; the
    # untrusted decoded source (and, on the vision path, the rendered page) is
    # the ONLY thing in the user turn, wrapped in a per-call nonce fence so an
    # injected "ignore the above …" line in a deposited document cannot steer
    # extraction. Hardening, not immunity — the JSON-array contract the parser
    # enforces is the structural backstop. The nonce is minted per request, so
    # pooled sibling requests in one batch never share a fence.
    # The cache split is *within* the system turn,
    # so both halves stay trusted and F3 is unchanged.
    if context is None:
        system, user = fenced_prompt(instructions, text, label="source")
    else:
        # Two untrusted segments share one nonce (the multi-segment form
        # ``fenced_prompt`` documents): the context first, then the source.
        nonce = make_nonce()
        system = f"{instructions.rstrip()}\n\n{data_fence_instruction(nonce)}"
        user = f"{fence(context, nonce, label='context')}\n\n{fence(text, nonce, label='source')}"
    cache_prefix = system[:cache_split] or None
    # the candidate-array schema rides along so a schema-enforcing
    # provider (LocalProvider structured output) can pin the reply shape.
    schema = _extraction_response_schema(
        scope_enabled=_cfg.extraction_scope.enabled,
        modality_enabled=_cfg.extraction_modality.enabled,
        polarity_enabled=_cfg.extraction_polarity.enabled,
        stance_enabled=_cfg.extraction_stance.enabled,
        validity_enabled=validity_enabled,
        structure_enabled=_cfg.structured_claim.enabled,
    )
    return _PlannedLLMCall(
        request=CompletionRequest(
            prompt=user, system=system[cache_split:], cache_prefix=cache_prefix
        ),
        response_schema=schema,
        max_tokens=cfg.max_tokens,
        images=images,
    )


#: The shape of one extraction reply, read off the parse (not off the adapter:
#: the port returns text only). ``truncated`` is a reply cut at the output
#: budget whose complete leading claims were salvaged; the claims after the cut
#: are lost. ``unparsed`` is a reply that yielded no JSON array at all.
ReplyShape = Literal["complete", "truncated", "unparsed"]


@dataclass
class ReplyTally:
    """What the extraction replies of one pass did with their output budget.

    Opened by :func:`tally_replies` around one snapshot's extraction and
    filled at the completion seam, so the aggregating paths (single pass,
    chunked, PDF pages) need no new plumbing. The first-run audit reads it to
    tell a fully extracted file from one that was cut short.
    """

    truncated: int = 0
    """Replies (after any retry) cut at the output budget: some claims were lost."""
    retried: int = 0
    """Replies re-issued once at ``extraction.retry_max_tokens``."""


_REPLY_TALLY: ContextVar[ReplyTally | None] = ContextVar("particles_reply_tally", default=None)


@contextlib.contextmanager
def tally_replies() -> Iterator[ReplyTally]:
    """Collect the :class:`ReplyTally` of every extraction reply in the block."""
    tally = ReplyTally()
    token = _REPLY_TALLY.set(tally)
    try:
        yield tally
    finally:
        _REPLY_TALLY.reset(token)


def _record_reply(shape: ReplyShape, *, retried: bool = False) -> None:
    tally = _REPLY_TALLY.get()
    if tally is None:
        return
    if shape == "truncated":
        tally.truncated += 1
    if retried:
        tally.retried += 1


@dataclass
class _FinishedReply:
    candidates: list[CandidateParticle]
    notes: list[str]
    transient: bool
    shape: ReplyShape


def _finish_llm_reply(
    raw: str | None,
    provider_model: str,
    images: list[VisionImage] | None,
) -> _FinishedReply:
    """Parse one reply and classify its shape; records nothing (see ``_settle_reply``).

    An ``unparsed`` reply is reported as ``transient``: the source is fine and
    the reply is not, which is the empty-completion case with some text in it.
    Stamping the snapshot COMPLETE with zero claims would hide it from every
    retry path (``extract --all-pending``, a re-run audit) and read as a source
    that holds nothing.
    """
    if raw is None:
        return _FinishedReply([], ["API error: batch result unavailable"], True, "unparsed")
    log.debug("Raw LLM response (%d chars):\n%s", len(raw), raw)
    candidates, notes, shape = _parse_extraction_reply(raw)
    for candidate in candidates:
        candidate.provider_model = provider_model
    if images:
        for candidate in candidates:
            props = candidate.properties or {}
            props[VISION_SOURCE_MODALITY_KEY] = VISION_SOURCE_MODALITY
            candidate.properties = props
    return _FinishedReply(candidates, notes, shape == "unparsed", shape)


async def _call_llm(
    text: str,
    images: list[VisionImage] | None = None,
    *,
    reference_published_at: datetime | None = None,
    tool_turns_present: bool = False,
    context: str | None = None,
) -> tuple[list[CandidateParticle], list[str], bool]:
    """Send text to the LLM and return (candidates, notes, transient_error).

    ``transient_error`` is True when the API call raised (rate limit, billing,
    5xx, network) — a structured signal the aggregation layers (chunked/PDF)
    propagate into ``ExtractionResult.transient_error_count`` so the pipeline
    never has to parse the ``"API error: …"`` note text (F4.1).

    When ``images`` is supplied, the call is multimodal — the page
    text and the rendered page image go to the vision-capable provider in one
    request, and the resulting candidates are stamped with the vision
    source-modality marker.

    Composed as build → complete → parse: the pooled extraction
    path reuses ``_build_llm_request`` and its twin :func:`_pooled_extract`
    around one merged batch call instead of this per-call middle.
    """
    from particles.llm import (
        AccountLevelLLMError,
        EmptyCompletionError,
        complete_with_provider_model,
        is_account_level_failure,
    )

    planned = _build_llm_request(
        text,
        images,
        reference_published_at=reference_published_at,
        tool_turns_present=tool_turns_present,
        context=context,
    )
    # resolve the provider ONCE and read its pairing here, at
    # the call site. ``get_provider`` reads live config on every call, so a
    # reload between chunks can change the pairing mid-pass — a value read
    # once at pipeline scope is not guaranteed to be what served this call.
    # The stamp is the *requested* pairing; no adapter reads the served model
    # back off the response.
    provider_model = ""
    try:
        raw, provider_model = await complete_with_provider_model(
            "extraction",
            planned.request.prompt,
            max_tokens=planned.max_tokens,
            system=planned.request.system,
            images=images,
            response_schema=planned.response_schema,
            cache_prefix=planned.request.cache_prefix,
        )
    except Exception as exc:
        # An account-level failure (bad key, no permission, no credit) will fail
        # every remaining call in this run, so it is NOT a per-call transient:
        # swallowing it here is what made `extract --all-pending` walk 68
        # snapshots — and every page of each — re-issuing a doomed request and
        # printing the same billing error hundreds of times. Raise so the bulk
        # caller can stop once; the pipeline's interrupt handler resets the
        # snapshot IN_PROGRESS → PENDING on the way out, so nothing is lost.
        if is_account_level_failure(exc):
            log.error("Extraction unavailable (account-level): %s", exc)
            raise AccountLevelLLMError(exc) from exc
        if not isinstance(exc, EmptyCompletionError):
            log.error("Extraction API call failed: %s", exc)
            return [], [f"API error: {exc}"], True
        first: _FinishedReply | None = None
        first_error: Exception | None = exc
    else:
        first = _finish_llm_reply(raw, provider_model, images)
        first_error = None
        if first.shape == "complete":
            return first.candidates, first.notes, first.transient

    # A reply with no text, cut at the budget, or unparseable is a budget
    # failure on this call, and the same call at the same budget tends to
    # reproduce it (an adaptive-thinking model spends the budget on thinking
    # first). Re-issue it ONCE at the larger budget, which is a different call.
    # A billing, network, or refusal failure never reaches here.
    retry_budget = get_config().extraction.retry_max_tokens
    if retry_budget <= planned.max_tokens:
        return _settle_reply(first, first_error, retried=False)
    log.warning(
        "Extraction reply at max_tokens=%d was %s; retrying once at %d",
        planned.max_tokens,
        "empty" if first is None else first.shape,
        retry_budget,
    )
    try:
        raw, provider_model = await complete_with_provider_model(
            "extraction",
            planned.request.prompt,
            max_tokens=retry_budget,
            system=planned.request.system,
            images=images,
            response_schema=planned.response_schema,
            cache_prefix=planned.request.cache_prefix,
        )
    except Exception as exc:
        if is_account_level_failure(exc):
            log.error("Extraction unavailable (account-level): %s", exc)
            raise AccountLevelLLMError(exc) from exc
        log.error("Extraction retry at max_tokens=%d failed: %s", retry_budget, exc)
        return _settle_reply(first, exc, retried=True)
    second = _finish_llm_reply(raw, provider_model, images)
    if first is not None and len(first.candidates) > len(second.candidates):
        # The retry did worse (a fresh sample can): keep what the first kept.
        return _settle_reply(first, None, retried=True)
    return _settle_reply(second, None, retried=True)


def _settle_reply(
    reply: _FinishedReply | None, error: Exception | None, *, retried: bool
) -> tuple[list[CandidateParticle], list[str], bool]:
    """Record the final reply of a ``_call_llm`` on the tally and return its triple."""
    if reply is None:
        _record_reply("unparsed", retried=retried)
        return [], [f"API error: {error}"], True
    _record_reply(reply.shape, retried=retried)
    if reply.shape == "truncated":
        log.warning(
            "Extraction reply was cut at the output-token limit%s; kept %d complete "
            "claim(s), and the claims after the cut were lost",
            " after one retry" if retried else "",
            len(reply.candidates),
        )
    notes = list(reply.notes)
    if retried and reply.shape != "complete":
        notes.append(
            f"Reply was still {reply.shape} after one retry at "
            f"extraction.retry_max_tokens={get_config().extraction.retry_max_tokens}"
        )
    return reply.candidates, notes, reply.transient


async def _pooled_group_complete(
    pool: CompletionPool,
    planned: list[_PlannedLLMCall],
    *,
    max_tokens: int | None = None,
    failures_out: list[RequestFailure | None] | None = None,
) -> tuple[list[str | None], str]:
    """Submit image-less planned calls as one pooled group.

    Returns ``(results, provider_model)`` positionally aligned with
    ``planned``. A job-level failure degrades to all-``None`` (the per-chunk
    transient path) unless it is account-level, which raises
    ``AccountLevelLLMError`` exactly as the sequential ``_call_llm`` does —
    same classification, same bulk-caller stop. ``max_tokens`` overrides the
    planned budget (the retry wave); ``failures_out`` receives the pool's
    per-request failure kinds.
    """
    from particles.llm import AccountLevelLLMError, RequestFailure, is_account_level_failure

    if not planned:
        return [], ""
    try:
        return await pool.complete_group(
            [p.request for p in planned],
            max_tokens=max_tokens if max_tokens is not None else planned[0].max_tokens,
            response_schema=planned[0].response_schema,
            failures_out=failures_out,
        )
    except Exception as exc:
        if is_account_level_failure(exc):
            log.error("Extraction unavailable (account-level): %s", exc)
            raise AccountLevelLLMError(exc) from exc
        log.error("Pooled extraction dispatch failed: %s", exc)
        if failures_out is not None:
            failures_out.extend([RequestFailure.UNAVAILABLE] * len(planned))
        return [None] * len(planned), ""


async def _pooled_extract(
    pool: CompletionPool,
    planned: list[_PlannedLLMCall],
) -> list[tuple[list[CandidateParticle], list[str], bool]]:
    """The pooled twin of :func:`_call_llm`: build is done, this is complete → parse.

    One ``(candidates, notes, transient_error)`` per planned call, in order.
    A reply that is cut at the output budget, unparseable, or empty is
    re-issued once at ``extraction.retry_max_tokens``, exactly as the
    sequential path does, as a **second pooled group**: the retries of every
    snapshot in the run park together and ride one follow-up batch, at the
    batch price. A request that never produced a reply (errored,
    expired, cancelled) is not retried: it is no budget problem, and a batch
    that outlived ``llm.batch.max_wait_seconds`` would only outlive it again.
    The snapshot goes back to PENDING for a later run, as before.
    """
    from particles.llm import RequestFailure

    failures: list[RequestFailure | None] = []
    results, provider_model = await _pooled_group_complete(pool, planned, failures_out=failures)
    firsts: list[_FinishedReply | None] = []
    retry_at: list[int] = []
    for i, raw in enumerate(results):
        kind = failures[i] if i < len(failures) else None
        if raw is None:
            firsts.append(None)
            # Only an empty reply is a budget failure; an unexplained None is
            # read as unavailable, which retries nothing.
            if kind is RequestFailure.EMPTY:
                retry_at.append(i)
            continue
        first = _finish_llm_reply(raw, provider_model, None)
        firsts.append(first)
        if first.shape != "complete":
            retry_at.append(i)

    retry_budget = get_config().extraction.retry_max_tokens
    seconds: dict[int, _FinishedReply] = {}
    retry_error: Exception | None = None
    if retry_at and retry_budget > planned[0].max_tokens:
        log.warning(
            "%d pooled extraction repl%s at max_tokens=%d came back cut, empty or "
            "unparseable; re-issuing them once at %d in a follow-up pooled batch",
            len(retry_at),
            "y" if len(retry_at) == 1 else "ies",
            planned[0].max_tokens,
            retry_budget,
        )
        retry_results, retry_pairing = await _pooled_group_complete(
            pool, [planned[i] for i in retry_at], max_tokens=retry_budget
        )
        for i, raw in zip(retry_at, retry_results, strict=True):
            if raw is not None:
                seconds[i] = _finish_llm_reply(raw, retry_pairing, None)
        if all(r is None for r in retry_results):
            retry_error = RuntimeError("batch result unavailable")
    else:
        retry_at = []

    settled: list[tuple[list[CandidateParticle], list[str], bool]] = []
    retried = set(retry_at)
    for i, head in enumerate(firsts):
        if i not in retried:
            if head is None:
                # Unavailable, or empty with no larger budget to retry at:
                # the per-request transient every aggregation layer propagates.
                reason = (
                    "reply carried no text at the output budget"
                    if i < len(failures) and failures[i] is RequestFailure.EMPTY
                    else "batch result unavailable"
                )
                settled.append(([], [f"API error: {reason}"], True))
            else:
                settled.append(_settle_reply(head, None, retried=False))
            continue
        second = seconds.get(i)
        if second is None:
            # The retry produced nothing: keep what the head reply kept.
            error = retry_error or RuntimeError("retry result unavailable")
            settled.append(_settle_reply(head, None if head is not None else error, retried=True))
        elif head is not None and len(head.candidates) > len(second.candidates):
            settled.append(_settle_reply(head, None, retried=True))
        else:
            settled.append(_settle_reply(second, None, retried=True))
    return settled


def _open_pdfium_for_vision(content: bytes) -> Any:
    """Open a ``pypdfium2`` document for page rasterization.

    The rasterizer lives behind the ``[vision]`` optional extra, imported here
    only when ``extraction_vision.enabled``. A missing extra raises an
    actionable :class:`ImportError` (not a silent no-op) so an operator who
    enabled vision without installing it gets a legible setup error.
    """
    try:
        import pypdfium2 as pdfium
    except ImportError as exc:  # pragma: no cover - exercised via a patched import
        raise ImportError(
            "Vision extraction is enabled (extraction_vision.enabled) but the "
            "optional 'vision' extra is not installed. Install it with: "
            "pip install 'particles[vision]'"
        ) from exc
    return pdfium.PdfDocument(content)


def _render_page_png(doc: Any, page_index: int, dpi: int) -> bytes:
    """Render one PDF page to PNG bytes at ``dpi`` via ``pypdfium2`` + Pillow."""
    import io

    page = doc[page_index]
    bitmap = page.render(scale=dpi / 72.0)
    image = bitmap.to_pil()
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _page_is_visual(
    page: Any,
    page_text: str,
    *,
    trigger: str,
    low_text_threshold: int,
) -> bool:
    """Decide whether a PDF page should take the vision path.

    ``always`` → every page. ``image_bearing`` (default) → the page carries
    embedded raster images, or its extracted text is below
    ``low_text_threshold`` (a scanned / figure-only page). The image check is
    defensive: ``pypdf``'s ``page.images`` can raise on malformed objects, in
    which case we fall back to the low-text signal alone.
    """
    if trigger == "always":
        return True
    if len(page_text.strip()) < low_text_threshold:
        return True
    try:
        return len(list(page.images)) > 0
    except Exception:
        return False


def content_to_text(content: bytes) -> str:
    """Convert raw bytes to plain text.

    PDF → pypdf page concatenation (single-pass; paged path is in _extract_pdf_paged).
    HTML → Markdown via html2text (preserves headings/lists, strips tags/links).
    Other → UTF-8 decode.
    """
    if content.startswith(b"%PDF"):
        import io

        from pypdf import PdfReader

        cfg = get_config().extraction
        max_pages = cfg.max_pdf_pages
        max_page_chars = cfg.max_pdf_page_chars
        reader = PdfReader(io.BytesIO(content))
        # Security: cap page count and per-page extracted text so a hostile
        # PDF can't force unbounded memory in the single-pass path.
        pages: list[str] = []
        for idx, page in enumerate(reader.pages):
            if idx >= max_pages:
                break
            text = page.extract_text() or ""
            if len(text) > max_page_chars:
                text = text[:max_page_chars]
            pages.append(text)
        return "\n\n".join(pages)
    if _is_html(content):
        import html2text as _h2t

        h = _h2t.HTML2Text()
        h.ignore_links = True
        h.ignore_images = True
        h.body_width = 0
        return h.handle(content.decode("utf-8", errors="replace"))
    return content.decode("utf-8", errors="replace")


def _is_html(content: bytes) -> bool:
    sniff = content[:1024].lower()
    return b"<!doctype html" in sniff or b"<html" in sniff


def _strip_obsidian_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Split Obsidian YAML frontmatter from a Markdown body.

    Obsidian (and most static-site generators) prefix notes with a
    ``---\\nkey: value\\n---\\n`` block of YAML metadata (tags, aliases,
    publication date, etc.). When the GeneralExtractor sees that block
    inside the LLM prompt, the model dutifully extracts "the document has
    a `tags` field of value [foo, bar]" — which is metadata noise, not a
    claim about the world. This helper detects the block and returns
    ``(parsed_frontmatter, body)`` so the extractor can drop the block
    before extraction.

    The block is only recognised when ``text`` begins with ``---\\n`` AND
    a closing ``\\n---`` line is present within the first ~4 KiB; otherwise
    the input is returned unchanged with an empty dict. Malformed YAML
    inside a syntactically-valid frontmatter block is dropped silently
    (returns ``{}`` for the metadata, plus the body) — operators don't
    want extraction to fail because their frontmatter has a syntax error.

    Args:
        text: Decoded source text (typically ``content.decode("utf-8")``).

    Returns:
        Tuple of (parsed-frontmatter dict, remaining body string). When
        no frontmatter is detected, returns ``({}, text)`` unmodified.
    """
    if not text.startswith("---\n") and not text.startswith("---\r\n"):
        return {}, text
    # Search for the closing fence within a reasonable distance — vault
    # operators rarely write multi-page YAML headers, and constraining
    # the search prevents pathological inputs from forcing a full-file
    # split. 8 KiB comfortably accommodates real-world Obsidian headers.
    lookup_end = min(len(text), 8192)
    after_open = 4 if text.startswith("---\n") else 5
    # The closing fence is a line containing exactly "---" (optionally
    # followed by trailing whitespace). Search line-by-line within the
    # bounded window.
    haystack = text[after_open:lookup_end]
    close_idx = -1
    cursor = 0
    while cursor < len(haystack):
        nl = haystack.find("\n", cursor)
        line = haystack[cursor : nl if nl != -1 else len(haystack)]
        if line.rstrip() == "---":
            close_idx = cursor
            break
        if nl == -1:
            break
        cursor = nl + 1
    if close_idx == -1:
        return {}, text
    yaml_block = haystack[:close_idx]
    # Position in the original text where the closing line ends; skip the
    # trailing newline after it so the body starts cleanly.
    end_of_close_line = after_open + close_idx + len("---")
    body_start = end_of_close_line
    if body_start < len(text) and text[body_start] == "\n":
        body_start += 1
    elif body_start < len(text) - 1 and text[body_start : body_start + 2] == "\r\n":
        body_start += 2
    body = text[body_start:]
    try:
        import yaml

        parsed = yaml.safe_load(yaml_block) if yaml_block.strip() else {}
        meta: dict[str, Any] = parsed if isinstance(parsed, dict) else {}
    except yaml.YAMLError:
        # Malformed frontmatter — still strip it from the body so the LLM
        # doesn't see metadata, but return an empty dict to signal "no
        # usable metadata recovered".
        meta = {}
    return meta, body


def _parse_iso_datetime(raw: str) -> datetime | None:
    """Parse an ISO-8601 date or datetime the model emitted, or None if malformed.

    A bare date (``2026-09-01``) parses to midnight; a trailing ``Z`` is accepted.
    An unparseable value (e.g. an unresolved relative expression like ``tomorrow``)
    returns None so the gate drops it rather than guessing a date.
    """
    try:
        return datetime.fromisoformat(raw.strip().replace("Z", "+00:00"))
    except ValueError:
        return None


def _gate_valid_until(
    raw_valid_until: object,
    raw_confidence: object,
    raw_basis: object,
    *,
    floor: float,
    now: datetime,
) -> tuple[datetime | None, float | None, str | None, str | None]:
    """Apply the three-condition validity-emission gate to one item.

    Returns ``(valid_until, confidence, basis, note)``. ``valid_until`` is set
    only when **all three** conditions hold, biased hard toward under-emission
    (a spurious boundary silently retires a durable fact via the §9.3 lint):

    1. **explicit boundary cue** — the model emitted a non-null, parseable date;
    2. **confidence floor** — ``validity_confidence >= floor`` (a distinct
       quantity from the claim's own confidence);
    3. **future date** — the resolved boundary is after ``now`` (a born-expired
       ``valid_until <= now`` is dropped, avoiding the worst silent-retirement
       case and keeping the born-ACTIVE invariant clean).

    Any condition failing yields ``valid_until=None`` and a non-empty ``note``
    naming the drop reason — never a silent drop. ``confidence`` / ``basis`` come
    back only alongside an emitted boundary, for the audit crumbs (§3).
    """
    # Condition 1: an explicit boundary cue — a non-null, parseable date.
    if not isinstance(raw_valid_until, str) or not raw_valid_until.strip():
        return None, None, None, None
    parsed = _parse_iso_datetime(raw_valid_until)
    if parsed is None:
        return (
            None,
            None,
            None,
            (
                f"VALIDITY_UNPARSEABLE: valid_until {raw_valid_until!r} is not a resolvable "
                "ISO-8601 date; no boundary set"
            ),
        )
    # Normalize naive → UTC (matches the §9.3 staleness lint's naive-as-UTC rule).
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    # Condition 2: the boundary-confidence floor.
    try:
        confidence: float | None = float(raw_confidence)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        confidence = None
    if confidence is not None and not math.isfinite(confidence):
        confidence = None
    if confidence is None or confidence < floor:
        return (
            None,
            None,
            None,
            (
                f"VALIDITY_BELOW_FLOOR: boundary {parsed.date().isoformat()} dropped "
                f"(validity_confidence {confidence} < floor {floor})"
            ),
        )
    # Condition 3: a future date — a born-expired past bound is dropped.
    if parsed <= now:
        return (
            None,
            None,
            None,
            (
                f"VALIDITY_BORN_EXPIRED: boundary {parsed.date().isoformat()} is already past "
                "at extraction time; dropped (the claim keeps its decay treatment)"
            ),
        )
    basis = raw_basis.strip() if isinstance(raw_basis, str) and raw_basis.strip() else None
    return parsed, confidence, basis, None


def _parse_extraction_response(
    raw: str,
) -> tuple[list[CandidateParticle], list[str]]:
    """Parse one extraction reply into candidates and quality notes.

    The shape-blind view of :func:`_parse_extraction_reply`, kept for the
    callers (and tests) that only need the claims.
    """
    candidates, notes, _shape = _parse_extraction_reply(raw)
    return candidates, notes


def _parse_extraction_reply(
    raw: str,
) -> tuple[list[CandidateParticle], list[str], ReplyShape]:
    """Parse one extraction reply and say whether it was whole, cut short, or unusable."""
    notes: list[str] = []
    shape: ReplyShape = "complete"
    try:
        # Strip markdown code fences if present
        if raw.startswith("```"):
            lines = raw.split("\n")
            raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        # Response may be truncated mid-array (max_tokens reached). Try to
        # recover complete objects by trimming after the last closing brace.
        # Whatever followed the cut is lost: nothing re-reads this text, so the
        # note says so rather than promising a later pass picks it up.
        last_brace = raw.rfind("}")
        if last_brace != -1:
            try:
                data = json.loads(raw[: last_brace + 1] + "]")
                shape = "truncated"
                notes.append(
                    f"Reply hit the model output-token limit; kept {len(data)} complete "
                    f"claims, and any claims after the cut were lost"
                )
                # Not yet a loss: the caller may retry this reply at a larger
                # budget, and the snapshot may still be handed back whole.
                # ``_settle_reply`` warns once the reply is final.
                log.info(
                    "Extraction reply hit the output-token limit; %d complete claim(s) "
                    "recovered before the cut",
                    len(data),
                )
            except json.JSONDecodeError:
                log.warning("Failed to parse extraction response: %s", exc)
                return [], [f"JSON parse error: {exc}"], "unparsed"
        else:
            log.warning("Failed to parse extraction response: %s", exc)
            return [], [f"JSON parse error: {exc}"], "unparsed"

    if not isinstance(data, list):
        return [], ["Response is not a JSON array"], "unparsed"

    candidates: list[CandidateParticle] = []
    # document-scope classification + mode handling (read once).
    scope_cfg = get_config().extraction_scope
    # assertion-modality classification (read once).
    modality_enabled = get_config().extraction_modality.enabled
    # cap. 1: claim-polarity classification (read once).
    polarity_enabled = get_config().extraction_polarity.enabled
    # endorsement-stance detection (read once).
    stance_enabled = get_config().extraction_stance.enabled
    # event-anchored validity gate (config + wall-clock read once). The
    # born-expired cutoff is the actual current instant, independent of the prompt
    # anchor used to resolve relative dates.
    validity_cfg = get_config().extraction_validity
    validity_now = datetime.now(UTC)
    # derived S-P-O annotation (read once).
    structure_enabled = get_config().structured_claim.enabled
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            notes.append(f"Item {i} is not an object; skipped")
            continue
        content = item.get("content", "").strip()
        if not content:
            notes.append(f"Item {i} has empty content; skipped")
            continue
        try:
            conf = float(item.get("confidence_value", 0.5))
            # Guard non-finite BEFORE clamping: json.loads accepts the
            # non-standard NaN / Infinity literals, and max(0.0, min(1.0, nan))
            # evaluates to 1.0 — so a poisoned source's NaN confidence would
            # silently become MAXIMUM. Route it to the same safe default the
            # invalid-confidence branch uses.
            if not math.isfinite(conf):
                conf = 0.5
                notes.append(f"Item {i} has non-finite confidence_value; defaulted to 0.5")
            else:
                # D1: clamp strictly inside (0, 1). 0.0 and 1.0 are
                # exact fixed points of the logit transform temperature
                # scaling runs in, so a particle minted at either is
                # uncalibratable for the rest of its life — and what is
                # stored is immutable. The rubric asks the model
                # not to emit them; this is what makes that hold when a model
                # does anyway — and whether a model complies varies by model
                # and by source genre, measured. Deliberately wider
                # than the rubric's [0.05, 0.97] guidance: this is a backstop
                # that should distort a stated value as little as possible,
                # not a target.
                conf = max(_CONFIDENCE_FLOOR, min(_CONFIDENCE_CEILING, conf))
        except (TypeError, ValueError):
            conf = 0.5
            notes.append(f"Item {i} has invalid confidence_value; defaulted to 0.5")
        try:
            nature = UncertaintyNature(item.get("uncertainty_nature", "EPISTEMIC"))
        except ValueError:
            nature = UncertaintyNature.EPISTEMIC
            notes.append(f"Item {i} has invalid uncertainty_nature; defaulted to EPISTEMIC")
        subjects_raw = item.get("subjects", [])
        subjects: list[str] = (
            [str(s) for s in subjects_raw if str(s).strip()]
            if isinstance(subjects_raw, list)
            else []
        )

        # act on the document-scope classification. A DOCUMENT_META
        # candidate is dropped (suppress), tagged-and-excluded (label), or
        # tagged-but-not-excluded (passthrough). WORLD candidates and the
        # disabled case fall through unchanged.
        properties: dict[str, object] | None = None
        is_meta = (
            scope_cfg.enabled
            and str(item.get("scope", "WORLD")).strip().upper() == SCOPE_DOCUMENT_META
        )
        if is_meta:
            snippet = content[:80]
            if scope_cfg.mode == "suppress":
                notes.append(f"DOCUMENT_META (suppressed): {snippet}")
                continue
            properties = {SCOPE_KEY: SCOPE_DOCUMENT_META}
            if scope_cfg.mode == "passthrough":
                properties[SCOPE_ACTION_KEY] = SCOPE_ACTION_OBSERVE
                notes.append(f"DOCUMENT_META (passthrough, not excluded): {snippet}")
            else:  # label
                notes.append(f"DOCUMENT_META (labelled, excluded from factual surface): {snippet}")

        # cap. 1: claim-polarity classification. Default-safe — a
        # missing / unknown value (or the disabled case) falls back to
        # ASSERTED, so a classifier miss can only ever leave a claim on the
        # factual surface, never wrongly hide a real decision. The two
        # non-asserted values are recorded on ``properties["extraction:polarity"]``; the
        # operation layer excludes them from the default surface (mirrors
        # scope — confidence is never the lever). No silent
        # truncation: every classification is appended to ``notes``.
        if polarity_enabled:
            raw_polarity = str(item.get("polarity", POLARITY_ASSERTED)).strip().upper()
            if raw_polarity in NON_ASSERTED_POLARITIES:
                properties = properties or {}
                properties[POLARITY_KEY] = raw_polarity
                notes.append(
                    f"{raw_polarity} (non-asserted, excluded from factual surface): {content[:80]}"
                )
            elif raw_polarity != POLARITY_ASSERTED:
                notes.append(
                    f"Item {i} has invalid polarity {raw_polarity!r}; defaulted to ASSERTED"
                )

        # assertion_modality classification. Default-safe — an
        # unknown / missing value (or the disabled case) falls back to
        # FALSIFIABLE, so a classifier miss never wrongly exempts a claim from
        # the truth engine.
        modality = AssertionModality.FALSIFIABLE
        if modality_enabled:
            try:
                modality = AssertionModality(
                    str(item.get("assertion_modality", "FALSIFIABLE")).strip().upper()
                )
            except ValueError:
                notes.append(f"Item {i} has invalid assertion_modality; defaulted to FALSIFIABLE")

        # stance reification fields. Default-safe — a missing /
        # unknown stance_kind, or an out-of-range / non-integer target, leaves
        # the candidate a plain claim (no stance). The pipeline supplies the
        # holder (the source author) and creates the edge.
        stance_kind: RelationType | None = None
        stance_target_index: int | None = None
        stance_magnitude: float | None = None
        if stance_enabled:
            raw_kind = str(item.get("stance_kind") or "").strip().upper()
            if raw_kind in (RelationType.ENDORSES.value, RelationType.DISPUTES.value):
                target_raw = item.get("stance_target")
                if isinstance(target_raw, int) and not isinstance(target_raw, bool):
                    stance_kind = RelationType(raw_kind)
                    stance_target_index = target_raw
                    mag_raw = item.get("stance_magnitude")
                    if isinstance(mag_raw, (int, float)) and not isinstance(mag_raw, bool):
                        stance_magnitude = max(0.0, min(1.0, float(mag_raw)))
                else:
                    notes.append(f"Item {i} stance_target is not an index; stance dropped")

        # event-anchored validity gate. Sets ``valid_until`` only when
        # the boundary cue + confidence floor + future-date conditions all hold;
        # every drop is a quality note (never silent). The basis + confidence
        # audit crumbs ride on ``properties`` beside any polarity/scope tags.
        valid_until: datetime | None = None
        if validity_cfg.enabled:
            valid_until, vconf, vbasis, vnote = _gate_valid_until(
                item.get("valid_until"),
                item.get("validity_confidence"),
                item.get("validity_basis"),
                floor=validity_cfg.min_boundary_confidence,
                now=validity_now,
            )
            if vnote:
                notes.append(f"Item {i}: {vnote}")
            if valid_until is not None:
                properties = properties or {}
                if vbasis:
                    properties[VALIDITY_BASIS_KEY] = vbasis
                if vconf is not None:
                    properties[VALIDITY_CONFIDENCE_KEY] = vconf

        # the derived S-P-O annotation, stamped with THIS extractor's
        # identity — the stamp records what actually produced the triple, which
        # is what makes the extraction-time and backfill populations tellable
        # apart. Tolerant like every field above: a malformed or absent payload
        # (and the disabled case) simply leaves the claim un-annotated.
        structured_claim: StructuredClaim | None = None
        if structure_enabled:
            structured_claim = parse_structured_claim_payload(
                item.get("structured_claim"),
                structurizer_id=EXTRACTOR_ID,
                structurizer_version=EXTRACTOR_VERSION,
            )

        candidates.append(
            CandidateParticle(
                content=content,
                confidence_value=conf,
                uncertainty_nature=nature,
                subjects=subjects,
                assertion_modality=modality,
                stance_kind=stance_kind,
                stance_target_index=stance_target_index,
                stance_magnitude=stance_magnitude,
                valid_until=valid_until,
                structured_claim=structured_claim,
                properties=properties,
            )
        )

    return candidates, notes, shape


def candidate_to_particle(
    candidate: CandidateParticle,
    corpus_entry_id: str,
    snapshot_id: str,
    asserted_by: str = EXTRACTOR_ID,
    subject_ids: list[str] | None = None,
    extractor_ref: ExtractorRef | None = None,
    calibration: ExtractorCalibration | None = None,
) -> Particle:
    """Convert a CandidateParticle to a Particle ready for insertion.

    When ``calibration`` is None (the default and the historical
    behaviour), the constructed particle carries
    ``calibration_source=EXTRACTOR_DIRECT`` and the raw
    ``candidate.confidence_value``. When a calibration record is supplied,
    the raw value is passed through a
    :class:`particles.extraction.calibration.TemperatureScaler` and the
    particle carries ``calibration_source=CALIBRATED_BENCHMARK``,
    ``calibration_method="temperature_scaling"``, and a
    ``calibration_ref`` of the form ``"<extractor_id>:<fitted_at_iso>"``
    so the audit trail back to the fit is grep-able.

    A supplied record whose ``transform`` this SDK will not apply (
    today, every fit predating it) is treated exactly as no record: the
    particle carries the raw value stamped ``EXTRACTOR_DIRECT``. So is a record
    fitted under an extractor version other than ``extractor_ref.version``,
    or under an unknown one, when ``extractor_ref`` is given.

    A candidate that declares its own ``calibration_source`` (today,
    a migration extractor stamping ``IMPORTED``) overrides both branches: the
    raw value is stored as given and no calibration is applied, because the
    number is not a model output for a scaler to correct.
    """
    # Local import: TemperatureScaler pulls in numpy + scipy.optimize, and the
    # rest of general.py does not need them. Keeping the import lazy preserves
    # the pre-ADR-0075 import cost of this module for callers that never set a
    # calibration record.
    scaler = None
    if calibration is not None and candidate.calibration_source is None:
        from particles.extraction.calibration import scaler_for_record, version_mismatch_reason

        scaler = scaler_for_record(calibration)
        # the pipeline drops a record fitted under another extractor
        # version before it gets here (one warning per run). This is the
        # backstop for any other caller that names the running version.
        if extractor_ref is not None and version_mismatch_reason(
            calibration, extractor_ref.version
        ):
            scaler = None

    # the extractor declared where this number came from, and it is
    # not a model output — a migration's flat import floor, not a logit.
    # Temperature scaling a value no model produced would be meaningless, so a
    # declared source suppresses calibration entirely (the `scaler` guard
    # above) and the raw value is stored as given. First, because it outranks
    # both calibration branches below.
    if candidate.calibration_source is not None:
        confidence = Confidence(
            value=candidate.confidence_value,
            calibration_source=candidate.calibration_source,
        )
    # `scaler_for_record` returns None for a record whose transform
    # this SDK will not apply (today: every pre-0238 fit). Falling through to
    # the EXTRACTOR_DIRECT branch is the documented fallback for an
    # uncalibrated pairing — the same state as having no record at all.
    elif calibration is not None and scaler is not None:
        confidence = Confidence(
            value=scaler.calibrate(candidate.confidence_value),
            calibration_source=CalibrationSource.CALIBRATED_BENCHMARK,
            calibration_method="temperature_scaling",
            calibration_ref=f"{asserted_by}:{calibration.fitted_at.isoformat()}",
        )
    else:
        confidence = Confidence(
            value=candidate.confidence_value,
            calibration_source=CalibrationSource.EXTRACTOR_DIRECT,
        )

    # STRUCTURED without a triple is the one state the Core
    # validator rejects, and raising here would lose every candidate in the pass
    # to one malformed sibling. Demote and warn instead — the tolerant-backstop
    # convention: never lose the claim to an annotation defect.
    # No shipped extractor can produce this; the guard is for the next producer.
    canonical_form = candidate.canonical_form
    if canonical_form is CanonicalForm.STRUCTURED and candidate.structured_claim is None:
        log.warning(
            "Candidate from %s claims canonical_form=STRUCTURED with no structured "
            "claim; demoting to PROSE. Content: %.80s",
            asserted_by,
            candidate.content,
        )
        canonical_form = CanonicalForm.PROSE

    return Particle(
        content=candidate.content,
        confidence=confidence,
        uncertainty_nature=candidate.uncertainty_nature,
        provenance=[
            ProvenanceRef(
                type=ProvenanceRefType.SOURCE,
                corpus_entry_id=corpus_entry_id,
                snapshot_id=snapshot_id,
                # the record's position inside the deposited
                # blob, so a migrated claim points at bytes the store holds
                # and hashed rather than at an unverifiable foreign reference.
                location=candidate.provenance_location,
                chunk_hash=candidate.chunk_hash,
            )
        ],
        asserted_by=asserted_by,
        schema_version=SCHEMA_VERSION,
        subject_ids=subject_ids or [],
        assertion_modality=candidate.assertion_modality,
        # the event-anchored validity boundary the parser gated in
        # (None for every claim without a genuine future-dated cue, and for the
        # disabled case). The §9.3 staleness lint + the as-of rung 3 consume it.
        valid_until=candidate.valid_until,
        particle_type=candidate.particle_type,
        extractor_ref=extractor_ref or ExtractorRef(name=EXTRACTOR_ID, version=EXTRACTOR_VERSION),
        # the pairing the completion seam stamped on this candidate.
        # ``None`` for a deterministic extractor — correct, not a gap.
        extraction_provider_model=candidate.provider_model,
        properties=candidate.properties,
        # the attributed act — who imported this record. ``None``
        # for every LLM extractor, which attributes through ``extractor_ref``.
        contributors=candidate.contributors,
        tags=candidate.tags,
        context_fingerprint=candidate.context_fingerprint,
        # the annotation, with its subject term now bound to the
        # Subject UUID the pipeline just resolved (``subject_ids`` is
        # positionally aligned with ``candidate.subjects``, and the candidate's
        # external refs give the URI rung its keys).
        structured_claim=(
            bind_subject_id(
                candidate.structured_claim,
                candidate.subjects,
                subject_ids or [],
                external_refs=candidate.external_refs,
            )
            if candidate.structured_claim is not None
            else None
        ),
        # PROSE for every LLM-driven extractor; STRUCTURED only
        # from a structure-native parser, which always carries the triple.
        canonical_form=canonical_form,
    )
