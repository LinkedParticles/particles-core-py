# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Journal-aware extractor.

Selected for ``JOURNAL``-typed corpus entries — the operator-explicit genre
signal (``particles deposit --journal`` / ``--source-type JOURNAL``; fork O1).
Unlike the genre-blind general classifier, it tunes for
personal-journal prose:

* **Reification + modality.** First-person inner states are reified into
  ``EXPERIENTIAL`` particles about the author; value / preference / political
  judgements are tagged ``EVALUATIVE`` (bare — fork O3), so the engine never
  contradiction-checks or trust-arbitrates feelings and opinions.
  The default direction is inverted relative to the general classifier: in a
  journal, *unsure ⇒ non-``FALSIFIABLE``* is the safer error (a feeling left in
  the truth engine is the failure mode this was written to fix).
* **NARRATIVE graph.** One NARRATIVE candidate labels the entry; a
  ``narrative_index`` is stamped on each claim in document order. The pipeline
  writes ``PART_OF`` (constituent → narrative) and ``SEQUENCE_IN`` (predecessor
  → successor) from these once particle ids exist.

Author resolution (byline → real-person Subject) is deliberately **not** done
here — it is gated on the privacy decision (§4 / fork O2). v1
leaves journal claims author-agnostic (*"the author …"*). That gate is why this
extractor's conformance baseline fails `subject_ids`: its inner-state claims
have exactly one candidate subject and it is the one the gate withholds
(proposed).

The extractor makes a single whole-entry LLM call for entries within the chunk
size — narrative grouping (one label, one sequence) is a whole-document concern.
Over-length entries (> ``extraction.html_chunk_size``) are extracted in multiple
carry-forward passes and their per-chunk NARRATIVE fragments are
merged into one whole-entry NARRATIVE by the Engine-side post-pass
(:func:`particles.ingest.narrative_merge.collapse_chunk_narratives`).
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from particles.config import get_config
from particles.core.schema import (
    ApplicabilityClause,
    AssertionModality,
    ParticleType,
    Snapshot,
    UncertaintyNature,
)
from particles.extraction.general import (
    CandidateParticle,
    ExtractionResult,
    _split_into_paragraph_chunks,
    _strip_obsidian_frontmatter,
    content_to_text,
)
from particles.extraction.subject_scope import SUBJECT_SCOPE_KEY, SUBJECT_SCOPE_SELF

log = logging.getLogger(__name__)

SOURCE_TYPE = "JOURNAL"
EXTRACTOR_ID = "journal-extractor"
# 0.4.0: the `subjects` instruction no longer says "[] for statements
#        only about the author's own inner state *or actions*", which suppressed
#        real subjects ("The author lost eight runs" is about the game); and each
#        claim now carries `subject_scope`, recorded as `extraction:subject_scope
#        = SELF` on a subjectless author-only claim so conformance and L-STR-09
#        can tell "no subject exists for this claim" from "subject resolution
#        failed". Output differs for every entry, so `reindex --extractor-version
#        0.3.0` re-extracts.
# 0.3.0: prompt-injection hardening (security F3) — trusted rules/schema move to
#        the `system` turn and the untrusted entry is wrapped in a per-call nonce
#        fence in the user turn (was: rules + "JOURNAL ENTRY:" + raw entry in one
#        user message). Mirrors general-extractor 0.11.0; benign output unchanged.
# 0.2.0: over-length entries (> html_chunk_size) are extracted in multiple
#        carry-forward passes and merged into one whole-entry NARRATIVE
# instead of truncated to a single leading pass. Output differs
#        for over-length entries, so the version-mismatch rule lets
#        `reindex --extractor-version 0.1.1` recover the previously-dropped tails.
# 0.1.1: _parse_journal_response recovers complete claims from a truncated /
#        partially-malformed response instead of dropping the whole extraction
#        (a dense entry whose JSON output hit the model token limit previously
#        yielded 0 particles). Output differs for those cases, so the
#        version-mismatch rule lets `reindex` re-extract them.
EXTRACTOR_VERSION = "0.4.0"
# Free-text prose, same provenance trust as the general extractor.
DEFAULT_TRUST_WEIGHT = 0.70
APPLICABILITY = [
    ApplicabilityClause(
        keyword="MUST",
        domain_uri="http://www.wikidata.org/entity/Q133492",  # diary
        domain_label="personal journal / diary",
        source_types=[SOURCE_TYPE],
    )
]

# Default confidence for the entry-level NARRATIVE label. The label is a
# one-sentence compression the extractor authored from the whole entry; it is
# stated, not hedged, so it sits in the same high band as an explicitly-stated
# claim (NARRATIVE confidence keeps the universal truth-likelihood
# meaning).
_NARRATIVE_CONFIDENCE = 0.9

_JOURNAL_RULES = """\
You are a particle extractor for a PERSONAL JOURNAL entry written in the first
person. The writer is "the author". Extract every claim-granular statement, IN
THE ORDER THEY APPEAR, and classify each one's assertion_modality. The journal
entry is supplied in the user message, wrapped in a data fence (see the
SECURITY note below).

Rules:
- One particle = one statement. Reify first-person statements into the third
  person about the author: "I felt anxious" → "The author felt anxious";
  "I have to pee" → "The author needs to urinate".
- assertion_modality — classify each statement:
  - EXPERIENTIAL: a report of the author's inner state, mood, bodily sensation,
    like/dislike, or felt experience ("The author felt anxious", "The author
    does not like doing math", "The author is impatient").
  - EVALUATIVE: a value judgement, preference, ranking, aesthetic or political
    opinion stated as if fact, with no observer-independent fact of the matter
    ("Balatro is tedious", "DEI initiatives are being lost", "hope is not a
    strategy"). Tag these EVALUATIVE — do NOT leave an opinion FALSIFIABLE.
  - FALSIFIABLE: an observer-independent fact that could in principle be shown
    true or false — dates, places, events, who-did-what, durations, counts
    ("The author spent two weeks in New Jersey", "The post was written on
    August 5, 2025").
  - CONSTITUTIVE: a rule or definition the text establishes (rare in journals).
  - WHEN UNSURE between FALSIFIABLE and a non-FALSIFIABLE tag, choose the
    non-FALSIFIABLE tag. A misplaced feeling/opinion in the fact engine is the
    error to avoid; a fact tagged EXPERIENTIAL/EVALUATIVE is harmless here.
- confidence_value [0.0–1.0]: how clearly and definitively the statement (or the
  author's having the feeling/opinion) is expressed in the source.
- uncertainty_nature: EPISTEMIC, or ALEATORY for genuinely random/future events.
- subjects: real-world entities the statement is about (games, medications,
  places, named people OTHER than the author). Name EVERY world entity the
  statement touches, even when the sentence is grammatically about the author:
  "The author lost eight runs in a row" is about the game; "The author flew home
  to Seattle" is about Seattle. Use [] only when the statement touches no world
  entity at all ("The author woke before the alarm", "The author felt anxious").
- subject_scope: "SELF" when the statement is about the author and nothing else
  — no place, no person, no object, no work. "WORLD" otherwise. A statement can
  be about the author's inner state AND about a world entity; that is "WORLD"
  with the entity listed in subjects. Do not use "SELF" to avoid naming a
  subject you are unsure of — leave subjects [] and say "WORLD" instead.
- narrative_label: ONE sentence capturing the entry's overall arc or theme — the
  connective tissue that makes these statements one journal entry rather than a
  loose pile of claims.
- Preserve URLs verbatim including the scheme."""

_JOURNAL_SCHEMA = """

Return ONLY a JSON object. No prose before or after:
{
  "narrative_label": "<one sentence describing the whole entry>",
  "claims": [
    {
      "content": "<the statement as a complete sentence about the author or world>",
      "subjects": ["<entity name>"],
      "subject_scope": "SELF" or "WORLD",
      "confidence_value": <float 0.0-1.0>,
      "uncertainty_nature": "EPISTEMIC" or "ALEATORY",
      "assertion_modality": "FALSIFIABLE" | "EVALUATIVE" | "EXPERIENTIAL" | "CONSTITUTIVE"
    }
  ]
}"""


def _build_journal_prompt() -> str:
    return _JOURNAL_RULES + _JOURNAL_SCHEMA


class JournalExtractor:
    """Whole-entry extractor for personal-journal prose.

    Accepts only the ``JOURNAL`` source_type, so it is selected ahead of the
    general fallback solely for entries the operator (or, in a later release, a
    journal importer) tagged as journals. Disabling it via
    ``journal_extractor.enabled = false`` makes ``accepts`` return ``False``, so
    ``JOURNAL`` entries fall through to the general extractor unchanged.
    """

    EXTRACTOR_ID: str = EXTRACTOR_ID
    EXTRACTOR_VERSION: str = EXTRACTOR_VERSION
    DEFAULT_TRUST_WEIGHT: float = DEFAULT_TRUST_WEIGHT
    APPLICABILITY = APPLICABILITY

    def accepts(self, source_type: str) -> bool:
        return source_type == SOURCE_TYPE and get_config().journal_extractor.enabled

    async def extract(
        self,
        snapshot: Snapshot,
        content: bytes,
        **kwargs: object,
    ) -> ExtractionResult:
        """LLM extraction → claims + NARRATIVE candidate(s).

        Entries within ``html_chunk_size`` take a single whole-entry call;
        over-length entries are chunked through ``extract_with_carry_forward``
         and their per-chunk NARRATIVE fragments are merged downstream
        by the Engine post-pass. ``session`` / ``corpus_entry_id`` (read from
        kwargs, passed by the pipeline) drive the carry-forward cache.
        """
        source_type = kwargs.get("source_type")
        is_markdown = isinstance(source_type, str) and source_type == "LOCAL_MARKDOWN"
        session_obj = kwargs.get("session")
        session: AsyncSession | None = (
            session_obj if isinstance(session_obj, AsyncSession) else None
        )
        entry_obj = kwargs.get("corpus_entry_id")
        corpus_entry_id: str | None = entry_obj if isinstance(entry_obj, str) else None
        # Reindex threads its supersede set so carry-forward treats the
        # marked particles as absent (see extract_with_carry_forward).
        sup_obj = kwargs.get("supersede_ids")
        supersede_ids: frozenset[str] = sup_obj if isinstance(sup_obj, frozenset) else frozenset()
        text = self._normalise_text(content, is_markdown=is_markdown)
        if not text.strip():
            return ExtractionResult(quality_notes=["Empty content"])
        return await self._extract_claims(
            text,
            session=session,
            corpus_entry_id=corpus_entry_id,
            supersede_ids=supersede_ids,
        )

    def _normalise_text(self, content: bytes, *, is_markdown: bool) -> str:
        """Decode bytes to prose (HTML→markdown via the shared helper)."""
        try:
            text = content_to_text(content)
        except Exception as exc:  # pragma: no cover - decode guard mirrors general
            log.warning("Journal decode error: %s", exc)
            return ""
        if is_markdown:
            _, text = _strip_obsidian_frontmatter(text)
        return text

    async def _extract_claims(
        self,
        text: str,
        *,
        session: AsyncSession | None = None,
        corpus_entry_id: str | None = None,
        supersede_ids: frozenset[str] = frozenset(),
    ) -> ExtractionResult:
        """Single whole-entry pass for short entries; multi-pass for long ones.

        Entries over ``html_chunk_size`` route through :meth:`_extract_chunked`
         instead of being truncated to the leading slice.
        """
        cfg = get_config().extraction
        if len(text) > cfg.html_chunk_size:
            return await self._extract_chunked(
                text,
                session=session,
                corpus_entry_id=corpus_entry_id,
                supersede_ids=supersede_ids,
            )
        candidates, notes, transient = await _call_journal_llm(text)
        return ExtractionResult(
            candidates=candidates,
            quality_notes=notes,
            transient_error_count=1 if transient else 0,
        )

    async def _extract_chunked(
        self,
        text: str,
        *,
        session: AsyncSession | None,
        corpus_entry_id: str | None,
        supersede_ids: frozenset[str] = frozenset(),
    ) -> ExtractionResult:
        """Paragraph-chunked multi-pass journal extraction with carry-forward.

        Over-length entries are split on paragraph boundaries and routed through
        :func:`extract_with_carry_forward` with the journal prompt (2). Each cache-miss chunk yields its claims (chunk-local
        ``narrative_index``) plus one per-chunk NARRATIVE candidate; the Engine
        post-pass
        (:func:`particles.ingest.narrative_merge.collapse_chunk_narratives`)
        collapses those fragments into one whole-entry NARRATIVE with a global
        ``SEQUENCE_IN`` order. Chunks beyond ``max_llm_calls_per_source`` emit
        the shared ``CHUNK_TRUNCATION`` note — the new, much higher truncation
        ceiling that replaces the old single-pass cap.
        """
        # Deferred import (cycle break — case 1): incremental imports
        # CandidateParticle from general, which journal also imports.
        from particles.extraction.incremental import ChunkUnit, extract_with_carry_forward

        cfg = get_config().extraction
        chunk_texts = _split_into_paragraph_chunks(text, cfg.html_chunk_size)
        chunks = [
            ChunkUnit(chunk_id=f"chunk_{i}", chunk_text=t)
            for i, t in enumerate(chunk_texts, start=1)
        ]
        log.info(
            "JournalExtractor: %d chunk(s) after paragraph splitting (input %d chars)",
            len(chunks),
            len(text),
        )
        return await extract_with_carry_forward(
            session=session,
            chunks=chunks,
            corpus_entry_id=corpus_entry_id,
            extractor_id=EXTRACTOR_ID,
            extractor_version=EXTRACTOR_VERSION,
            max_llm_calls=cfg.max_llm_calls_per_source,
            call_llm=_call_journal_llm,
            supersede_ids=supersede_ids,
        )


async def _call_journal_llm(text: str) -> tuple[list[CandidateParticle], list[str], bool]:
    """Run one journal-prompt LLM call → ``(candidates, notes, transient_error)``.

    The per-chunk caller injected into :func:`extract_with_carry_forward`
    ; mirrors the general extractor's ``_call_llm`` but builds the
    journal prompt and parses via :func:`_parse_journal_response`. ``transient``
    is True when the API call raised, so the pipeline resets the snapshot to
    PENDING for retry rather than stamping it COMPLETE with partial output.
    """
    from particles.llm import (
        AccountLevelLLMError,
        complete_with_provider_model,
        fenced_prompt,
        is_account_level_failure,
    )

    cfg = get_config().extraction
    # F3 hardening: trusted rules/schema in the ``system`` turn, the untrusted
    # journal entry alone in the user turn behind a per-call nonce fence (was:
    # rules + "JOURNAL ENTRY:" + raw entry in one user message). Mirrors the
    # general extractor's ``_call_llm``.
    system, user = fenced_prompt(_build_journal_prompt(), text, label="journal_entry")
    # the journal extractor's own completion seam; stamped
    # here for the same reason as ``general.py::_call_llm``.
    provider_model = ""
    try:
        raw, provider_model = await complete_with_provider_model(
            "extraction", user, max_tokens=cfg.max_tokens, system=system
        )
    except Exception as exc:
        # Account-level failures abort rather than degrading to a per-call
        # transient — see the matching branch in ``general.py::_call_llm``.
        if is_account_level_failure(exc):
            log.error("Journal extraction unavailable (account-level): %s", exc)
            raise AccountLevelLLMError(exc) from exc
        log.error("Journal extraction API call failed: %s", exc)
        return [], [f"API error: {exc}"], True

    log.debug("Raw journal LLM response (%d chars):\n%s", len(raw), raw)
    candidates, notes = _parse_journal_response(raw)
    for candidate in candidates:
        candidate.provider_model = provider_model
    return candidates, notes, False


def _subject_scope_properties(
    item: dict[str, Any], subjects: list[str]
) -> dict[str, object] | None:
    """Record an author-scoped claim or ``None`` for the normal case.

    Two conditions, both required. The model must say ``SELF`` — absence, an
    unknown value, and ``WORLD`` all mean the ordinary case, so a reply from a
    prompt that predates this axis behaves exactly as before. And the claim must
    carry **no subjects**: the key asserts that the claim's only available
    subject is the author, which is false of a claim that just named Balatro.

    That second condition is also what keeps the key from becoming a way to
    clear the `subject_ids` floor. It can only ever apply to a claim
    that would otherwise count as an unpopulated REQUIRED field — which is the
    point — so it must not additionally be settable on claims that already pass.
    """
    if subjects:
        return None
    if str(item.get("subject_scope", "")).strip().upper() != SUBJECT_SCOPE_SELF:
        return None
    return {SUBJECT_SCOPE_KEY: SUBJECT_SCOPE_SELF}


def _parse_journal_response(raw: str) -> tuple[list[CandidateParticle], list[str]]:
    """Parse the journal LLM object into claim candidates + one NARRATIVE.

    Claims become ``CLAIM`` candidates carrying a 0-based ``narrative_index`` in
    document order. A single ``NARRATIVE`` candidate is appended (last, for
    deterministic ordering) when a non-empty label and at least one claim are
    present — a narrative with no constituents is pointless. Default-safe: an
    unknown / missing ``assertion_modality`` falls back to ``FALSIFIABLE`` (the
    parser-level safety net; the prompt asks the model to prefer non-FALSIFIABLE
    when unsure).

    **Resilient (v0.1.1):** when the whole response is not valid JSON
    — truncated at the model's output-token limit, or carrying a single
    malformed claim (e.g. an unescaped quote) — it recovers every *complete*
    claim object plus the label instead of dropping the entire extraction. A
    dense entry then loses at most the truncated tail / one bad claim, not all
    of them. This mirrors the general extractor's array-truncation recovery.
    """
    notes: list[str] = []
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

    claims_raw: list[Any]
    label: str
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        # Recover what came back rather than discarding it all.
        claims_raw = _salvage_claim_objects(raw)
        label = _salvage_narrative_label(raw)
        if not claims_raw:
            log.warning("Journal response unparseable and nothing recoverable: %s", exc)
            return [], [f"JSON parse error: {exc}"]
        notes.append(
            f"Journal response was not valid JSON ({exc.msg}); salvaged "
            f"{len(claims_raw)} complete claim(s) from the truncated/malformed output"
        )
    else:
        if not isinstance(data, dict):
            return [], ["Journal response is not a JSON object"]
        raw_claims = data.get("claims", [])
        if not isinstance(raw_claims, list):
            return [], ["Journal response 'claims' is not an array"]
        claims_raw = raw_claims
        label = str(data.get("narrative_label", "")).strip()

    candidates: list[CandidateParticle] = []
    for i, item in enumerate(claims_raw):
        if not isinstance(item, dict):
            notes.append(f"Claim {i} is not an object; skipped")
            continue
        content = str(item.get("content", "")).strip()
        if not content:
            notes.append(f"Claim {i} has empty content; skipped")
            continue
        try:
            conf = max(0.0, min(1.0, float(item.get("confidence_value", 0.5))))
        except (TypeError, ValueError):
            conf = 0.5
            notes.append(f"Claim {i} has invalid confidence_value; defaulted to 0.5")
        try:
            nature = UncertaintyNature(str(item.get("uncertainty_nature", "EPISTEMIC")))
        except ValueError:
            nature = UncertaintyNature.EPISTEMIC
            notes.append(f"Claim {i} has invalid uncertainty_nature; defaulted to EPISTEMIC")
        try:
            modality = AssertionModality(
                str(item.get("assertion_modality", "FALSIFIABLE")).strip().upper()
            )
        except ValueError:
            modality = AssertionModality.FALSIFIABLE
            notes.append(f"Claim {i} has invalid assertion_modality; defaulted to FALSIFIABLE")
        subjects_raw = item.get("subjects", [])
        subjects: list[str] = (
            [str(s) for s in subjects_raw if str(s).strip()]
            if isinstance(subjects_raw, list)
            else []
        )
        candidates.append(
            CandidateParticle(
                content=content,
                confidence_value=conf,
                uncertainty_nature=nature,
                subjects=subjects,
                assertion_modality=modality,
                narrative_index=len(candidates),
                properties=_subject_scope_properties(item, subjects),
            )
        )

    if label and candidates:
        candidates.append(
            CandidateParticle(
                content=label,
                confidence_value=_NARRATIVE_CONFIDENCE,
                uncertainty_nature=UncertaintyNature.EPISTEMIC,
                subjects=[],
                particle_type=ParticleType.NARRATIVE,
            )
        )
    elif not label:
        notes.append("Journal response missing narrative_label; no NARRATIVE emitted")

    return candidates, notes


def _salvage_claim_objects(raw: str) -> list[dict[str, Any]]:
    """Recover complete claim objects from a malformed / truncated claims array.

    Walks the ``claims`` array with ``JSONDecoder.raw_decode``, parsing one
    object at a time starting at each ``{``. A single unparseable object (e.g.
    an unescaped quote in ``content``) or a truncated tail object costs that one
    claim — the scan resyncs at the next ``{`` and keeps going — rather than
    failing the whole response. Objects are returned in document order; only
    those with non-empty ``content`` are kept (so a stray non-claim object never
    becomes a particle).
    """
    match = re.search(r'"claims"\s*:\s*\[', raw)
    if match is None:
        return []
    decoder = json.JSONDecoder()
    objs: list[dict[str, Any]] = []
    pos = match.end()
    while True:
        start = raw.find("{", pos)
        if start == -1:
            break
        try:
            obj, end = decoder.raw_decode(raw, start)
        except json.JSONDecodeError:
            pos = start + 1  # resync at the next object start
            continue
        if isinstance(obj, dict) and str(obj.get("content", "")).strip():
            objs.append(obj)
        pos = end
    return objs


def _salvage_narrative_label(raw: str) -> str:
    """Recover the ``narrative_label`` string from an unparseable response.

    The label sits near the top of the object, so it almost always survives a
    truncated tail. Matches a JSON string body (honouring escapes) and unescapes
    it; returns ``""`` when absent.
    """
    match = re.search(r'"narrative_label"\s*:\s*"((?:[^"\\]|\\.)*)"', raw)
    if match is None:
        return ""
    try:
        return str(json.loads(f'"{match.group(1)}"')).strip()
    except json.JSONDecodeError:
        return match.group(1).strip()
