"""Structured-claim production — the S-P-O rendering of a prose claim.

Two producers share this module, and they are the **only** two places a
structured claim may ever be generated:

1. **Extraction time** — the general extractor's reply carries a
   ``structured_claim`` object per candidate, parsed here by
   :func:`parse_structured_claim_payload`. No extra LLM call: the triple rides
   the reply the SDK is already paying for.
2. **Backfill** — :func:`structure_content` calls the LLM once over stored
   ``content`` for a particle that has no annotation (or carries a superseded
   structurizer version). Driven by ``particles structure``.

Client layer: no store, corpus, db or ingest import. The triple this
module produces carries *unresolved* terms only — binding a subject term to a
Subject UUID is the Engine's job, done in ``ingest.pipeline`` and
``operations.structure`` where the Subject store is reachable.

**The annotation is never an assertion.** Nothing here writes ``content``,
``confidence`` or provenance. A parse failure, an LLM failure, or
prose with no honest triple all produce ``None`` — absence is a legal permanent
state, and a fabricated triple is worse than none.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Mapping
from typing import Any

from particles.core.jsonld_context import is_published_curie
from particles.core.schema import ClaimTerm, ExternalRef, StructuredClaim, TermKind

log = logging.getLogger(__name__)

#: Identity of the standalone backfill structurizer. Extraction-time triples are
#: stamped with the *extractor's* id instead — the stamp records what actually
#: produced the triple, so the two populations stay tellable apart.
STRUCTURIZER_ID = "content-structurizer"

#: Bumped whenever the prompt or the parsing contract changes in a way that
#: makes previously-generated triples worth regenerating. ``particles structure
#: --structurizer-version <old>`` then discovers them, mirroring
#: ``reindex --extractor-version``.
STRUCTURIZER_VERSION = "1.0.0"

# A term is a URI when it is an absolute IRI or a CURIE in a prefix the
# published JSON-LD context knows. Anything else the model offers is recorded
# honestly as a TOKEN rather than being coerced into a namespace we would then
# have to defend. The prefix set is *read from* the artifact rather than
# hand-listed here: the hand-list had drifted to name six prefixes
# the context did not publish, so a `wd:` predicate was a URI term nothing could
# expand — exactly the state rule exists to prevent.
_ABSOLUTE_IRI = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://")


def _looks_like_uri(value: str) -> bool:
    return bool(_ABSOLUTE_IRI.match(value)) or is_published_curie(value)


def _parse_term(raw: Any, *, position: str) -> ClaimTerm | None:
    """Parse one term. ``None`` when the shape is unusable.

    Accepts either a bare string (kind inferred: URI when it looks like one,
    else TOKEN in subject/predicate position and LITERAL in object position) or
    an object ``{"kind": …, "value": …, "datatype": …, "language": …}``. The
    bare-string form is what an LLM reliably produces; the object form is what a
    structured extractor will emit.
    """
    if isinstance(raw, str):
        value = raw.strip()
        if not value:
            return None
        if _looks_like_uri(value):
            return ClaimTerm(kind=TermKind.URI, value=value)
        if position == "object":
            return ClaimTerm(kind=TermKind.LITERAL, value=value)
        return ClaimTerm(kind=TermKind.TOKEN, value=value)

    if not isinstance(raw, dict):
        return None
    value = str(raw.get("value", "")).strip()
    if not value:
        return None
    kind_raw = str(raw.get("kind", "")).strip().upper()
    try:
        kind = TermKind(kind_raw)
    except ValueError:
        kind = TermKind.URI if _looks_like_uri(value) else TermKind.TOKEN
    # A literal is only legal in object position; anything else demotes to a
    # token rather than failing the whole triple.
    if kind is TermKind.LITERAL and position != "object":
        kind = TermKind.TOKEN
    datatype = raw.get("datatype") or None
    language = raw.get("language") or None
    if kind is not TermKind.LITERAL:
        datatype = language = None
    elif datatype is not None and language is not None:
        # RDF permits one or the other. Prefer the datatype; a language tag on a
        # typed literal is the more common model slip.
        language = None
    try:
        return ClaimTerm(kind=kind, value=value, datatype=datatype, language=language)
    except ValueError as exc:  # pragma: no cover — the coercions above prevent it
        log.debug("Discarding malformed %s term: %s", position, exc)
        return None


def parse_structured_claim_payload(
    raw: Any,
    *,
    structurizer_id: str,
    structurizer_version: str,
) -> StructuredClaim | None:
    """Parse a model-supplied triple into a stamped :class:`StructuredClaim`.

    Tolerant by design, in the house convention of
    ``_parse_extraction_response``: a missing, malformed, or partial payload
    returns ``None`` and the caller keeps the claim un-annotated. The claim is
    never lost to a structurizing failure.

    Args:
        raw: the ``structured_claim`` value from a model reply — an object with
            ``subject`` / ``predicate`` / ``object``, or ``None``.
        structurizer_id: what produced it (an extractor id, or STRUCTURIZER_ID).
        structurizer_version: that producer's version.

    Returns:
        The stamped annotation, or ``None`` when no usable triple was supplied.
    """
    if not isinstance(raw, dict):
        return None
    subject = _parse_term(raw.get("subject"), position="subject")
    predicate = _parse_term(raw.get("predicate"), position="predicate")
    obj = _parse_term(raw.get("object"), position="object")
    if subject is None or predicate is None or obj is None:
        return None
    try:
        return StructuredClaim(
            subject=subject,
            predicate=predicate,
            object=obj,
            structurizer_id=structurizer_id,
            structurizer_version=structurizer_version,
        )
    except ValueError as exc:  # pragma: no cover — position coercion prevents it
        log.debug("Discarding malformed structured claim: %s", exc)
        return None


def bind_subject_id(
    claim: StructuredClaim,
    subject_names: list[str],
    subject_ids: list[str],
    external_refs: Mapping[str, ExternalRef] | None = None,
) -> StructuredClaim:
    """Bind the triple's subject term to a resolved Subject UUID.

    The structurizer emits an unresolved subject *term*; this is where it
    becomes a join back into the graph, so ``L-STR-11`` can check membership
    and a future relational query can plan as an indexed join.

    Two rungs, tried in order:

    1. **By name** — case-insensitive on the trimmed name, against the very
       names the pipeline just resolved. This is the LLM-structurizer path: the
       model was asked to reuse one of those names verbatim.
    2. **By URI** — when the subject term is a ``URI``, match it
       against the URIs in the candidate's external refs. A structure-native
       parser emits an IRI subject term while the candidate's *names* are
       human-readable labels, so rung 1 could never bind for exactly the
       population carrying the best keys. Comparison is exact: an IRI differing
       by case or trailing slash is a different resource, not the same one
       spelled differently.

    No match leaves ``subject_id`` at ``None``: the honest record of "the
    subject term resolved to nothing", which lint deliberately does not flag
    (a store may simply have no Subject for it).

    Args:
        claim: the parsed annotation.
        subject_names: the candidate's subject names, in resolution order.
        subject_ids: the resolved Subject UUIDs, positionally aligned.
        external_refs: the candidate's name → :class:`ExternalRef` map, whose
            ``uri`` values give rung 2 its keys. ``None`` skips that rung.

    Returns:
        The annotation, with ``subject_id`` set when the term matched.
    """
    wanted = claim.subject.value.strip()
    folded = wanted.casefold()
    for name, subject_id in zip(subject_names, subject_ids, strict=False):
        if name.strip().casefold() == folded:
            return claim.model_copy(update={"subject_id": subject_id})

    if external_refs and claim.subject.kind is TermKind.URI:
        for name, subject_id in zip(subject_names, subject_ids, strict=False):
            ref = external_refs.get(name)
            if ref is not None and ref.uri == wanted:
                return claim.model_copy(update={"subject_id": subject_id})
    return claim


_STRUCTURIZE_PROMPT = """You render a single knowledge claim as one
subject-predicate-object triple. You do not judge, correct, or extend the
claim — only re-express its relational core.

Rules:
- Emit ONE triple, for the claim's single main assertion.
- subject: the entity the claim is about. Prefer one of the SUBJECTS listed
  below, verbatim, so the triple binds to the same entity the claim does. Use a
  URI (e.g. wd:Q42) only if the claim itself supplies one.
- predicate: the relation, as a short lowercase verb phrase ("was minted at",
  "has weight"). Use an ontology URI only when the claim itself names one.
- object: the value or entity the relation points at.
- Return null — not a guess — when the claim has no single clean triple: a
  compound statement, a hedge, an evaluation, a first-person report, or
  anything whose relational core you would have to invent. A missing triple is
  harmless; a wrong one is a false statement this system will publish and act
  on.

Return ONLY a JSON object, no prose before or after:
{"subject": "<entity>", "predicate": "<relation>", "object": "<value>"}
or
null"""


async def structure_content(content: str, subject_names: list[str]) -> StructuredClaim | None:
    """Render one stored claim as a triple — the backfill call.

    One LLM completion on the ``extraction`` purpose; no new purpose
    is minted. Every failure mode — API error, unparseable reply, an explicit
    ``null`` from the model — returns ``None``, leaving the particle exactly as
    it was.

    Args:
        content: the particle's claim text. Read only.
        subject_names: canonical names of the particle's resolved subjects,
            offered to the model so the subject term matches an entity the
            store already knows.

    Returns:
        The stamped annotation, or ``None``.
    """
    from particles.llm import complete, fenced_prompt

    instructions = _STRUCTURIZE_PROMPT
    if subject_names:
        instructions += "\n\nSUBJECTS: " + ", ".join(subject_names)
    # Same F3 hardening as extraction: trusted instructions in the system turn,
    # the stored claim (which came from an untrusted source) fenced in the user
    # turn, so an injected instruction inside a claim cannot steer the call.
    system, user = fenced_prompt(instructions, content, label="claim")
    try:
        raw = await complete("extraction", user, max_tokens=512, system=system)
    except Exception as exc:
        log.warning("Structurizer call failed: %s", exc)
        return None

    payload = _extract_json_object(raw)
    if payload is None:
        return None
    return parse_structured_claim_payload(
        payload,
        structurizer_id=STRUCTURIZER_ID,
        structurizer_version=STRUCTURIZER_VERSION,
    )


def _extract_json_object(raw: str) -> dict[str, Any] | None:
    """Pull the JSON object out of a model reply, tolerating fences and prose."""
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text).strip()
    if not text or text == "null":
        return None
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        parsed = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        log.debug("Structurizer reply was not JSON: %s", exc)
        return None
    return parsed if isinstance(parsed, dict) else None
