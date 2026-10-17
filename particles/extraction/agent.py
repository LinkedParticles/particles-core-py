# SPDX-FileCopyrightText: 2026 The Particles authors
# SPDX-License-Identifier: Apache-2.0

"""Agent-side extraction: the session's own agent produces the candidates.

The in-session Claude Code agent plays the candidate-producer role for a span
of its own deposited transcript. This module is the store-free half (Client
layer):

* the **prompt** the Stop hook hands the agent, built from the general
  extractor's configured rules so both extractors claim at one granularity;
* the **evidence verifier**, the load-bearing safety rule: every candidate
  must quote the span verbatim, and a quote found only on a tool-call summary
  line does not count;
* the **coverage rule**, which decides whether a submission may close its span;
* two extractor plugins under one id: :class:`AgentExtractor`, which runs the
  same prompt through the ``extraction`` purpose so the maintainers can
  benchmark and calibrate a model on the API (the proxy the shipped maps are
  fitted with), and :class:`SubmittedCandidatesExtractor`, which hands the
  pipeline a submission the agent already made.

Neither plugin is ever routed to. :class:`AgentExtractor` accepts every source
type, so a benchmark suite runs every case through it, and carries
``ROUTED = False``, which :func:`~particles.extraction.registry.select_extractor`
honours.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import Any

from particles.config import AgentExtractionConfig, get_config
from particles.core.schema import Snapshot
from particles.extraction.components import assemble_prompt
from particles.extraction.general import (
    PROMPT_SCHEMA,
    CandidateParticle,
    ExtractionResult,
    _configured_prompt_components,
    _drop_session_state_candidates,
    _parse_extraction_response,
    content_to_text,
)
from particles.extraction.tool_turns import _TOOL_BRACKET, _TOOL_LINE
from particles.llm.fencing import data_fence_instruction, fence, make_nonce

log = logging.getLogger(__name__)

EXTRACTOR_ID = "claude-code-agent"
#: The agent prompt's version. A calibration map applies only to the version
#: it was fitted under, so bump this on any prompt change.
EXTRACTOR_VERSION = "0.2.0"

#: The MCP tool the agent submits through.
SUBMIT_TOOL = "extraction_submit"

#: Output budget for the API-proxy run of the agent prompt.
_PROXY_MAX_TOKENS = 16_000

_EVIDENCE_RULE = """
- evidence: a passage copied **verbatim** from the source text that states the
  claim, long enough to be unambiguous (one sentence is usually right). Copy it
  exactly, character for character; do not paraphrase, abridge or join two
  passages. A claim with no single passage that states it must be left out. A
  line of the form "[tool: …]" records that a tool was called, not what anyone
  asserted, and is never evidence."""

#: The general schema's head sentence; the ask replaces it, since the agent
#: answers through a tool call rather than with a bare JSON reply.
_JSON_ONLY = "Return ONLY a JSON array. No prose before or after. Each element:"

_AGENT_SCHEMA_FIELD = ',\n  "evidence": "<verbatim passage from the source text>"'

_SUBMIT_INSTRUCTION = """\
Particles memory: before you finish, extract the claims in the session text below.

The text is the part of this conversation not yet recorded. When you have
extracted it, call the `{tool}` tool exactly once with `ask_id` set to
"{ask_id}" and `candidates` set to the JSON array described. If the text
holds no claim the rules admit, call it with an empty array. Do not summarise
what you recorded afterwards; reply with one short line.

{thoroughness}

"""

#: Shared by the ask and the API proxy, so the proxy's benchmark and
#: calibration measure the prompt the agent is given. The shadow week found the
#: 0.1.0 ask yielding about 0.4x the general extractor's claims on the same
#: transcripts: the agent read "durable claims ... worth recording" as an
#: invitation to curate ("keeping things modest"), took a whole span in one
#: pass where the general extractor reads 7,500-character chunks, and wrote
#: every schema field of every candidate, nulls included.
_THOROUGHNESS = """\
Extract exactly as a background extractor reading this text would: every
individual claim the rules admit, not a summary and not a selection of
highlights. Do not leave a claim out because it seems minor, obvious, or true
only for now; what is kept is decided later, not here. Work through the text
in order, a few paragraphs at a time, and finish each part before moving on.

The text is about {kchars} thousand characters long. Text like this usually
yields about {expected} claims. That is a guide to how thorough to be, never a
quota: do not invent, split, or repeat claims to reach it, and record fewer
when the text holds fewer.

Keep each candidate compact. Always give content, subjects, confidence_value
and evidence; leave out every other field whose value would be null or the
default (uncertainty_nature EPISTEMIC, scope WORLD, assertion_modality
FALSIFIABLE, polarity ASSERTED)."""


def expected_claims(span_chars: int) -> int:
    """How many claims a span this size usually yields: the ask's thoroughness guide."""
    rate = get_config().extraction.agent_expected_claims_per_kchar
    return max(1, round(rate * span_chars / 1000))


def thoroughness(span_text: str) -> str:
    """The completeness and compactness instructions for a span of this size."""
    chars = len(span_text)
    return _THOROUGHNESS.format(kchars=max(1, round(chars / 1000)), expected=expected_claims(chars))


def agent_rules(reference_date: str) -> str:
    """The agent extractor's rules: the general extractor's, plus the evidence field.

    Built from :func:`~particles.extraction.general._configured_prompt_components`,
    so every config-enabled rule (scope, modality, validity, structure) applies
    to the agent exactly as it does to the general extractor. The output schema
    gains one field, ``evidence``, which the verifier checks.
    """
    components = _configured_prompt_components(tool_turns_present=False)
    out: list[Any] = []
    for c in components:
        if c.name == PROMPT_SCHEMA:
            out.append(
                replace(c, rule=_EVIDENCE_RULE + c.rule, field=_AGENT_SCHEMA_FIELD + c.field)
            )
        else:
            out.append(c)
    return assemble_prompt(out).replace("{reference_date}", reference_date)


def build_ask(ask_id: str, span_text: str, *, reference_date: str | None = None) -> str:
    """The block reason a Stop hook returns: instructions, rules, and the fenced span."""
    nonce = make_nonce()
    date = reference_date or datetime.now(UTC).date().isoformat()
    rules = agent_rules(date).replace(_JSON_ONLY, "The `candidates` array. Each element:")
    return (
        _SUBMIT_INSTRUCTION.format(
            tool=SUBMIT_TOOL, ask_id=ask_id, thoroughness=thoroughness(span_text)
        )
        + rules
        + "\n\n"
        + data_fence_instruction(nonce)
        + "\n\n"
        + fence(span_text, nonce, label="session")
    )


# ---------------------------------------------------------------------------
# The evidence verifier
# ---------------------------------------------------------------------------

_WS = re.compile(r"\s+")


def _normalise(text: str) -> tuple[str, list[int]]:
    """Collapse whitespace runs to one space; return the text and each char's source index."""
    out: list[str] = []
    index: list[int] = []
    pending_space = False
    for i, ch in enumerate(text):
        if ch.isspace():
            pending_space = bool(out)
            continue
        if pending_space:
            out.append(" ")
            index.append(i - 1)
            pending_space = False
        out.append(ch)
        index.append(i)
    return "".join(out), index


def _is_tool_line(line: str) -> bool:
    return bool(_TOOL_BRACKET.match(line) or _TOOL_LINE.match(line))


@dataclass(frozen=True)
class EvidenceMatch:
    """Where a candidate's evidence was found: character offsets into the span text."""

    start: int
    end: int


def locate_evidence(evidence: str, span_text: str) -> EvidenceMatch | None:
    """Find ``evidence`` verbatim in ``span_text`` after whitespace normalisation.

    Returns the first occurrence that is not wholly on tool-call summary lines,
    or ``None`` when there is none. Whitespace runs compare equal to one space
    on both sides, so a quote copied across a line break still matches.
    """
    needle, _ = _normalise(evidence)
    if not needle:
        return None
    hay, index = _normalise(span_text)
    start = 0
    while (found := hay.find(needle, start)) != -1:
        s, e = index[found], index[found + len(needle) - 1] + 1
        line_start = span_text.rfind("\n", 0, s) + 1
        line_end = span_text.find("\n", e)
        lines = span_text[line_start : line_end if line_end != -1 else len(span_text)]
        if not all(_is_tool_line(line) for line in lines.splitlines() if line.strip()):
            return EvidenceMatch(start=s, end=e)
        start = found + 1
    return None


@dataclass
class VerifiedSubmission:
    """What the verifier made of one submission."""

    accepted: list[CandidateParticle] = field(default_factory=list)
    #: Character offsets into the span text, parallel to ``accepted``.
    matches: list[EvidenceMatch] = field(default_factory=list)
    submitted: int = 0
    malformed: int = 0
    no_evidence: int = 0
    tool_line_only: int = 0

    @property
    def evidence_dropped(self) -> int:
        """Candidates dropped because their evidence was absent or only on a tool line."""
        return self.no_evidence + self.tool_line_only


def verify_submission(
    raw: Sequence[Any], span_text: str, *, span_start: int = 0
) -> VerifiedSubmission:
    """Parse and verify a submitted candidate array against the span (pure).

    Each element is parsed with the general extractor's own parser, so every
    optional field means what it means there. A candidate is accepted only when
    its ``evidence`` occurs in ``span_text`` (see :func:`locate_evidence`). The
    accepted candidate's ``provenance_location`` records the evidence as byte
    offsets into the snapshot, ``span_start`` being the span's own offset.
    """
    out = VerifiedSubmission(submitted=len(raw))
    for item in raw:
        if not isinstance(item, dict):
            out.malformed += 1
            continue
        evidence = item.get("evidence")
        parsed, _notes = _parse_extraction_response(json.dumps([item]))
        if len(parsed) != 1 or not isinstance(evidence, str):
            out.malformed += 1
            continue
        match = locate_evidence(evidence, span_text)
        if match is None:
            if evidence.strip() and _normalise(evidence)[0] in _normalise(span_text)[0]:
                out.tool_line_only += 1
            else:
                out.no_evidence += 1
            continue
        candidate = parsed[0]
        prefix = span_text[: match.start].encode("utf-8")
        quoted = span_text[match.start : match.end].encode("utf-8")
        begin = span_start + len(prefix)
        candidate.provenance_location = f"bytes={begin}-{begin + len(quoted)}"
        out.accepted.append(candidate)
        out.matches.append(match)
    return out


def covers_span(verified: VerifiedSubmission, span_chars: int, cfg: AgentExtractionConfig) -> bool:
    """Whether a submission may close its span.

    Both must hold: enough accepted claims for the span's size, and not too
    many candidates dropped for evidence the span does not contain.
    """
    if verified.submitted and verified.evidence_dropped / verified.submitted > (
        cfg.max_evidence_drop_ratio
    ):
        return False
    return len(verified.accepted) >= cfg.min_claims_per_kchar * span_chars / 1000


# ---------------------------------------------------------------------------
# The plugins
# ---------------------------------------------------------------------------


class SubmittedCandidatesExtractor:
    """Hands the pipeline a submission the agent already made.

    Shares :data:`EXTRACTOR_ID` with :class:`AgentExtractor`, so a calibration
    fitted through the proxy applies to real submissions. ``read_through`` is
    where the span ends, in extraction-text characters, or ``None`` at the end
    of the snapshot.
    """

    EXTRACTOR_ID = EXTRACTOR_ID
    EXTRACTOR_VERSION = EXTRACTOR_VERSION

    def __init__(
        self,
        candidates: Sequence[CandidateParticle],
        *,
        provider_model: str,
        read_through: int | None,
        notes: Sequence[str] = (),
    ) -> None:
        self._candidates = list(candidates)
        self._provider_model = provider_model
        self._read_through = read_through
        self._notes = list(notes)

    def accepts(self, source_type: str) -> bool:
        return False

    async def extract(
        self, snapshot: Snapshot, content: bytes, **kwargs: object
    ) -> ExtractionResult:
        for candidate in self._candidates:
            candidate.provider_model = self._provider_model
        result = ExtractionResult(
            candidates=list(self._candidates),
            quality_notes=list(self._notes),
            read_through=self._read_through,
        )
        _screen_session_state(result, content, kwargs.get("source_type"))
        return result


class AgentExtractor:
    """The agent prompt run through the ``extraction`` purpose.

    Never routed to. It exists so ``particles extractor benchmark`` and
    ``particles extractor calibrate claude-code-agent`` can measure and fit a
    model on the agent's prompt through the API, which is how the shipped
    calibration maps are produced. The whole source is the span, and the same
    evidence verifier applies.
    """

    EXTRACTOR_ID = EXTRACTOR_ID
    EXTRACTOR_VERSION = EXTRACTOR_VERSION
    #: Never chosen by routing; invoked only by name (benchmark, calibrate).
    ROUTED = False

    def accepts(self, source_type: str) -> bool:
        return True

    async def extract(
        self, snapshot: Snapshot, content: bytes, **kwargs: object
    ) -> ExtractionResult:
        from particles.llm import complete_with_provider_model

        # Decoded the way the general extractor decodes a source, so an HTML
        # benchmark fixture reaches the model as text, not markup.
        text = content_to_text(content)
        nonce = make_nonce()
        date = datetime.now(UTC).date().isoformat()
        system = (
            thoroughness(text) + "\n\n" + agent_rules(date) + "\n\n" + data_fence_instruction(nonce)
        )
        try:
            raw, provider_model = await complete_with_provider_model(
                "extraction",
                fence(text, nonce, label="source"),
                max_tokens=_PROXY_MAX_TOKENS,
                system=system,
            )
        except Exception as exc:  # noqa: BLE001 — reported as a transient failure
            log.error("Agent-extractor proxy call failed: %s", exc)
            return ExtractionResult(quality_notes=[f"API error: {exc}"], transient_error_count=1)
        try:
            items = json.loads(raw.strip().removeprefix("```json").removesuffix("```"))
        except json.JSONDecodeError:
            return ExtractionResult(quality_notes=["agent proxy reply was not a JSON array"])
        verified = verify_submission(items if isinstance(items, list) else [], text)
        for candidate in verified.accepted:
            candidate.provider_model = provider_model
        notes: list[str] = []
        if verified.evidence_dropped or verified.malformed:
            notes.append(
                f"agent extractor: {verified.evidence_dropped} candidate(s) dropped for "
                f"evidence not in the source, {verified.malformed} malformed"
            )
        result = ExtractionResult(
            candidates=verified.accepted, quality_notes=notes, answered_calls=1
        )
        _screen_session_state(result, content, kwargs.get("source_type"))
        return result


def _screen_session_state(result: ExtractionResult, content: bytes, source_type: object) -> None:
    """Apply the general extractor's session-state filter to an agent result.

    The filter runs inside ``GeneralExtractor.extract``, after every text path,
    so a path that does not go through that method must apply it itself.
    """
    if source_type in get_config().extraction.session_state_source_types:
        _drop_session_state_candidates(result, content)


#: Text that opens every ask. A transcript record carrying it is the ask, never
#: a prompt a person typed (see ``ask_exclusions`` in the Claude Code adapter).
ASK_MARKER = "Particles memory: before you finish"


def purpose_servable(purpose: str) -> bool:
    """Whether this install can serve an LLM purpose at all.

    The Anthropic route needs a key. A named OpenAI-compatible provider is
    taken as servable: a local endpoint usually needs none, and whether it is
    up is not a configuration fact. Under ``network.egress``, a purpose whose
    endpoint the posture refuses is not servable, exactly as for a keyless
    install, so agent-side near pairs are held for review.
    """
    from particles.llm import purpose_allowed
    from particles.secrets import get_anthropic_api_key_optional

    if not purpose_allowed(purpose):
        return False
    selection = get_config().llm.for_purpose(purpose)
    if selection.provider == "anthropic":
        return get_anthropic_api_key_optional() is not None
    return True


@dataclass(frozen=True)
class AskDecision:
    """Whether a Stop asks, and over which byte span of the distilled transcript."""

    ask: bool
    reason: str
    start: int = 0
    end: int = 0


def decide_ask(
    *,
    text: bytes,
    covered_through: int,
    now: datetime,
    asks_at: Sequence[datetime],
    capped: bool,
    cfg: AgentExtractionConfig,
    last_ask_answered: bool = True,
    asked_through: int = 0,
    last_ask_start: int | None = None,
) -> AskDecision:
    """The ask rule (pure).

    ``covered_through`` is how far the transcript entry was already extracted,
    in bytes. ``asked_through`` is how far earlier answered asks reached, for
    the ask's own start only: the span starts at the later of the two, and
    ends at the text's end, or at the last paragraph break within
    ``max_span_chars``. Only a shadow install passes it, since a shadow
    answer writes nothing and so never moves ``covered_through``; without it
    every Stop would re-ask the same span. ``asks_at`` are this session's
    earlier asks. ``capped`` applies ``max_asks_per_session``, which only an
    install with a background extractor does. ``last_ask_answered`` says the
    session's latest ask was submitted and ``last_ask_start`` is where its span
    began; catch-up waives the interval only when that answer moved the start.
    """
    start = max(covered_through, asked_through)
    owed = len(text) - start
    if owed < cfg.min_new_chars:
        return AskDecision(False, "below threshold")
    if capped and cfg.max_asks_per_session is not None and len(asks_at) >= cfg.max_asks_per_session:
        return AskDecision(False, "session cap reached")
    # Catch-up waives the interval only after an answer that moved the start.
    # An unanswered ask, or an answer below the coverage floor that left its
    # span to the background pass, leaves the backlog where it was, so
    # waiving the interval then would re-ask the same span at every turn's end.
    progressed = last_ask_start is None or start > last_ask_start
    backlog = cfg.catch_up and owed > cfg.max_span_chars and last_ask_answered and progressed
    if asks_at and not backlog:
        since = (now - max(asks_at)).total_seconds()
        if since < cfg.min_interval_seconds:
            return AskDecision(False, "inside interval")
    end = len(text)
    if owed > cfg.max_span_chars:
        limit = start + cfg.max_span_chars
        cut = text.rfind(b"\n\n", start, limit)
        end = cut + 2 if cut > start else limit
        # Never split a UTF-8 sequence when falling back to a hard cut.
        while end > start and end < len(text) and (text[end] & 0xC0) == 0x80:
            end -= 1
    return AskDecision(True, "ask", start=start, end=end)
