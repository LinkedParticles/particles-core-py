# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""The store-free half of agent-side extraction.

The evidence verifier is the load-bearing safety rule, so its edges are pinned:
whitespace-insensitive matching, a quote only on a tool-call summary line, a
paraphrase, and byte offsets past multi-byte text. The ask rule and the
coverage rule are pinned against the values the defaults were measured with.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from particles.config import AgentExtractionConfig
from particles.extraction import agent
from particles.extraction.registry import get_extractors, select_extractor

SPAN = (
    "## User\n\nWe moved the staging database to Postgres 17 last week.\n\n"
    "## Assistant\n\nNoted. [tool: Bash — psql --version]\n\n"
    "[tool: Read — config/postgres-18.yaml]\n\n"
    "The deploy pipeline now runs   nightly at 02:00 UTC.\n"
)


def _candidate(content: str, evidence: Any, **extra: Any) -> dict[str, Any]:
    return {
        "content": content,
        "subjects": ["staging database"],
        "confidence_value": 0.9,
        "uncertainty_nature": "EPISTEMIC",
        "evidence": evidence,
        **extra,
    }


class TestLocateEvidence:
    def test_verbatim_quote_is_found(self) -> None:
        match = agent.locate_evidence("moved the staging database to Postgres 17", SPAN)
        assert match is not None
        assert SPAN[match.start : match.end] == "moved the staging database to Postgres 17"

    def test_whitespace_runs_compare_equal(self) -> None:
        match = agent.locate_evidence("runs nightly\nat 02:00 UTC", SPAN)
        assert match is not None
        assert SPAN[match.start : match.end] == "runs   nightly at 02:00 UTC"

    def test_a_quote_only_on_a_tool_line_is_not_evidence(self) -> None:
        assert agent.locate_evidence("config/postgres-18.yaml", SPAN) is None

    def test_a_paraphrase_is_not_evidence(self) -> None:
        assert agent.locate_evidence("staging now runs Postgres 17", SPAN) is None

    def test_empty_evidence_is_not_evidence(self) -> None:
        assert agent.locate_evidence("   ", SPAN) is None


class TestVerifySubmission:
    def test_counts_each_drop_reason(self) -> None:
        raw = [
            _candidate("Staging runs Postgres 17.", "the staging database to Postgres 17"),
            _candidate("Postgres 18 is configured.", "config/postgres-18.yaml"),
            _candidate("Staging is on Postgres 16.", "staging is on 16"),
            _candidate("No evidence field.", None),
            "not an object",
        ]
        verified = agent.verify_submission(raw, SPAN)
        assert verified.submitted == 5
        assert [c.content for c in verified.accepted] == ["Staging runs Postgres 17."]
        assert verified.tool_line_only == 1
        assert verified.no_evidence == 1
        assert verified.malformed == 2
        assert verified.evidence_dropped == 2

    def test_location_is_byte_offsets_into_the_snapshot(self) -> None:
        span = "## User\n\nCafé opens at 08:00 on weekdays.\n"
        verified = agent.verify_submission(
            [_candidate("The café opens at 08:00.", "opens at 08:00")], span, span_start=100
        )
        location = verified.accepted[0].provenance_location
        begin = 100 + len(span[: span.index("opens")].encode())
        assert location == f"bytes={begin}-{begin + len(b'opens at 08:00')}"

    def test_optional_fields_parse_as_the_general_extractor_parses_them(self) -> None:
        verified = agent.verify_submission(
            [_candidate("Staging runs Postgres 17.", "Postgres 17", uncertainty_nature="ALEATORY")],
            SPAN,
        )
        assert verified.accepted[0].uncertainty_nature.value == "ALEATORY"

    def test_a_compact_candidate_takes_the_defaults(self) -> None:
        # The 0.2.0 ask tells the agent to leave out null and default fields.
        compact = {
            "content": "Staging runs Postgres 17.",
            "subjects": ["staging database"],
            "confidence_value": 0.9,
            "evidence": "Postgres 17",
        }
        verified = agent.verify_submission([compact], SPAN)
        assert len(verified.accepted) == 1 and not verified.malformed
        assert verified.accepted[0].uncertainty_nature.value == "EPISTEMIC"


class TestCoverage:
    cfg = AgentExtractionConfig(min_claims_per_kchar=2.0, max_evidence_drop_ratio=0.5)

    def _verified(self, accepted: int, dropped: int) -> agent.VerifiedSubmission:
        v = agent.VerifiedSubmission(submitted=accepted + dropped, no_evidence=dropped)
        v.accepted = [object()] * accepted  # type: ignore[list-item]
        return v

    def test_enough_claims_covers(self) -> None:
        assert agent.covers_span(self._verified(20, 0), 10_000, self.cfg)

    def test_too_few_claims_does_not(self) -> None:
        assert not agent.covers_span(self._verified(19, 0), 10_000, self.cfg)

    def test_too_many_evidence_drops_does_not(self) -> None:
        assert not agent.covers_span(self._verified(20, 21), 10_000, self.cfg)


class TestDecideAsk:
    now = datetime(2026, 10, 9, 12, tzinfo=UTC)

    def _decide(self, text: bytes, covered: int = 0, **kw: Any) -> agent.AskDecision:
        cfg = AgentExtractionConfig(
            min_new_chars=100,
            min_interval_seconds=600,
            max_span_chars=1000,
            max_asks_per_session=2,
            **kw.pop("cfg", {}),
        )
        return agent.decide_ask(
            text=text,
            covered_through=covered,
            now=self.now,
            asks_at=kw.pop("asks_at", []),
            capped=kw.pop("capped", True),
            cfg=cfg,
            last_ask_answered=kw.pop("last_ask_answered", True),
            asked_through=kw.pop("asked_through", 0),
            last_ask_start=kw.pop("last_ask_start", None),
        )

    def test_below_threshold(self) -> None:
        assert not self._decide(b"x" * 150, covered=100).ask

    def test_asks_over_the_owed_text(self) -> None:
        d = self._decide(b"x" * 600, covered=100)
        assert (d.ask, d.start, d.end) == (True, 100, 600)

    def test_interval_holds_unless_a_backlog_is_owed(self) -> None:
        recent = [self.now - timedelta(seconds=60)]
        assert not self._decide(b"x" * 600, asks_at=recent).ask
        assert self._decide(b"x" * 3000, asks_at=recent).ask
        assert not self._decide(b"x" * 3000, asks_at=recent, cfg={"catch_up": False}).ask

    def test_an_unanswered_ask_never_waives_the_interval(self) -> None:
        recent = [self.now - timedelta(seconds=60)]
        assert not self._decide(b"x" * 3000, asks_at=recent, last_ask_answered=False).ask
        later = [self.now - timedelta(seconds=700)]
        assert self._decide(b"x" * 3000, asks_at=later, last_ask_answered=False).ask

    def test_an_answered_shadow_ask_moves_the_next_start_past_its_span(self) -> None:
        # A shadow answer over [0, 600) never moves the extraction base.
        d = self._decide(b"x" * 900, asked_through=600, capped=False)
        assert (d.ask, d.start, d.end) == (True, 600, 900)
        # Fewer than min_new_chars past the answered span: no ask at all.
        assert self._decide(b"x" * 650, asked_through=600).reason == "below threshold"

    def test_the_ask_cursor_never_moves_the_start_backwards(self) -> None:
        d = self._decide(b"x" * 900, covered=700, asked_through=600)
        assert (d.ask, d.start) == (True, 700)

    def test_catch_up_needs_an_answer_that_moved_the_start(self) -> None:
        recent = [self.now - timedelta(seconds=60)]
        # An under-covered `on` answer left its span owed: same start, no waiver.
        stuck = self._decide(b"x" * 3000, asks_at=recent, last_ask_start=0)
        assert stuck.reason == "inside interval"
        moved = self._decide(b"x" * 3000, covered=500, asks_at=recent, last_ask_start=0)
        assert (moved.ask, moved.start) == (True, 500)
        shadow = self._decide(b"x" * 3000, asks_at=recent, last_ask_start=0, asked_through=1000)
        assert (shadow.ask, shadow.start) == (True, 1000)

    def test_cap_applies_only_when_capped(self) -> None:
        old = [self.now - timedelta(hours=2), self.now - timedelta(hours=1)]
        assert not self._decide(b"x" * 600, asks_at=old).ask
        assert self._decide(b"x" * 600, asks_at=old, capped=False).ask

    def test_a_long_span_is_cut_at_a_paragraph_break(self) -> None:
        text = (b"a" * 700) + b"\n\n" + (b"b" * 700)
        d = self._decide(text)
        assert d.end == 702

    def test_a_hard_cut_never_splits_a_utf8_sequence(self) -> None:
        text = "é".encode() * 800
        d = self._decide(text)
        assert d.end <= 1000
        text[d.start : d.end].decode("utf-8")


class TestPrompt:
    def test_ask_names_the_tool_the_id_and_fences_the_span(self) -> None:
        ask = agent.build_ask("ask-123", "## User\n\nhello", reference_date="2026-10-09")
        assert ask.startswith(agent.ASK_MARKER)
        assert "`extraction_submit`" in ask and '"ask-123"' in ask
        assert '"evidence"' in ask
        assert "Return ONLY a JSON array" not in ask
        assert '<session nonce="' in ask and "hello" in ask
        assert "{reference_date}" not in ask

    def test_ask_asks_for_every_claim_and_names_the_expected_count(self) -> None:
        span = "## User\n\n" + "x" * 9990
        ask = agent.build_ask("ask-123", span, reference_date="2026-10-09")
        assert "durable" not in ask.split("Rules:")[0]
        assert "not a selection of" in ask
        # 10,000 characters at the default 1.9 claims per 1000.
        assert "about 10 thousand characters" in ask and "about 19 claims" in ask
        assert "never a\nquota" in ask

    def test_expected_count_follows_config(self, monkeypatch: pytest.MonkeyPatch) -> None:
        cfg = agent.get_config().extraction
        monkeypatch.setattr(cfg, "agent_expected_claims_per_kchar", 3.0)
        assert agent.expected_claims(20_000) == 60
        assert agent.expected_claims(10) == 1

    async def test_the_proxy_is_given_the_same_guide(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen: dict[str, str] = {}

        async def fake(purpose: str, user: str, **kwargs: Any) -> tuple[str, str]:
            seen["system"] = kwargs["system"]
            return "[]", "anthropic:test"

        monkeypatch.setattr("particles.llm.complete_with_provider_model", fake)
        await agent.AgentExtractor().extract(
            None,  # type: ignore[arg-type]
            SPAN.encode(),
        )
        assert agent.thoroughness(SPAN) in seen["system"]

    def test_rules_carry_the_general_extractor_rules(self) -> None:
        rules = agent.agent_rules("2026-10-09")
        assert "One particle = one falsifiable assertion" in rules
        assert "evidence" in rules


class TestPlugins:
    def test_never_routed(self) -> None:
        assert any(e.EXTRACTOR_ID == agent.EXTRACTOR_ID for e in get_extractors())
        for source_type in ("CONVERSATION", "WEB_PAGE", "LOCAL_MARKDOWN", "JOURNAL"):
            assert select_extractor(source_type).EXTRACTOR_ID != agent.EXTRACTOR_ID

    @pytest.mark.asyncio
    async def test_submitted_candidates_get_the_session_state_filter(self) -> None:
        span = (
            "## Assistant\n\nYou are in the worktree .claude/worktrees/brave-otter-1a2b3c.\n\n"
            "The release ships on Friday.\n"
        )
        verified = agent.verify_submission(
            [
                _candidate(
                    "The session is in worktree .claude/worktrees/brave-otter-1a2b3c.",
                    "worktree .claude/worktrees/brave-otter-1a2b3c",
                ),
                _candidate("The release ships on Friday.", "The release ships on Friday."),
            ],
            span,
        )
        extractor = agent.SubmittedCandidatesExtractor(
            verified.accepted, provider_model="claude-code:claude-opus-5-5", read_through=None
        )
        result = await extractor.extract(None, span.encode(), source_type="CONVERSATION")  # type: ignore[arg-type]
        assert [c.content for c in result.candidates] == ["The release ships on Friday."]
        assert result.candidates[0].provider_model == "claude-code:claude-opus-5-5"


def test_purpose_servable_needs_a_key_only_for_anthropic(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert not agent.purpose_servable("extraction")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    assert agent.purpose_servable("extraction")
