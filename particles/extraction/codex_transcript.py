# SPDX-FileCopyrightText: 2026 The Particles authors
# SPDX-License-Identifier: Apache-2.0

"""Store-free Codex rollout contract; unknown complete envelopes fail closed."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal
from uuid import UUID

from particles.extraction.day_marks import escape_mark_lines, format_turn_heading

CONTRACT = "codex-conversation/1"


class UnsupportedTranscript(ValueError):
    """A redacted reason code, never the offending source text."""


def session_uuid(value: object) -> str:
    """Validate canonical UUID spelling before it can name a path or URI."""
    if not isinstance(value, str):
        raise UnsupportedTranscript("invalid_session_identity")
    try:
        parsed = str(UUID(value))
    except ValueError:
        raise UnsupportedTranscript("invalid_session_identity") from None
    if value != parsed:
        raise UnsupportedTranscript("noncanonical_session_identity")
    return parsed


def _time(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return result.astimezone(UTC) if result.tzinfo else None
    except ValueError:
        return None


def _object(value: object) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(k, str) for k in value):
        raise UnsupportedTranscript("invalid_object")
    return value


@dataclass(frozen=True)
class Ancestry:
    """A measured paginated fork's exact inherited boundary."""

    session_id: str
    end_ordinal: int
    end_byte_offset: int


@dataclass(frozen=True)
class Turn:
    """An accepted human or assistant text turn, not yet redacted."""

    message_id: str
    speaker: Literal["User", "Assistant"]
    text: str
    timestamp: datetime | None


@dataclass(frozen=True)
class Transcript:
    """Validated local session identity and its own testimony (child delta only)."""

    session_id: str
    cwd: str
    started_at: datetime
    turns: tuple[Turn, ...]
    ancestry: Ancestry | None

    def render(
        self, redact: Callable[[str], str], *, body_lengths: list[int] | None = None
    ) -> tuple[str, list[dict[str, str]]]:
        """Redact before rendering and hashing, optionally recording rendered body lengths."""
        day = self.started_at.date()
        rendered: list[str] = []
        identities: list[dict[str, str]] = []
        for turn in self.turns:
            current = max(day, turn.timestamp.date()) if turn.timestamp else day
            body = redact(turn.text)
            heading = format_turn_heading(turn.speaker, current if current != day else None)
            escaped = escape_mark_lines(body)
            rendered.append(f"{heading}\n\n{escaped}\n\n")
            if body_lengths is not None:
                body_lengths.append(len(escaped))
            identities.append(
                {"id": turn.message_id, "body_hash": hashlib.sha256(body.encode()).hexdigest()}
            )
            day = current
        return "".join(rendered), identities


def _ancestry(meta: dict[str, Any]) -> Ancestry | None:
    base = meta.get("history_base")
    if base is None:
        if meta.get("forked_from_id") is not None:
            raise UnsupportedTranscript("unsupported_legacy_fork")
        return None
    base = _object(base)
    parent = session_uuid(base.get("thread_id"))
    ordinal, offset = base.get("end_ordinal_exclusive"), base.get("end_byte_offset")
    if type(ordinal) is not int or type(offset) is not int or ordinal < 0 or offset < 0:
        raise UnsupportedTranscript("invalid_fork_boundary")
    if meta.get("forked_from_id") not in (None, parent):
        raise UnsupportedTranscript("conflicting_fork_identity")
    return Ancestry(parent, ordinal, offset)


def _bookkeeping(kind: str, payload: dict[str, Any]) -> bool:
    if kind == "event_msg":
        event = payload.get("type")
        expected: dict[str, tuple[str, type[object]]] = {
            "agent_message": ("message", str),
            "user_message": ("message", str),
            "task_started": ("turn_id", str),
            "task_complete": ("turn_id", str),
            "turn_aborted": ("reason", str),
            "thread_settings_applied": ("thread_settings", dict),
            "item_completed": ("item", dict),
        }
        if event == "token_count" and isinstance(payload.get("info"), (dict, type(None))):
            return True
        if isinstance(event, str) and event in expected:
            field, shape = expected[event]
            if isinstance(payload.get(field), shape):
                return True
        raise UnsupportedTranscript("unknown_event_envelope")
    if kind == "world_state":
        if type(payload.get("full")) is bool and isinstance(payload.get("state"), dict):
            return True
        raise UnsupportedTranscript("unknown_world_state")
    if kind == "turn_context":
        if isinstance(payload.get("cwd"), str) and isinstance(payload.get("model"), str):
            return True
        raise UnsupportedTranscript("unknown_turn_context")
    if kind == "token_usage_record":
        if isinstance(payload.get("usage"), dict) and isinstance(payload.get("session_id"), str):
            return True
        raise UnsupportedTranscript("unknown_token_usage")
    if kind == "compacted":
        if (
            isinstance(payload.get("message"), str)
            and isinstance(payload.get("replacement_history"), list)
            and all(isinstance(item, dict) for item in payload["replacement_history"])
        ):
            return True
        raise UnsupportedTranscript("unknown_compaction_envelope")
    return False


def _turn(payload: dict[str, Any], stamp: object, emitted: frozenset[str]) -> Turn | None:
    variant = payload.get("type")
    if not isinstance(variant, str):
        raise UnsupportedTranscript("unknown_response_item")
    if variant in {"function_call", "custom_tool_call"}:
        field = "arguments" if variant == "function_call" else "input"
        if all(isinstance(payload.get(key), str) for key in ("call_id", "name", field)):
            return None
        raise UnsupportedTranscript("unknown_tool_call")
    if variant in {"function_call_output", "custom_tool_call_output"}:
        if isinstance(payload.get("call_id"), str) and isinstance(
            payload.get("output"), (str, list)
        ):
            return None
        raise UnsupportedTranscript("unknown_tool_output")
    if variant == "reasoning":
        if isinstance(payload.get("summary"), list):
            return None
        raise UnsupportedTranscript("unknown_reasoning_envelope")
    if variant != "message":
        raise UnsupportedTranscript("unknown_response_item")
    role, content, message_id = payload.get("role"), payload.get("content"), payload.get("id")
    if (
        not isinstance(role, str)
        or role not in {"user", "assistant", "developer", "system"}
        or not isinstance(content, list)
    ):
        raise UnsupportedTranscript("unknown_message_envelope")
    parts: list[str] = []
    for item in content:
        block = _object(item)
        if block.get("type") not in {"input_text", "output_text"} or not isinstance(
            block.get("text"), str
        ):
            raise UnsupportedTranscript("unknown_content_block")
        parts.append(block["text"])
    if role in {"system", "developer"}:
        return None
    if role == "user":
        metadata = _object(payload.get("internal_chat_message_metadata_passthrough"))
        labels, kinds = metadata.get("content_item_metadata"), metadata.get("content_item_kinds")
        if (
            not isinstance(labels, list)
            or not isinstance(kinds, list)
            or len(labels) != len(parts)
            or len(kinds) != len(parts)
            or not parts
        ):
            raise UnsupportedTranscript("unlabelled_user_envelope")
        origins = [_object(_object(item).get("provenance")).get("type") for item in labels]
        if all(origin == "harness" for origin in origins) and all(
            isinstance(k, str) for k in kinds
        ):
            return None
        if any(
            origin != "user" or kind != "user.text"
            for origin, kind in zip(origins, kinds, strict=True)
        ):
            raise UnsupportedTranscript("mixed_or_unknown_user_provenance")
        if any(hashlib.sha256(part.encode()).hexdigest() in emitted for part in parts):
            raise UnsupportedTranscript("ambiguous_echo")
    if not isinstance(message_id, str) or not message_id:
        raise UnsupportedTranscript("missing_message_id")
    return Turn(
        message_id, "User" if role == "user" else "Assistant", "\n".join(parts), _time(stamp)
    )


def parse_transcript(text: str, *, emitted_hashes: frozenset[str] = frozenset()) -> Transcript:
    """Validate every complete envelope; incomplete final JSON is retried later."""
    meta: dict[str, Any] | None = None
    started: datetime | None = None
    turns: list[Turn] = []
    seen: dict[str, Turn] = {}
    lines = text.splitlines(keepends=True)
    for index, line in enumerate(lines):
        try:
            record = _object(json.loads(line))
        except json.JSONDecodeError:
            if index == len(lines) - 1 and not line.endswith("\n"):
                break
            raise UnsupportedTranscript("malformed_complete_record") from None
        if "ordinal" in record and (type(record["ordinal"]) is not int or record["ordinal"] < 0):
            raise UnsupportedTranscript("invalid_record_ordinal")
        kind, payload = record.get("type"), _object(record.get("payload"))
        if index == 0 and kind != "session_meta":
            raise UnsupportedTranscript("missing_initial_metadata")
        if kind == "session_meta":
            if meta is not None:
                raise UnsupportedTranscript("repeated_session_metadata")
            session_id = session_uuid(payload.get("session_id", payload.get("id")))
            if payload.get("id") not in (None, session_id):
                raise UnsupportedTranscript("conflicting_session_identity")
            if not isinstance(payload.get("cwd"), str) or not payload["cwd"].startswith("/"):
                raise UnsupportedTranscript("invalid_launch_cwd")
            started = _time(payload.get("timestamp", record.get("timestamp")))
            if started is None:
                raise UnsupportedTranscript("missing_start_timestamp")
            meta = payload
        elif isinstance(kind, str) and _bookkeeping(kind, payload):
            continue
        elif kind == "response_item":
            turn = _turn(payload, record.get("timestamp"), emitted_hashes)
            if turn is None:
                continue
            old = seen.get(turn.message_id)
            if old is not None and (old.speaker, old.text) != (turn.speaker, turn.text):
                raise UnsupportedTranscript("conflicting_message_id")
            if old is None:
                turns.append(turn)
                seen[turn.message_id] = turn
        else:
            raise UnsupportedTranscript("unknown_record_type")
    if meta is None or started is None:
        raise UnsupportedTranscript("missing_session_metadata")
    return Transcript(
        session_uuid(meta.get("session_id", meta.get("id"))),
        meta["cwd"],
        started,
        tuple(turns),
        _ancestry(meta),
    )
