"""Redacted, bounded JSONL telemetry: privacy invariants and crash-proof appends."""

import json
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from jev_fastpath.telemetry import TelemetryWriter, redact
from jev_fastpath.types import TelemetryEvent

CANARIES = (
    "password: hunter2",
    "api_key=sk-live-abcdefghij",
    "token=ts_SECRET_VALUE",
    "Bearer Abcdef1234567890",
    "AKIAIOSFODNN7EXAMPLE",
    "user@example.com",
)


def _event(**overrides):
    values = dict(
        outcome="no_candidate",
        reason="local_filter",
        text="quanto é 2 + 2?",
        candidates=(),
    )
    values.update(overrides)
    return TelemetryEvent(**values)


def _context(**overrides):
    values = dict(
        mode="active",
        session_id="session-abc-123",
        turn_id="turn-xyz-789",
        platform="telegram",
        provider="nous",
        model="hermes-4-405b",
        api_mode="chat_completions",
    )
    values.update(overrides)
    return values


@pytest.fixture
def writer(tmp_path):
    return TelemetryWriter(tmp_path)


@pytest.fixture
def records(tmp_path):
    writer = TelemetryWriter(tmp_path)

    def _records():
        path = tmp_path / "artifacts" / "jev_fastpath" / "decisions.jsonl"
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    return writer, _records


def test_writes_one_json_object_per_line(records):
    writer, read = records
    writer.write(_event(), _context())
    writer.write(_event(outcome="fallback", reason="JevError", text="x"), _context())
    rows = read()
    assert len(rows) == 2
    assert rows[0]["outcome"] == "no_candidate"
    assert rows[1]["outcome"] == "fallback"


def test_record_contains_fixed_field_set(records):
    writer, read = records
    writer.write(_event(), _context())
    row = read()[0]
    expected = {
        "ts", "mode", "session_hash", "turn_hash", "platform", "provider", "model",
        "api_mode", "text_hash", "text_preview", "candidates", "selected_handler",
        "confidence", "short_circuit_probability", "latency_ms", "outcome", "reason",
        "provider_call_avoided", "usage",
    }
    assert set(row) == expected


def test_session_and_turn_are_hashed_not_raw(records):
    writer, read = records
    writer.write(_event(), _context())
    row = read()[0]
    assert "session-abc-123" not in json.dumps(row)
    assert "turn-xyz-789" not in json.dumps(row)
    assert len(row["session_hash"]) == 16
    assert len(row["turn_hash"]) == 16


def test_secret_canaries_never_appear(records):
    writer, read = records
    for canary in CANARIES:
        writer.write(_event(text=f"please {canary} now"), _context())
    serialized = json.dumps(read())
    assert "hunter2" not in serialized
    assert "sk-live-abcdefghij" not in serialized
    assert "ts_SECRET_VALUE" not in serialized
    assert "Abcdef1234567890" not in serialized
    assert "AKIAIOSFODNN7EXAMPLE" not in serialized
    assert "user@example.com" not in serialized


def test_preview_and_reason_are_bounded(records):
    writer, read = records
    writer.write(_event(text="x" * 5000, reason="y" * 500), _context())
    row = read()[0]
    assert len(row["text_preview"]) <= 160
    assert len(row["reason"]) <= 160


def test_usage_is_filtered_to_known_numeric_fields(records):
    writer, read = records
    writer.write(_event(usage={
        "cost": 0.001, "input_tokens": 10, "output_tokens": 5,
        "raw_payload": {"nested": "drop me"}, "model": "nope", "nan_field": float("nan"),
    }), _context())
    row = read()[0]
    assert row["usage"] == {"cost": 0.001, "input_tokens": 10, "output_tokens": 5}


def test_provider_call_avoided_only_for_short_circuit(records):
    writer, read = records
    writer.write(_event(outcome="no_candidate"), _context())
    writer.write(_event(outcome="short_circuit", selected_handler="calculator"), _context())
    rows = read()
    assert rows[0]["provider_call_avoided"] is False
    assert rows[1]["provider_call_avoided"] is True


def test_concurrent_appends_produce_valid_lines(tmp_path):
    writer = TelemetryWriter(tmp_path)
    lines_lock = threading.Lock()

    def _write(turn):
        writer.write(_event(text=f"turn {turn}"), _context(turn_id=f"t{turn}"))
        return True

    with ThreadPoolExecutor(max_workers=8) as pool:
        assert all(pool.map(_write, range(64)))
    path = tmp_path / "artifacts" / "jev_fastpath" / "decisions.jsonl"
    content = path.read_text(encoding="utf-8")
    assert content.count("\n") == 64
    for line in content.splitlines():
        assert json.loads(line)


def test_io_errors_are_swallowed(tmp_path, caplog):
    blocker = tmp_path / "artifacts"
    blocker.mkdir()
    (blocker / "jev_fastpath").write_text("not a directory", encoding="utf-8")
    writer = TelemetryWriter(tmp_path)
    writer.write(_event(), _context())


def test_missing_context_fields_render_empty(records):
    writer, read = records
    writer.write(_event(), {})
    row = read()[0]
    assert row["mode"] == ""
    assert row["session_hash"] == "" or row["session_hash"]


def test_redact_handles_non_string():
    assert redact(None) == ""
    assert redact(42) == ""
