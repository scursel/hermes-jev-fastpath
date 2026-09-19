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
        "api_mode", "text_hash", "text_length", "text_preview", "candidates",
        "selected_handler", "confidence", "short_circuit_probability", "latency_ms",
        "outcome", "reason", "provider_call_avoided", "usage",
        "answer_hash", "answer_preview",
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


def test_no_candidate_rows_store_hash_and_length_but_no_preview(records):
    writer, read = records
    text = "quanto é 2 + 2?"
    writer.write(_event(outcome="no_candidate", text=text), _context())
    row = read()[0]
    assert row["text_preview"] == ""
    assert row["text_length"] == len(text)
    assert len(row["text_hash"]) == 16


def test_candidate_rows_store_bounded_preview(records):
    writer, read = records
    writer.write(_event(outcome="normal_llm", text="y" * 5000), _context())
    row = read()[0]
    assert 0 < len(row["text_preview"]) <= 160
    assert row["text_length"] == 5000


def test_shadow_and_active_rows_carry_answer_evidence(records):
    writer, read = records
    writer.write(_event(outcome="would_short_circuit", text="2 + 2", answer="2 + 2 = 4"), _context())
    writer.write(_event(outcome="short_circuit", text="2 + 2", answer="2 + 2 = 4"), _context())
    writer.write(_event(outcome="no_candidate", text="hello"), _context())
    rows = read()
    assert rows[0]["answer_preview"] == "2 + 2 = 4"
    assert len(rows[0]["answer_hash"]) == 16
    assert rows[1]["answer_preview"] == "2 + 2 = 4"
    assert rows[2]["answer_preview"] == ""
    assert rows[2]["answer_hash"] == ""


def test_rotation_keeps_a_single_bounded_backup(tmp_path):
    writer = TelemetryWriter(tmp_path, max_bytes=2000)
    for index in range(40):
        writer.write(
            _event(outcome="normal_llm", text=f"turn {index} " + "z" * 200), _context()
        )
    current = tmp_path / "artifacts" / "jev_fastpath" / "decisions.jsonl"
    backup = tmp_path / "artifacts" / "jev_fastpath" / "decisions.jsonl.1"
    assert backup.exists()
    assert current.stat().st_size <= 2000 + 400
    lines = current.read_text(encoding="utf-8").splitlines()
    assert all(json.loads(line) for line in lines)


def test_log_file_has_restrictive_permissions(tmp_path):
    writer = TelemetryWriter(tmp_path)
    writer.write(_event(), _context())
    path = tmp_path / "artifacts" / "jev_fastpath" / "decisions.jsonl"
    assert (path.stat().st_mode & 0o777) == 0o600


def test_credential_pattern_coverage(records):
    writer, read = records
    samples = (
        "stripe key sk_live_abcdefghijklmnopqrst",
        "pk_live_abcdefghijklmnopqrst",
        "github token ghp_abcdefghijklmnopqrstuvwx",
        "github_pat_11AAAAAAA0abcdefghijklmnopqrstuv",
        "key AIzaSyA-1234567890abcdefghijklmnopqrstu",
        "-----BEGIN PRIVATE KEY-----",
        "aws_secret_access_key = wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
        "the password is hunter2 with spaces",
    )
    for sample in samples:
        writer.write(_event(outcome="normal_llm", text=sample), _context())
    serialized = json.dumps(read())
    for secret in (
        "sk_live_abcdefghijklmnopqrst", "pk_live_abcdefghijklmnopqrst",
        "ghp_abcdefghijklmnopqrstuvwx", "github_pat_11AAAAAAA0",
        "AIzaSyA-1234567890", "wJalrXUtnFEMI", "hunter2",
        "BEGIN PRIVATE KEY",
    ):
        assert secret not in serialized, secret


def test_redaction_is_bounded_for_megabyte_input():
    import time

    text = "x" * (1024 * 1024) + " password: hunter2"
    started = time.monotonic()
    out = redact(text)
    elapsed = time.monotonic() - started
    assert elapsed < 2.0  # bounded, CI-safe
    assert "hunter2" not in out[:200]


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


def test_missing_context_fields_hash_the_empty_string(records):
    writer, read = records
    writer.write(_event(), {})
    row = read()[0]
    import hashlib

    empty = hashlib.sha256(b"").hexdigest()[:16]
    assert row["session_hash"] == empty
    assert row["turn_hash"] == empty
    assert row["mode"] == ""


def test_redact_handles_non_string():
    assert redact(None) == ""
    assert redact(42) == ""
