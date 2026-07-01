"""Tests for the SQLite traceability store — req-id + history, NO raw PII."""

from __future__ import annotations

import re

import pytest

from oc14_triage.agent.state import DOSSIER_FIELDS
from oc14_triage.agent.store import Store

UUID4_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "trace.db")


def test_columns_match_dossier_fields_and_no_raw_text(store):
    cols = store._columns()
    assert set(cols) == set(DOSSIER_FIELDS)
    assert "raw_text" not in cols


def test_record_then_get_roundtrips_fields(store):
    case = {
        "interaction_id": "int-1",
        "session_id": "sess-1",
        "symptoms_anon": "douleur thoracique",
        "urgency": "urgence maximale",
        "justification": "red flag",
        "input_sha256": "deadbeef",
    }
    iid = store.record(case)
    assert iid == "int-1"

    row = store.get("int-1")
    assert row is not None
    assert row["interaction_id"] == "int-1"
    assert row["session_id"] == "sess-1"
    assert row["symptoms_anon"] == "douleur thoracique"
    assert row["urgency"] == "urgence maximale"
    assert row["justification"] == "red flag"
    assert row["input_sha256"] == "deadbeef"


def test_record_without_interaction_id_autogenerates_uuid(store):
    iid = store.record({"session_id": "sess-x", "symptoms_anon": "toux"})
    assert UUID4_RE.match(iid)
    assert store.get(iid)["interaction_id"] == iid


def test_record_fills_timestamp_when_absent(store):
    iid = store.record({"session_id": "sess-t"})
    row = store.get(iid)
    assert row["timestamp_utc"]  # non-empty, auto-filled


def test_get_unknown_returns_none(store):
    assert store.get("nope") is None


def test_history_orders_by_timestamp_and_excludes_soft_deleted(store):
    store.record(
        {"interaction_id": "a", "session_id": "s1", "timestamp_utc": "2026-01-01T00:00:00"}
    )
    store.record(
        {"interaction_id": "b", "session_id": "s1", "timestamp_utc": "2026-01-03T00:00:00"}
    )
    store.record(
        {"interaction_id": "c", "session_id": "s1", "timestamp_utc": "2026-01-02T00:00:00"}
    )
    store.record(
        {"interaction_id": "other", "session_id": "s2", "timestamp_utc": "2026-01-01T00:00:00"}
    )

    store.soft_delete("b")

    rows = store.history("s1")
    assert [r["interaction_id"] for r in rows] == ["a", "c"]  # b excluded, ordered by ts


def test_soft_delete_hides_from_history_but_row_remains(store):
    store.record({"interaction_id": "d", "session_id": "s3"})
    store.soft_delete("d")
    assert store.history("s3") == []
    # row still physically present (audit trail), flagged deleted
    assert store.get("d")["deleted"] == 1


def test_all_sessions_returns_distinct_session_ids(store):
    store.record({"interaction_id": "1", "session_id": "s1"})
    store.record({"interaction_id": "2", "session_id": "s1"})
    store.record({"interaction_id": "3", "session_id": "s2"})
    assert set(store.all_sessions()) == {"s1", "s2"}
