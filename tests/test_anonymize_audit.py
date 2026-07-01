"""The dataset-level Presidio audit — the RGPD verification deliverable. We test the
aggregation on a small sample (real Presidio); the full-corpus run is a CLI on top."""
import pytest

pytest.importorskip("presidio_analyzer")

from oc14_triage.data.anonymize_audit import audit_records


def test_audit_counts_pii_and_hashes_each_record():
    recs = [
        {"text": "Monsieur Jean Dupont, tél 06.12.34.56.78, se plaint d'une douleur thoracique",
         "lang": "fr"},
        {"text": "Le patient présente une toux sèche depuis trois jours", "lang": "fr"},
    ]
    a = audit_records(recs)
    assert a["n_records"] == 2
    assert a["entities_by_type"].get("PERSON", 0) >= 1        # Jean Dupont
    assert a["entities_by_type"].get("PHONE_NUMBER", 0) >= 1  # the phone
    assert a["n_records_with_pii"] >= 1
    assert a["engine"] and a["engine_version"]                # recorded for the audit trail
    assert len(a["per_record"]) == 2
    assert all(len(r["sha256"]) == 64 for r in a["per_record"])  # one-way hash, no raw text
    assert all("text" not in r for r in a["per_record"])         # raw text never stored in audit


def test_audit_empty_is_zero():
    a = audit_records([])
    assert a["n_records"] == 0 and a["n_records_with_pii"] == 0
