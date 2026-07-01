"""End-to-end tests for the LangGraph triage chain. Backend is stubbed (no model needed).
The chain nodes ARE the mentor's patient-journey: anonymisation → prétraitement →
triage → explication → persistance → SIH."""
import json
import os

import pytest

pytest.importorskip("langgraph")
os.environ["OC14_TRIAGE_STUB"] = "1"

from oc14_triage.agent import graph as gmod  # noqa: E402
from oc14_triage.agent.store import Store  # noqa: E402

_REDFLAG = "Jean Dupont, né le 3/2/1980, tél 06.12.34.56.78 — douleur thoracique aiguë et sueurs"


def test_redflag_case_end_to_end(tmp_path):
    final = gmod.process_case(_REDFLAG, lang="fr", session_id="s1", store=Store(tmp_path / "d.db"))
    assert final["urgency"] == "urgence maximale"          # red-flag → highest
    assert "Jean Dupont" not in final["anon_text"]          # PII anonymised
    assert "douleur thoracique" in final["anon_text"]       # clinical signal kept
    assert final["input_sha256"]                            # hash-for-traceability
    assert final["disclaimer_present"] is True
    assert final.get("raw_text", "") == ""                  # raw input cleared from state
    assert final["sih_record"]["resourceType"] == "Bundle"  # FHIR record built


def test_persistence_stores_no_pii(tmp_path):
    store = Store(tmp_path / "d.db")
    gmod.process_case("Marie Curie 07.11.22.33.44 mal de gorge léger depuis hier",
                      lang="fr", session_id="s2", store=store)
    hist = store.history("s2")
    assert len(hist) == 1
    assert "Marie Curie" not in json.dumps(hist[0], ensure_ascii=False)  # no PII persisted
    assert hist[0]["urgency"] in ("urgence maximale", "urgence modérée", "urgence différée")


def test_safety_override_escalates_undertriaged_redflag(tmp_path, monkeypatch):
    # Model under-triages a red-flag → the chain must escalate to maximale.
    monkeypatch.setattr(gmod, "triage_once",
                        lambda *a, **k: "1. Niveau d'urgence : urgence différée. "
                        "2. Justification : x. 3. Recommandation : y. "
                        "Cet avis ne remplace pas une consultation médicale.")
    final = gmod.process_case("douleur thoracique constrictive", lang="fr",
                              session_id="s3", store=Store(tmp_path / "d.db"))
    assert final["urgency"] == "urgence maximale"


def test_unstructured_model_output_defaults_to_moderee(tmp_path, monkeypatch):
    # Model returns garbage with no parseable urgency and no red-flag input →
    # the chain must NOT crash and must default the urgency to modérée.
    monkeypatch.setattr(gmod, "triage_once", lambda *a, **k: "garbage no structure")
    final = gmod.process_case("mal de dos depuis une semaine", lang="fr",
                              session_id="s5", store=Store(tmp_path / "d.db"))
    assert final["urgency"] == "urgence modérée"
    assert final["sih_record"]["resourceType"] == "Bundle"  # no KeyError(None) crash


def test_unsupported_lang_is_coerced_and_triages(tmp_path):
    # An unsupported lang (e.g. "es") must be coerced to "fr" rather than 500 / KeyError.
    final = gmod.process_case("dolor de cabeza leve", lang="es", session_id="s6",
                              store=Store(tmp_path / "d.db"))
    assert final["urgency"] in ("urgence maximale", "urgence modérée", "urgence différée")


def test_trace_records_node_timing(tmp_path):
    final = gmod.process_case("angine légère depuis 2 jours", lang="fr",
                              session_id="s4", store=Store(tmp_path / "d.db"))
    nodes = [t["node"] for t in final["trace"]]
    assert {"anonymisation", "pretraitement", "triage", "explication", "persistance", "sih"} <= set(nodes)
    assert all(isinstance(t["ms"], (int, float)) for t in final["trace"])
