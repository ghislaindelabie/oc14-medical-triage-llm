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


# --- Derived runtime confidence (option 2: from rule/model agreement + parse success) -------

def test_confidence_high_when_redflag_and_model_agree(tmp_path):
    # Red-flag input → stub returns maximale → rule & model AGREE → high confidence, no review.
    final = gmod.process_case("douleur thoracique aiguë et sueurs", lang="fr",
                              session_id="c1", store=Store(tmp_path / "d.db"))
    assert final["urgency"] == "urgence maximale"
    assert final["confidence"] == "high"
    assert final["needs_review"] is False


def test_confidence_low_when_override_corrects_undertriage(tmp_path, monkeypatch):
    # Model under-triages a red-flag → safety override escalates, but rule/model DISAGREE →
    # low confidence, flag for clinician review.
    monkeypatch.setattr(gmod, "triage_once",
                        lambda *a, **k: "1. Niveau d'urgence : urgence différée. "
                        "2. Justification : x. 3. Recommandation : y. "
                        "Cet avis ne remplace pas une consultation médicale.")
    final = gmod.process_case("douleur thoracique constrictive", lang="fr",
                              session_id="c2", store=Store(tmp_path / "d.db"))
    assert final["urgency"] == "urgence maximale"
    assert final["confidence"] == "low"
    assert final["needs_review"] is True


def test_confidence_low_when_output_unstructured(tmp_path, monkeypatch):
    monkeypatch.setattr(gmod, "triage_once", lambda *a, **k: "garbage no structure")
    final = gmod.process_case("mal de dos depuis une semaine", lang="fr",
                              session_id="c3", store=Store(tmp_path / "d.db"))
    assert final["confidence"] == "low"
    assert final["needs_review"] is True


def test_confidence_medium_on_clean_case_without_redflag(tmp_path):
    final = gmod.process_case("mal de gorge modéré depuis hier", lang="fr",
                              session_id="c4", store=Store(tmp_path / "d.db"))
    assert final["confidence"] == "medium"
    assert final["needs_review"] is False


def test_model_unavailable_no_redflag_defers_to_clinician(tmp_path, monkeypatch):
    # Model down / cold-starting + no red-flag → NO fabricated verdict: defer to a clinician + retry.
    monkeypatch.setattr(gmod, "triage_once", lambda *a, **k: gmod._UNAVAILABLE)
    final = gmod.process_case("mal de dos depuis une semaine", lang="fr",
                              session_id="u1", store=Store(tmp_path / "d.db"))
    assert final["urgency"] is None                       # no fabricated urgency level
    assert final["needs_review"] is True
    assert "clinicien" in (final["recommendation"] or "").lower()
    j = (final["justification"] or "").lower()
    assert "indispon" in j or "démarre" in j
    assert final["sih_record"]["resourceType"] == "Bundle"  # None urgency must not crash FHIR


def test_model_unavailable_with_redflag_still_escalates(tmp_path, monkeypatch):
    # Safety override wins even during an outage: a detected red-flag still escalates to maximale.
    monkeypatch.setattr(gmod, "triage_once", lambda *a, **k: gmod._UNAVAILABLE)
    final = gmod.process_case("douleur thoracique aiguë et sueurs", lang="fr",
                              session_id="u2", store=Store(tmp_path / "d.db"))
    assert final["urgency"] == "urgence maximale"
    assert final["needs_review"] is True


def test_dossier_records_confidence_and_anonymised_vitals(tmp_path):
    store = Store(tmp_path / "d.db")
    gmod.process_case("mal de tête modéré", lang="fr", session_id="c5", store=store,
                      answers={"motif": "mal de tête modéré", "vitals": "T° 37, SpO2 98%"})
    row = store.history("c5")[0]
    assert row["confidence_level"] in ("high", "medium", "low")
    assert "SpO2 98%" in row["constantes"]   # vitals persisted (no PII to strip)


# --- Unintelligible-input guardrail (deterministic OOD guard) -------------------------------

def test_gibberish_input_refuses_without_a_verdict(tmp_path, monkeypatch):
    # The observed bug: gibberish must NOT yield a confident (over-triaged) verdict. The
    # deterministic guard flags it BEFORE the model, and the model output is discarded.
    monkeypatch.setattr(gmod, "triage_once",
                        lambda *a, **k: "1. Niveau d'urgence : urgence maximale. "
                        "2. Justification : détresse respiratoire. 3. Recommandation : SAMU. "
                        "Cet avis ne remplace pas une consultation médicale.")
    final = gmod.process_case("ddsd dsfdsx dfd dsfd", lang="fr",
                              session_id="g1", store=Store(tmp_path / "d.db"))
    assert final["urgency"] is None                         # no fabricated urgency
    assert final["needs_review"] is True
    assert final["valid"] is False                          # flagged in prétraitement
    j = (final["justification"] or "").lower()
    assert "intelligible" in j or "insuffisant" in j        # honest reason surfaced
    assert "détresse respiratoire" not in j                 # model hallucination NOT used
    assert final["sih_record"]["resourceType"] == "Bundle"  # None urgency must not crash FHIR


def test_terse_real_symptom_passes_the_guard(tmp_path):
    # A short but REAL symptom must go through the normal triage path (guard is conservative).
    final = gmod.process_case("fièvre depuis 2 jours", lang="fr",
                              session_id="g2", store=Store(tmp_path / "d.db"))
    assert final["valid"] is True
    assert final["urgency"] in ("urgence maximale", "urgence modérée", "urgence différée")


def test_redflag_amid_gibberish_still_escalates(tmp_path, monkeypatch):
    # Safety override wins over the unintelligible guard: a red-flag buried in noise must
    # still escalate to maximale rather than be refused as gibberish.
    monkeypatch.setattr(gmod, "triage_once", lambda *a, **k: gmod._UNAVAILABLE)
    final = gmod.process_case("ddsd douleur thoracique zzz", lang="fr",
                              session_id="g3", store=Store(tmp_path / "d.db"))
    assert final["urgency"] == "urgence maximale"
    assert final["needs_review"] is True
