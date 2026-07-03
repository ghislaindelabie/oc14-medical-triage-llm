"""Tests for the deterministic input-sanity guardrail (built strict TDD).

`is_intelligible` is the deterministic half of the hybrid LLM+rules design: it catches
CLEAR gibberish / non-linguistic input BEFORE the model triage so the chain refuses with
an honest "entrée inintelligible" notice instead of confabulating a confident verdict.
It is CONSERVATIVE by design — it errs toward letting input through to the model / HITL,
and must never reject short but real symptoms.
"""

from __future__ import annotations

import json
from pathlib import Path

from oc14_triage.agent.validation import intelligibility_reason, is_intelligible

# --- the observed real bug: gibberish must be flagged --------------------------------------

def test_observed_gibberish_bug_is_not_intelligible():
    # The exact input that produced a hallucinated "urgence maximale" verdict.
    assert is_intelligible("ddsd dsfdsx dfd dsfd", "fr") is False


def test_gibberish_variants_are_not_intelligible():
    for junk in ("xkcd zzzz qqq fff", "aaaa", "zzzzzzzzzzzz", "b c d f g h", "!!! ??? ..."):
        assert is_intelligible(junk, "fr") is False, junk


def test_gibberish_reason_is_explainable():
    reason = intelligibility_reason("ddsd dsfdsx dfd dsfd", "fr")
    assert reason is not None
    assert isinstance(reason, str) and reason.strip()


# --- conservatism: short but REAL symptoms must pass ----------------------------------------

def test_terse_real_symptoms_are_intelligible():
    for symptom in ("fièvre", "toux", "mal au ventre", "fièvre depuis 2 jours",
                    "maux de tête", "vomissements", "j'ai mal au dos"):
        assert is_intelligible(symptom, "fr") is True, symptom
        assert intelligibility_reason(symptom, "fr") is None, symptom


def test_terse_english_symptoms_are_intelligible():
    for symptom in ("fever", "cough", "chest pain", "headache since yesterday"):
        assert is_intelligible(symptom, "en") is True, symptom


def test_vitals_shorthand_is_intelligible():
    # Clinical shorthand mixes letters/digits/symbols but is meaningful input, not gibberish.
    assert is_intelligible("SpO2 88%, FC 120, TA 90/60", "fr") is True


# --- red-flag phrase amid noise still reads as intelligible (safety path stays open) --------

def test_redflag_phrase_amid_noise_is_intelligible():
    # Even with junk tokens, a real clinical phrase must not be discarded as gibberish —
    # the safety override downstream depends on this reaching the model / rules.
    assert is_intelligible("ddsd douleur thoracique zzz", "fr") is True


# --- empty / whitespace: not intelligible, but that's the existing "empty input" path -------

def test_empty_and_whitespace_are_not_intelligible():
    assert is_intelligible("", "fr") is False
    assert is_intelligible("   \n\t ", "fr") is False


# --- calibration against the real gold eval set: every real clinical case must pass ---------

def _gold_cases() -> list[str]:
    for rel in ("data/kaggle_upload_v10/triage_eval_gold.jsonl",
                "data/kaggle_upload/triage_eval_gold.jsonl"):
        p = Path(__file__).resolve().parents[1] / rel
        if p.exists():
            return [json.loads(line)["user"] for line in p.read_text(encoding="utf-8").splitlines()
                    if line.strip()]
    return []


def test_all_gold_clinical_cases_are_intelligible():
    cases = _gold_cases()
    assert cases, "gold eval set not found — calibration cannot run"
    for text in cases:
        assert is_intelligible(text, "fr") is True, text[:80]
