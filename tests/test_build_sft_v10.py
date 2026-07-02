"""Unit tests for the SFT v10 builder's core row-construction logic.

v10 is the v9 recipe with two improvements: (1) each triage row carries the REAL per-case
consensus justification (a rater whose urgency == the case's consensus urgency), not a canned
sentence; (2) the weak classes are rebalanced. These tests exercise the pure helpers on a tiny
synthetic labeled sample so they run without the real dataset (CI-safe).
"""
from __future__ import annotations

from oc14_triage.data.templates import DISCLAIMER
from scripts.build_sft_v10 import (
    CANNED_FALLBACK_FR,
    build_triage_pool,
    pick_justification,
    triage_row_from_case,
)

# One synthetic labeled record: consensus 'urgence différée', two consistent raters agree,
# one dissenting rater. The matching-rater justification is the distinctive real sentence.
REAL_JUSTIF = "Prurit chronique sans signe d'alerte ni détresse, présentation dermatologique mineure."
_CASE = {
    "case_id": "syn-001",
    "text": "Un homme de 18 ans consulte pour des lésions cutanées prurigineuses depuis un mois.",
    "urgency": "urgence différée",
    "esi": 5,
    "unanimous": False,
    "is_gold": False,
    "flagged": True,
    "labels": [
        {"model": "openai", "is_triage_case": True, "urgency": "urgence différée", "esi": 5,
         "consistent": True, "justification": REAL_JUSTIF, "red_flags": [], "error": None},
        {"model": "mistral", "is_triage_case": True, "urgency": "urgence différée", "esi": 5,
         "consistent": True, "justification": "lésions cutanées chroniques stables", "red_flags": [],
         "error": None},
        {"model": "anthropic", "is_triage_case": False, "urgency": None, "esi": None,
         "consistent": False, "justification": "Cas pédagogique, pas un triage.", "red_flags": [],
         "error": None},
    ],
}

# A case where NO rater matches the consensus urgency justification (all justifs empty) -> fallback.
_CASE_NO_MATCH = {
    "case_id": "syn-002",
    "text": "Femme de 40 ans, fièvre modérée et fatigue depuis deux jours, sans autre signe.",
    "urgency": "urgence modérée",
    "esi": 3,
    "unanimous": False,
    "is_gold": False,
    "flagged": True,
    "labels": [
        {"model": "openai", "is_triage_case": True, "urgency": "urgence modérée", "esi": 3,
         "consistent": True, "justification": "", "red_flags": [], "error": None},
        {"model": "mistral", "is_triage_case": True, "urgency": "urgence modérée", "esi": 3,
         "consistent": True, "justification": "", "red_flags": [], "error": None},
        {"model": "anthropic", "is_triage_case": True, "urgency": "urgence maximale", "esi": 2,
         "consistent": True, "justification": "signes d'alerte", "red_flags": [], "error": None},
    ],
}


def _assistant(row):
    return row["messages"][2]["content"]


def test_pick_justification_prefers_matching_rater_not_canned():
    justif = pick_justification(_CASE, "urgence différée")
    assert justif == REAL_JUSTIF
    assert justif != CANNED_FALLBACK_FR["urgence différée"]


def test_pick_justification_falls_back_when_no_matching_rater():
    justif = pick_justification(_CASE_NO_MATCH, "urgence maximale")  # no rater said maximale w/ a justif... it did
    # 'urgence maximale' HAS a justif from anthropic, so that returns the real one:
    assert justif == "signes d'alerte"
    # but no rater gave a justif for the CONSENSUS level 'urgence modérée' -> fallback:
    justif_mod = pick_justification(_CASE_NO_MATCH, "urgence modérée")
    assert justif_mod == CANNED_FALLBACK_FR["urgence modérée"]


def test_triage_row_carries_real_justification_and_3part_format():
    row = triage_row_from_case(_CASE, "urgence différée")
    content = _assistant(row)
    # (1) real per-case justification present, canned sentence NOT used
    assert REAL_JUSTIF in content
    assert CANNED_FALLBACK_FR["urgence différée"] not in content
    # (2) fixed 3-part format
    assert "1. Niveau d'urgence : urgence différée." in content
    assert "2. Justification clinique :" in content
    assert "3. Recommandation :" in content
    # (3) FR disclaimer preserved (behavioural metric depends on it)
    assert DISCLAIMER["fr"] in content
    # ChatML message shape matches v9 rows
    assert [m["role"] for m in row["messages"]] == ["system", "user", "assistant"]
    assert row["kind"] == "triage"
    assert row["lang"] == "fr"


def test_build_triage_pool_excludes_eval_gold_leak():
    # The synthetic case's text is declared held-out -> it MUST NOT appear in the pool.
    gold_keys = {" ".join(_CASE["text"].split()).lower()[:80]}
    pool = build_triage_pool([_CASE, _CASE_NO_MATCH], gold_keys)
    users = {r["messages"][1]["content"] for r in pool}
    assert _CASE["text"] not in users, "leaked eval-gold case present in triage pool"
    assert _CASE_NO_MATCH["text"] in users, "non-leaked case wrongly excluded"


def test_build_triage_pool_all_rows_wellformed_with_disclaimer():
    pool = build_triage_pool([_CASE, _CASE_NO_MATCH], set())
    assert pool
    for r in pool:
        c = _assistant(r)
        assert "1. Niveau d'urgence :" in c
        assert "2. Justification clinique :" in c
        assert "3. Recommandation :" in c
        assert DISCLAIMER["fr"] in c
