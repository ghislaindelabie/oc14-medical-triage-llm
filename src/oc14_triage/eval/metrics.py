"""Triage-first evaluation metrics — pure functions, no model/network needed.

Leads with what fine-tuning actually changes (triage behaviour), per mentor §0b:
urgency agreement, red-flag escalation recall, disclaimer presence, response-format
adherence, language match, and a no-<think> check. MCQA accuracy lives separately as
a "didn't lose medical knowledge" sanity check. Kept dependency-light; scikit-learn is
used only if installed (the `eval` extra) for Cohen's κ.
"""

from __future__ import annotations

import re
from collections import Counter

from ..config import URGENCY_LEVELS

_DISCLAIMER = ("ne remplace pas", "does not replace")
_FR_HINTS = {"le", "la", "les", "une", "des", "et", "avec", "niveau", "urgence", "patient",
             "ans", "clinique", "recommandation"}
_EN_HINTS = {"the", "and", "with", "level", "urgency", "patient", "old", "years", "clinical",
             "recommendation"}


def extract_urgency(text: str) -> str | None:
    """Return the urgency level mentioned in a response, or None."""
    low = (text or "").lower()
    found = [(low.index(lv), lv) for lv in URGENCY_LEVELS if lv in low]
    return min(found)[1] if found else None


def has_disclaimer(text: str) -> bool:
    low = (text or "").lower()
    return any(m in low for m in _DISCLAIMER)


def has_think_block(text: str) -> bool:
    """A served triage answer must never contain reasoning <think> blocks (§0b Decision H)."""
    return "<think>" in (text or "").lower()


def format_ok(text: str) -> bool:
    """Does the answer follow the triage structure (urgency level + a recommendation)?"""
    low = (text or "").lower()
    return extract_urgency(text) is not None and ("recommand" in low or "recommendation" in low)


def response_language(text: str) -> str:
    """Crude FR/EN detector (documented as heuristic) for language-match scoring."""
    words = re.findall(r"[a-zàâçéèêëîïôûùüÿñæœ]+", (text or "").lower())
    fr = sum(w in _FR_HINTS for w in words) + sum(c in "àâçéèêëîïôûù" for c in (text or ""))
    en = sum(w in _EN_HINTS for w in words)
    return "fr" if fr >= en else "en"


def triage_report(pairs: list[tuple[str, str]]) -> dict:
    """Score (predicted, gold) urgency pairs. Reports accuracy, per-class recall, confusion,
    and the safety-critical recall on 'urgence maximale'. Adds Cohen's κ if sklearn is present."""
    pairs = [(p, g) for p, g in pairs if g in URGENCY_LEVELS]
    n = len(pairs)
    if not n:
        return {"n": 0}
    correct = sum(p == g for p, g in pairs)
    confusion = Counter((g, p) for p, g in pairs)
    recall = {}
    for lv in URGENCY_LEVELS:
        gold = [p for p, g in pairs if g == lv]
        recall[lv] = round(sum(p == lv for p in gold) / len(gold), 3) if gold else None
    out = {
        "n": n,
        "accuracy": round(correct / n, 3),
        "recall_per_level": recall,
        "recall_urgence_maximale": recall["urgence maximale"],  # safety-critical
        "confusion_gold_pred": {f"{g}->{p}": c for (g, p), c in sorted(confusion.items())},
    }
    try:
        from sklearn.metrics import cohen_kappa_score  # optional (eval extra)
        out["cohen_kappa"] = round(
            cohen_kappa_score([g for _, g in pairs], [p for p, _ in pairs],
                              labels=list(URGENCY_LEVELS)), 3)
    except Exception:  # noqa: BLE001
        out["cohen_kappa"] = None
    return out


def behavioural_report(responses: list[dict]) -> dict:
    """Aggregate behavioural rates over responses [{text, lang}]. Rates in [0,1]."""
    n = len(responses) or 1
    return {
        "n": len(responses),
        "disclaimer_rate": round(sum(has_disclaimer(r["text"]) for r in responses) / n, 3),
        "format_ok_rate": round(sum(format_ok(r["text"]) for r in responses) / n, 3),
        "no_think_rate": round(sum(not has_think_block(r["text"]) for r in responses) / n, 3),
        "language_match_rate": round(
            sum(response_language(r["text"]) == r.get("lang", "fr") for r in responses) / n, 3),
    }
