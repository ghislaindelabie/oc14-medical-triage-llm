"""Pure helpers for the RunPod experiment loop: verify the DPO-pair balance invariant, and parse a
job's results JSON into a decision (did DPO beat / hold vs the SFT v9 baseline; did rpo_alpha stop
the collapse). No I/O — fed dicts/lists so it is unit-tested directly.
"""
from __future__ import annotations

from collections import Counter

LEVELS = ("urgence maximale", "urgence modérée", "urgence différée")
# The served model's canonical macro-F1 (n=300 triage_eval_gold, oc14-sft-eval harness) = 0.822.
# FAIR same-harness comparison (eval_fair_startup.sh: BOTH models on the SAME n=300 with the
# canonical batched harness, adapter toggled via disable_adapter): v9=0.827, v9+DPO(rpo_alpha=1.0)
# =0.845 -> +0.018, NO class collapse (recalls 0.82-0.88). That +0.018 IS apples-to-apples.
SFT_V9_MACRO_F1 = 0.822


def _level_of(msgs) -> str | None:
    """First urgency level named in a chosen/rejected message list."""
    text = ""
    if isinstance(msgs, list) and msgs:
        text = (msgs[0].get("content") or "") if isinstance(msgs[0], dict) else str(msgs[0])
    else:
        text = str(msgs)
    low = text.lower()
    for lv in LEVELS:
        if lv in low:
            return lv
    return None


def pair_balance(pairs: list[dict]) -> dict:
    """Distribution of chosen and rejected levels across a DPO pair list."""
    return {
        "n": len(pairs),
        "chosen": dict(Counter(_level_of(p.get("chosen")) for p in pairs)),
        "rejected": dict(Counter(_level_of(p.get("rejected")) for p in pairs)),
    }


def check_balance(pairs: list[dict]) -> tuple[bool, list[str]]:
    """The anti-collapse balance invariant on the preference set.

    Guards exactly the failure modes that sank the prior DPO attempts:
      1. every urgency level appears as BOTH chosen and rejected (no structural class penalty);
      2. the middle class 'urgence modérée' is NOT disproportionately the rejected one — it must
         also be chosen a healthy number of times (the prior collapse pushed everything off the
         middle). We require modérée-chosen >= 25% of modérée-rejected.
    Returns (ok, reasons_if_not_ok).
    """
    bal = pair_balance(pairs)
    reasons: list[str] = []
    for lv in LEVELS:
        if bal["chosen"].get(lv, 0) == 0:
            reasons.append(f"{lv!r} never appears as CHOSEN")
        if bal["rejected"].get(lv, 0) == 0:
            reasons.append(f"{lv!r} never appears as REJECTED")
    mod_chosen = bal["chosen"].get("urgence modérée", 0)
    mod_rejected = bal["rejected"].get("urgence modérée", 0)
    if mod_rejected and mod_chosen < 0.25 * mod_rejected:
        reasons.append(
            f"'urgence modérée' structurally penalized: chosen {mod_chosen} < 25% of "
            f"rejected {mod_rejected} — this caused the prior middle-class collapse")
    return (not reasons, reasons)


def summarize_dpo_result(result: dict, baseline: float = SFT_V9_MACRO_F1) -> dict:
    """Turn a DPO job's results dict into a soutenance-ready verdict.

    Expected `result` keys (written by the pod): `macro_f1`, `recall_per_level` (dict),
    optionally `accuracy`, `run_url`. Detects collapse = any class recall == 0 (the tell-tale of
    the prior no-rpo_alpha runs, which zeroed out whole classes).
    """
    macro = result.get("macro_f1")
    recalls = result.get("recall_per_level") or {}
    collapsed_classes = [lv for lv in LEVELS if (recalls.get(lv) is not None and recalls[lv] == 0)]
    delta = None if macro is None else round(macro - baseline, 4)
    if macro is None:
        verdict = "no macro_f1 in results"
    elif collapsed_classes:
        verdict = f"COLLAPSE: zero recall on {collapsed_classes}"
    elif macro >= baseline:
        verdict = f"beats/ties SFT v9 ({macro:.3f} >= {baseline:.3f})"
    else:
        verdict = f"below SFT v9 ({macro:.3f} < {baseline:.3f}) but no collapse"
    return {
        "macro_f1": macro,
        "baseline": baseline,
        "delta_vs_v9": delta,
        "collapsed_classes": collapsed_classes,
        "no_collapse": not collapsed_classes,
        "verdict": verdict,
    }
