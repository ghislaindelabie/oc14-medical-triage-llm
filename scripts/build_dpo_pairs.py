"""Triage-preference DPO set (clear-cut, DIRECTION-BALANCED) — the fair second DPO attempt.

`chosen` = the correct (consensus gold) level ; `rejected` = a WRONG adjacent level. Both sides use the
SAME level-appropriate generic justification, so DPO learns the *level* preference, not justification
richness. **Direction-balanced** so we don't re-introduce a monotonic over-triage bias: every scarce
clear-*différée* case feeds the OVER-direction (protect low-acuity recall), the maximale pool is *capped*
to ~2:1 for a modest safety lean (not 30:1), modérée contributes both directions. Sources: the unanimous
gold (EXCLUDING the held-out eval-gold) + the 11 hand-written safety pairs. Ambiguous 2-of-3 splits are
NOT used. Writes dpo_train/dpo_val into data/kaggle_upload/.
"""
from __future__ import annotations

import json
import random
from collections import Counter

from oc14_triage.config import PROCESSED, ROOT, SEED, SYSTEM_PROMPT, VAL_FRACTION
from oc14_triage.data.build_sft import TRIAGE_JUSTIF_FR
from oc14_triage.data.templates import RECO, triage_response
from oc14_triage.data.vignettes import dpo_safety_pairs

UP = ROOT / "data" / "kaggle_upload"
MAX, MOD, DIF = "urgence maximale", "urgence modérée", "urgence différée"


def _resp(level: str) -> str:
    return triage_response(level, TRIAGE_JUSTIF_FR[level], RECO["fr"][level], "fr")


def _pair(case: str, correct: str, wrong: str, source: str) -> dict:
    return {"prompt": [{"role": "system", "content": SYSTEM_PROMPT["fr"]},
                       {"role": "user", "content": case}],
            "chosen": [{"role": "assistant", "content": _resp(correct)}],
            "rejected": [{"role": "assistant", "content": _resp(wrong)}],
            "lang": "fr", "source": source}


def _load(path):
    return [json.loads(x) for x in path.read_text(encoding="utf-8").split("\n") if x.strip()]


eval_ids = {r["case_id"] for r in _load(UP / "triage_eval_gold.jsonl")}
gold = [r for r in _load(PROCESSED / "triage_labeled.jsonl")
        if r.get("is_gold") and r["case_id"] not in eval_ids]
by = {lv: [r for r in gold if r["urgency"] == lv] for lv in (MAX, MOD, DIF)}
rng = random.Random(SEED)
for v in by.values():
    rng.shuffle(v)

n_dif = len(by[DIF])
n_under = min(len(by[MAX]), 2 * n_dif)   # safety lean ~2:1, bounded — NOT a monotonic upward push
n_mod = min(len(by[MOD]), n_dif)

pairs = [_pair(r["text"], DIF, MOD, "dpo_over") for r in by[DIF]]                 # protect différée recall
pairs += [_pair(r["text"], MAX, MOD, "dpo_under") for r in by[MAX][:n_under]]     # safety: never under-triage
pairs += [_pair(r["text"], MOD, DIF if i % 2 else MAX, "dpo_mod")                 # modérée: both directions
          for i, r in enumerate(by[MOD][:n_mod])]
pairs += dpo_safety_pairs()                                                       # 11 clear-cut safety pairs

rng.shuffle(pairs)
n_val = round(VAL_FRACTION * len(pairs))
val, train = pairs[:n_val], pairs[n_val:]
UP.mkdir(parents=True, exist_ok=True)
for name, rows in (("dpo_train.jsonl", train), ("dpo_val.jsonl", val)):
    with open(UP / name, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
under = sum(p["source"] in ("dpo_under", "safety") for p in pairs)
over = sum(p["source"] == "dpo_over" for p in pairs)
print(f"dpo train {len(train)} / val {len(val)} | sources {dict(Counter(p['source'] for p in pairs))}")
print(f"direction balance — under(+safety): {under}  over: {over}  (modérée mixed: {n_mod})")
