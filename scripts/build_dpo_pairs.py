"""Triage-preference DPO set (cost-weighted, clear-cut) — the fair second DPO attempt.

`chosen` = the correct (consensus gold) level ; `rejected` = a WRONG level — BOTH directions, with
**extra weight on under-triage** (the dangerous error). Both sides use the SAME level-appropriate
generic justification, so DPO learns the *level* preference, not justification richness. Sources: the
**unanimous gold** cases (clear-cut), EXCLUDING the held-out eval-gold (no leak), plus the **11
hand-written safety pairs**. The ambiguous 2-of-3 splits are deliberately NOT used (unreliable
preference). Writes dpo_train/dpo_val into data/kaggle_upload/.
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
UNDER = {"urgence maximale": "urgence modérée", "urgence modérée": "urgence différée"}  # under-triage by 1
OVER = {"urgence différée": "urgence modérée", "urgence modérée": "urgence maximale"}    # over-triage by 1
N_GOLD = 1200  # cap base pairs to keep DPO ~30-40 min on a T4


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
rng = random.Random(SEED)
rng.shuffle(gold)
gold = gold[:N_GOLD]

pairs = []
for i, r in enumerate(gold):
    lvl = r["urgency"]
    if lvl == "urgence maximale":                       # don't UNDER-triage an emergency (safety)
        p = _pair(r["text"], lvl, UNDER[lvl], "dpo_under")
        pairs += [p, p]                                  # 2× cost weight
    elif lvl == "urgence différée":                      # don't OVER-triage a clear low-acuity case
        pairs.append(_pair(r["text"], lvl, OVER[lvl], "dpo_over"))
    else:                                                # modérée — alternate both directions
        pairs.append(_pair(r["text"], lvl, UNDER[lvl] if i % 2 else OVER[lvl], "dpo_mod"))

pairs += dpo_safety_pairs()                              # 11 clear-cut hand-written safety pairs
rng.shuffle(pairs)
n_val = round(VAL_FRACTION * len(pairs))
val, train = pairs[:n_val], pairs[n_val:]

UP.mkdir(parents=True, exist_ok=True)
for name, rows in (("dpo_train.jsonl", train), ("dpo_val.jsonl", val)):
    with open(UP / name, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"dpo train {len(train)} / val {len(val)} | sources {dict(Counter(p['source'] for p in pairs))}")
