"""Build the SFT *retrain* set on the LLM-consensus triage labels.

Decision (Option B): drop the OLD heuristic triage rows (`source=mediqal_triage` — the same MediQAl
vignettes we just relabelled by 3-model consensus, so keeping them = contradictory labels for the
same cases), KEEP the medical-QA + EN breadth (mediqal_mcqu/oeq, medquad, vignettes), and ADD the
LLM-consensus triage. Writes the combined train/val into data/kaggle_upload/ and copies the
300-case stratified eval-gold alongside, ready for `kaggle datasets version`.
"""
from __future__ import annotations

import collections
import json
import random
import shutil

from oc14_triage.config import PROCESSED, ROOT, SEED

UP = ROOT / "data" / "kaggle_upload"


def load(name: str) -> list[dict]:
    return [json.loads(x) for x in (PROCESSED / name).read_text(encoding="utf-8").split("\n") if x.strip()]


def dump(rows: list[dict], name: str) -> None:
    with open(UP / name, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


old_train, old_val, new_tri = load("sft_train.jsonl"), load("sft_val.jsonl"), load("triage_sft_train.jsonl")

# Drop the old heuristic triage from both splits.
kept = lambda rows: [r for r in rows if r.get("source") != "mediqal_triage"]  # noqa: E731
old_train_kept, old_val_kept = kept(old_train), kept(old_val)

# Carve a small triage validation slice so val loss reflects the triage task too.
rng = random.Random(SEED)
rng.shuffle(new_tri)
tri_val, tri_train = new_tri[:150], new_tri[150:]

train, val = old_train_kept + tri_train, old_val_kept + tri_val
rng.shuffle(train)
rng.shuffle(val)

UP.mkdir(parents=True, exist_ok=True)
dump(train, "sft_train.jsonl")
dump(val, "sft_val.jsonl")
shutil.copy(PROCESSED / "triage_eval_gold.jsonl", UP / "triage_eval_gold.jsonl")

comp = lambda rows: dict(collections.Counter(r.get("source") for r in rows))  # noqa: E731
print(f"train {len(train)} (old-non-triage {len(old_train_kept)} + new-triage {len(tri_train)})")
print(f"   sources: {comp(train)}")
print(f"val   {len(val)} (old-non-triage {len(old_val_kept)} + new-triage {len(tri_val)})")
print(f"dropped heuristic triage: train {len(old_train) - len(old_train_kept)}, val {len(old_val) - len(old_val_kept)}")
print(f"copied triage_eval_gold.jsonl -> {UP}")
