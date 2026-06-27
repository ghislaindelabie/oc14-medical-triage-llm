"""Build the SFT *retrain* set on the LLM-consensus triage labels — the single clean chokepoint.

  - keep ONLY the medical-QA + EN breadth from the old build (drop ALL kind=triage: the heuristic
    `mediqal_triage` AND the old vignettes);
  - re-add ALL hand-written vignettes fresh (E5 — the old build randomly dropped 4/11);
  - add the clean LLM-consensus triage (cmd_build output, already non-consensus-filtered, E1);
  - DROP any train/val row whose clinical case is a held-out eval-gold case (E2/E4 leakage fix);
  - carve a 150-row triage validation slice so val loss reflects the task.

Writes train/val into data/kaggle_upload/ + copies the 300-case eval-gold alongside.
"""
from __future__ import annotations

import collections
import json
import random
import shutil

from oc14_triage.config import PROCESSED, ROOT, SEED
from oc14_triage.data.vignettes import sft_triage_rows

UP = ROOT / "data" / "kaggle_upload"


def load(name: str) -> list[dict]:
    return [json.loads(x) for x in (PROCESSED / name).read_text(encoding="utf-8").split("\n") if x.strip()]


def dump(rows: list[dict], name: str) -> None:
    with open(UP / name, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def _user(r: dict) -> str:
    m = r["messages"]
    return m[1]["content"] if len(m) > 1 else ""


def _key(text: str) -> str:  # normalized 80-char prefix — robust clinical-case identity
    return " ".join((text or "").split()).lower()[:80]


# Oversample the hand-written vignettes: they are high-quality, the main EN-triage signal, AND carry
# balanced 'différée' — the two classes the model is short on. Modest overfit on these archetypes is fine.
VIGN_OVERSAMPLE = 8

old_train, old_val, new_tri = load("sft_train.jsonl"), load("sft_val.jsonl"), load("triage_sft_train.jsonl")
gold = load("triage_eval_gold.jsonl")

# Keep ONLY QA from the old build (drop all kind=triage: heuristic + old vignettes — re-added fresh).
qa = lambda rows: [r for r in rows if r.get("kind") != "triage"]  # noqa: E731
old_qa_train, old_qa_val = qa(old_train), qa(old_val)
vignettes = sft_triage_rows()  # all 11 hand-written, always kept (E5)

# Carve a triage validation slice so val loss reflects the triage task.
rng = random.Random(SEED)
rng.shuffle(new_tri)
tri_val, tri_train = new_tri[:150], new_tri[150:]

train = old_qa_train + vignettes * VIGN_OVERSAMPLE + tri_train
val = old_qa_val + tri_val

# (E2/E4) Drop any train/val row whose clinical case is a held-out eval-gold case (input leak via the
# QA reshape, which embeds the same clinical_case text under a different task framing).
gold_keys = {_key(g["user"]) for g in gold}
n0 = (len(train), len(val))
train = [r for r in train if _key(_user(r)) not in gold_keys]
val = [r for r in val if _key(_user(r)) not in gold_keys]
leaked = (n0[0] - len(train), n0[1] - len(val))
assert all(_key(_user(r)) not in gold_keys for r in train + val), "eval-gold leak remains"

rng.shuffle(train)
rng.shuffle(val)
UP.mkdir(parents=True, exist_ok=True)
dump(train, "sft_train.jsonl")
dump(val, "sft_val.jsonl")
shutil.copy(PROCESSED / "triage_eval_gold.jsonl", UP / "triage_eval_gold.jsonl")

comp = lambda rows: dict(collections.Counter(r.get("source") for r in rows))  # noqa: E731
print(f"train {len(train)}: {comp(train)}")
print(f"val   {len(val)}: {comp(val)}")
print(f"vignettes (×{VIGN_OVERSAMPLE}): {sum(r.get('source') == 'vignette' for r in train)} "
      f"(expect {len(vignettes) * VIGN_OVERSAMPLE})")
print(f"eval-gold leak rows dropped: train {leaked[0]}, val {leaked[1]}")
print(f"copied triage_eval_gold.jsonl ({len(gold)}) -> {UP}")
