"""Build the DPO preference dataset (conversational JSONL).

Two parts in one consistent schema (system+user prompt, assistant chosen/rejected):
  • hand-written **bilingual safety pairs** (chosen = correct triage escalation,
    rejected = unsafe reassurance) — the genuine clinical-safety signal;
  • a filtered slice of **UltraMedical-Preference** (English, GPT-4-scored) kept only
    where the score gap is meaningful — framed as a DPO technique demo, not a
    clinical-quality signal (mentor §0b / Decision C).

Usage:  uv run python -m oc14_triage.data.build_dpo
"""

from __future__ import annotations

import json
import random
from collections import Counter

import pandas as pd

from ..config import DPO_TARGET, PROCESSED, RAW, SEED, SYSTEM_PROMPT, VAL_FRACTION
from .vignettes import dpo_safety_pairs

CHOSEN_MIN = 4.5
REJECTED_MAX = 4.0


def _assistant_text(turns) -> str:
    """Last assistant message content from a chosen/rejected turn list."""
    if turns is None:
        return ""
    seq = list(turns)
    for t in reversed(seq):
        if isinstance(t, dict) and t.get("role") == "assistant":
            return (t.get("content") or "").strip()
    return (seq[-1].get("content", "") if seq and isinstance(seq[-1], dict) else "").strip()


def _score(meta, side: str) -> float | None:
    if isinstance(meta, str):
        try:
            meta = json.loads(meta)
        except Exception:  # noqa: BLE001
            return None
    try:
        return float(meta[side]["score"])
    except (KeyError, TypeError, ValueError):
        return None


def _ultramedical(limit: int) -> list[dict]:
    rows = pd.read_parquet(RAW / "ultramedical_pref" / "default__train.parquet").to_dict("records")
    out = []
    for r in rows:
        if str(r.get("prompt_id", "")).startswith("MedQuad"):  # avoid SFT overlap
            continue
        cs, rs = _score(r.get("metadata"), "chosen"), _score(r.get("metadata"), "rejected")
        if cs is None or rs is None or cs < CHOSEN_MIN or rs > REJECTED_MAX:
            continue
        chosen, rejected = _assistant_text(r.get("chosen")), _assistant_text(r.get("rejected"))
        prompt = (r.get("prompt") or "").strip()
        if not (chosen and rejected and prompt) or chosen == rejected:
            continue
        out.append({
            "prompt": [{"role": "system", "content": SYSTEM_PROMPT["en"]},
                       {"role": "user", "content": prompt}],
            "chosen": [{"role": "assistant", "content": chosen}],
            "rejected": [{"role": "assistant", "content": rejected}],
            "lang": "en",
            "source": "ultramedical",
        })
        if len(out) >= limit:
            break
    return out


def build() -> dict:
    rng = random.Random(SEED)
    safety = dpo_safety_pairs()
    ultra = _ultramedical(max(0, DPO_TARGET - len(safety)))
    rows = safety + ultra
    rng.shuffle(rows)

    val_n = round(VAL_FRACTION * len(rows))
    val, train = rows[:val_n], rows[val_n:]

    PROCESSED.mkdir(parents=True, exist_ok=True)
    for name, part in (("dpo_train", train), ("dpo_val", val)):
        with open(PROCESSED / f"{name}.jsonl", "w", encoding="utf-8") as fh:
            for r in part:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    def stats(part):
        return {"n": len(part), "lang": dict(Counter(r["lang"] for r in part)),
                "source": dict(Counter(r["source"] for r in part))}

    report = {"seed": SEED, "filter": {"chosen_score>=": CHOSEN_MIN, "rejected_score<=": REJECTED_MAX},
              "safety_pairs": len(safety), "ultramedical_kept": len(ultra),
              "train": stats(train), "val": stats(val)}
    (PROCESSED / "_dpo_stats.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


if __name__ == "__main__":
    print(json.dumps(build(), indent=2, ensure_ascii=False))
