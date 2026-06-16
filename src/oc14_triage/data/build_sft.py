"""Build the SFT dataset (messages-format JSONL) from the collected raw parquet.

Triage-first composition (mentor §0b): keep ALL triage rows (hand-written vignettes
+ MediQAl 'Urgences' reshaped into the urgency→justification→recommendation structure),
then fill to the target with medical-QA rows that give the model the knowledge it needs
to justify a triage. ~80% FR / 20% EN; deterministic via SEED. Writes train/val JSONL,
a stats JSON, and a human-readable data card.

Usage:  uv run python -m oc14_triage.data.build_sft
"""

from __future__ import annotations

import json
import math
import random
from collections import Counter

import pandas as pd

from ..config import (
    LANG_TARGET,
    PROCESSED,
    RAW,
    SEED,
    SFT_TARGET,
    TRIAGE_MIN,
    URGENCY_LEVELS,
    VAL_FRACTION,
)
from .templates import RECO, chat_example, heuristic_urgency, triage_response
from .vignettes import sft_triage_rows

MEDQUAD_QTYPES = {"symptoms", "treatment", "exams and tests", "complications", "prevention"}

# Urgency-matched justifications used when reshaping a clinical case into the triage form.
TRIAGE_JUSTIF_FR = {
    "urgence maximale": "Le tableau clinique comporte des signes d'alerte qui imposent une prise "
    "en charge sans délai.",
    "urgence modérée": "Le tableau clinique justifie une évaluation rapide, sans signe de gravité "
    "immédiate.",
    "urgence différée": "Le tableau clinique ne comporte pas de signe de gravité aiguë.",
}


def _clean(v):  # pandas reads empty string cells as NaN (float); normalise to ""
    return "" if (v is None or (isinstance(v, float) and math.isnan(v))) else v


def _rows(path) -> list[dict]:
    if not path.exists():
        return []
    return [{k: _clean(v) for k, v in r.items()}
            for r in pd.read_parquet(path).to_dict("records")]


def _mediqal_oeq_qa() -> list[dict]:
    out = []
    for r in _rows(RAW / "mediqal" / "oeq__test.parquet"):
        ans = (r.get("answer") or "").strip()
        if len(ans) < 50:
            continue
        case = (r.get("clinical_case") or "").strip()
        q = (r.get("question") or "").strip()
        user = f"{case}\n\n{q}".strip() if case else q
        out.append(chat_example(user, ans, "fr", "mediqal_oeq", "qa"))
    return out


def _mediqal_mcqu_qa() -> list[dict]:
    out = []
    for r in _rows(RAW / "mediqal" / "mcqu__train.parquet"):
        if (r.get("clinical_case") or "").strip():
            continue  # case-bearing rows are reshaped into triage instead
        letter = (r.get("correct_answers") or "").strip()
        if len(letter) != 1:
            continue
        ans_text = (r.get(f"answer_{letter.lower()}") or "").strip()
        if not ans_text:
            continue
        opts = "  ".join(f"{c.upper()}) {r.get('answer_' + c, '')}" for c in "abcde"
                         if r.get(f"answer_{c}"))
        case = (r.get("clinical_case") or "").strip()
        q = (r.get("question") or "").strip()
        user = "\n".join(x for x in (case, q, opts) if x)
        out.append(chat_example(user, f"La réponse est {letter} : {ans_text}.", "fr",
                                "mediqal_mcqu", "qa"))
    return out


def _mediqal_clinical_triage() -> list[dict]:
    """Reshape MediQAl clinical-case rows into the triage structure (heuristic urgency).

    A clinical_case IS a patient presentation — the natural input to triage. Urgency is a
    documented HEURISTIC (red-flag keywords); the held-out eval vignettes are hand-labelled
    by a different process, so the triage metric is not circular (§0b).
    """
    out = []
    for f in ("mcqu__train.parquet", "mcqm__train.parquet"):
        for r in _rows(RAW / "mediqal" / f):
            case = (r.get("clinical_case") or "").strip()
            if len(case) < 30:
                continue
            level = heuristic_urgency(case, "fr")
            assistant = triage_response(level, TRIAGE_JUSTIF_FR[level], RECO["fr"][level], "fr")
            out.append(chat_example(case, assistant, "fr", "mediqal_triage", "triage"))
    return out


def _medquad_en_qa() -> list[dict]:
    out = []
    for r in _rows(RAW / "medquad" / "default__train.parquet"):
        if str(r.get("qtype", "")).lower() not in MEDQUAD_QTYPES:
            continue
        ans = (r.get("Answer") or "").strip()
        q = (r.get("Question") or "").strip()
        if len(ans) < 100 or not q:
            continue
        out.append(chat_example(q, ans, "en", "medquad", "qa"))
    return out


def _dedup(rows: list[dict]) -> list[dict]:
    seen, out = set(), []
    for r in rows:
        key = r["messages"][1]["content"][:200]  # user turn
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


def build() -> dict:
    rng = random.Random(SEED)
    triage = _dedup(sft_triage_rows() + _mediqal_clinical_triage())
    fr_qa = _dedup(_mediqal_oeq_qa() + _mediqal_mcqu_qa())
    en_qa = _dedup(_medquad_en_qa())
    for pool in (triage, fr_qa, en_qa):
        rng.shuffle(pool)

    total = round(SFT_TARGET / (1 - VAL_FRACTION))  # build a bit extra, then split off val
    n_triage = min(len(triage), max(TRIAGE_MIN, round(0.28 * total)))
    n_en = min(len(en_qa), round(LANG_TARGET["en"] * total))
    n_fr = max(0, total - n_triage - n_en)
    selected = triage[:n_triage] + fr_qa[:n_fr] + en_qa[:n_en]
    rng.shuffle(selected)

    val_n = round(VAL_FRACTION * len(selected))
    val, train = selected[:val_n], selected[val_n:]

    PROCESSED.mkdir(parents=True, exist_ok=True)
    for name, rows in (("sft_train", train), ("sft_val", val)):
        with open(PROCESSED / f"{name}.jsonl", "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    def stats(rows):
        urg = Counter()
        for r in rows:
            if r["kind"] != "triage":
                continue
            a = r["messages"][2]["content"]
            urg[next((lv for lv in URGENCY_LEVELS if lv in a), "?")] += 1
        return {"n": len(rows), "lang": dict(Counter(r["lang"] for r in rows)),
                "kind": dict(Counter(r["kind"] for r in rows)),
                "source": dict(Counter(r["source"] for r in rows)),
                "urgency_in_triage": dict(urg)}

    report = {"seed": SEED, "available": {"triage": len(triage), "fr_qa": len(fr_qa),
              "en_qa": len(en_qa)}, "train": stats(train), "val": stats(val)}
    (PROCESSED / "_sft_stats.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


if __name__ == "__main__":
    rep = build()
    print(json.dumps(rep, indent=2, ensure_ascii=False))
