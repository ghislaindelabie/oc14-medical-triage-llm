"""Load source material from the local MediQAl parquet: triage vignettes + MCQU questions."""

from __future__ import annotations

import hashlib
import math

import pandas as pd

from ..config import RAW

_MEDIQAL = RAW / "mediqal"


def _clean(v) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return ""
    return str(v).strip()


def load_triage_cases(limit: int | None = None) -> list[dict]:
    """Unique clinical-case vignettes (len>=40) across all MediQAl splits, stable id, sorted."""
    seen: dict[str, str] = {}
    for f in sorted(_MEDIQAL.glob("*.parquet")):
        df = pd.read_parquet(f)
        if "clinical_case" not in df.columns:
            continue
        for v in df["clinical_case"].tolist():
            t = _clean(v)
            if len(t) >= 40:
                cid = "mtc-" + hashlib.sha1(t.encode("utf-8")).hexdigest()[:10]
                seen.setdefault(cid, t)
    cases = [{"case_id": c, "text": t} for c, t in sorted(seen.items())]
    return cases[:limit] if limit else cases


def load_mcqu_questions(limit: int | None = None, split: str = "test") -> list[dict]:
    """MediQAl MCQU single-answer questions with the real answer key (for labeler calibration)."""
    path = _MEDIQAL / f"mcqu__{split}.parquet"
    if not path.exists():
        return []
    out = []
    for r in pd.read_parquet(path).to_dict("records"):
        letter = _clean(r.get("correct_answers")).upper()
        if len(letter) != 1 or letter not in "ABCDE":
            continue
        opts = "\n".join(f"{c.upper()}) {_clean(r.get('answer_' + c))}" for c in "abcde"
                         if _clean(r.get(f"answer_{c}")))
        case = _clean(r.get("clinical_case"))
        q = _clean(r.get("question"))
        stem = "\n".join(x for x in (case, q, opts) if x)
        out.append({"qid": _clean(r.get("id")) or f"q{len(out)}", "prompt": stem, "answer": letter})
        if limit and len(out) >= limit:
            break
    return out
