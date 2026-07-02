"""Build the **SFT v10** training set — the DPO post-mortem's top pick: not DPO, a better SFT.

v10 = the v9 recipe (QA breadth from the old build + all hand-written vignettes ×N + the clean
LLM-consensus triage, leak-free against the 300 held-out eval-gold cases), with TWO improvements:

  1. REAL per-case justifications. Each triage row is rebuilt straight from `triage_labeled.jsonl`,
     re-deriving consensus with the current logic, and taking the justification from a rater whose
     `urgency` == the case's consensus urgency (prefer a `consistent: true` rater; first if several;
     canned fallback only if none matches). The 3-part output format + FR disclaimer are unchanged,
     so the behavioural metrics stay 1.00. (v9's llm_triage rows already carried real justifications;
     v10 makes that a guarantee of construction and rebalances on top of it.)

  2. Rebalance the weak classes. v9's triage urgency mix was maximale-heavy (train 1098/689/254).
     v10 keeps ALL 'différée' and ALL 'modérée' consensus cases and caps the abundant 'maximale' with
     a seeded downsample, lifting the différée/modérée share while staying FR-primary.

ADDITIVE artifact: writes to `data/kaggle_upload_v10/` (v9's `data/kaggle_upload/` is left intact) and
copies the existing dpo_train/dpo_val/triage_eval_gold alongside, so the folder is a complete,
uploadable dataset with the SAME filenames the Kaggle notebook globs.

Usage:  uv run python scripts/build_sft_v10.py
"""
from __future__ import annotations

import collections
import json
import random
import shutil

from oc14_triage.config import PROCESSED, ROOT, SEED, URGENCY_LEVELS, VAL_FRACTION
from oc14_triage.data.templates import DISCLAIMER, RECO, chat_example, triage_response
from oc14_triage.data.vignettes import sft_triage_rows
from oc14_triage.labeling.aggregate import Label, consensus

V9 = ROOT / "data" / "kaggle_upload"  # shipped v9 — read (old QA + DPO/eval to copy), never written
V10 = ROOT / "data" / "kaggle_upload_v10"  # v10 output (additive)
LABELED = PROCESSED / "triage_labeled.jsonl"

# Same oversample as v9: the hand-written vignettes are the high-quality, balanced-'différée' core.
VIGN_OVERSAMPLE = 8

# Rebalance cap: keep ALL 'différée' + ALL 'modérée' consensus cases, downsample the abundant
# 'maximale' to at most this many (seeded). Chosen to lift the two weak classes' share materially
# while keeping the triage set large and FR-primary. See report for the resulting distribution.
MAX_MAXIMALE = 500

# Fallback ONLY when no rater's justification matches the consensus urgency (rare). Mirrors v9's
# TRIAGE_JUSTIF_FR wording so the format is unchanged; the whole point of v10 is that this is rare.
CANNED_FALLBACK_FR = {
    "urgence maximale": "Le tableau clinique comporte des signes d'alerte qui imposent une prise "
    "en charge sans délai.",
    "urgence modérée": "Le tableau clinique justifie une évaluation rapide, sans signe de gravité "
    "immédiate.",
    "urgence différée": "Le tableau clinique ne comporte pas de signe de gravité aiguë.",
}


def load(path) -> list[dict]:
    # split on "\n" only (NOT splitlines() — that also splits on U+2028/U+0085 in medical text)
    return [json.loads(x) for x in path.read_text(encoding="utf-8").split("\n") if x.strip()]


def dump(rows: list[dict], name: str) -> None:
    with open(V10 / name, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def _user(r: dict) -> str:
    m = r["messages"]
    return m[1]["content"] if len(m) > 1 else ""


def _key(text: str) -> str:  # normalized 80-char prefix — same robust case identity v9 dedups on
    return " ".join((text or "").split()).lower()[:80]


def _label_from_dict(d: dict) -> Label:
    """Reconstruct a labeling.Label from a stored JSONL label dict (same as labeling.run)."""
    return Label(model=d.get("model", ""), is_triage_case=bool(d.get("is_triage_case")),
                 urgency=d.get("urgency"), esi=d.get("esi"), consistent=bool(d.get("consistent")),
                 justification=d.get("justification", "") or "",
                 red_flags=tuple(d.get("red_flags") or ()), error=d.get("error"))


def pick_justification(case: dict, level: str) -> str:
    """The REAL per-case justification: from a rater whose urgency == the case's consensus level.

    Prefer a `consistent: true` rater; if several, the first; if none matches (no rater at that level
    with a non-empty justification), fall back to the canned sentence. Never invents text.
    """
    labels = case.get("labels", [])
    matching = [d for d in labels
                if d.get("urgency") == level and (d.get("justification") or "").strip()]
    if not matching:
        return CANNED_FALLBACK_FR[level]
    consistent = [d for d in matching if d.get("consistent")]
    chosen = consistent[0] if consistent else matching[0]
    return (chosen.get("justification") or "").strip()


def triage_row_from_case(case: dict, level: str) -> dict:
    """One v10 triage SFT row: fixed 3-part FR format + disclaimer, REAL per-case justification."""
    assistant = triage_response(level, pick_justification(case, level), RECO["fr"][level], "fr")
    return chat_example(case["text"], assistant, "fr", "llm_triage", "triage")


def build_triage_pool(records: list[dict], gold_keys: set[str]) -> list[dict]:
    """From labeled records, build leak-free triage rows with real justifications, rebalanced.

    Keeps gold-leftover (unanimous) + 2-of-3 majority (n_agree>=2) cases with a consensus urgency,
    excludes any case whose text is a held-out eval-gold case, keeps ALL 'différée'/'modérée' and
    caps 'maximale' with a seeded downsample. Returns messages-format rows.
    """
    by_level: dict[str, list[dict]] = {lv: [] for lv in URGENCY_LEVELS}
    for r in records:
        if _key(r["text"]) in gold_keys:  # (E2/E4) never train on a held-out eval-gold case
            continue
        c = consensus(r["case_id"], [_label_from_dict(d) for d in r["labels"]])
        # keep gold-leftover (unanimous) OR legitimate 2-of-3 majority; drop no-majority 3-way splits
        if not c.urgency or c.n_agree < 2:
            continue
        by_level[c.urgency].append(triage_row_from_case(r, c.urgency))

    rng = random.Random(SEED)
    pool: list[dict] = []
    for lv in URGENCY_LEVELS:
        rows = by_level[lv]
        if lv == "urgence maximale" and len(rows) > MAX_MAXIMALE:
            rng.shuffle(rows)
            rows = rows[:MAX_MAXIMALE]  # cap the abundant class; keep 'modérée'/'différée' in full
        pool.extend(rows)
    return pool


def build() -> dict:
    old_train, old_val = load(V9 / "sft_train.jsonl"), load(V9 / "sft_val.jsonl")
    gold = load(V9 / "triage_eval_gold.jsonl")
    labeled = load(LABELED)

    # Keep ONLY QA from the v9 build (drop v9's kind=triage — re-added fresh, rebalanced, below).
    old_qa_train = [r for r in old_train if r.get("kind") != "triage"]
    old_qa_val = [r for r in old_val if r.get("kind") != "triage"]
    vignettes = sft_triage_rows()  # all hand-written, always kept (E5)

    gold_keys = {_key(g["user"]) for g in gold}
    triage_pool = build_triage_pool(labeled, gold_keys)

    # Carve a triage validation slice so val loss reflects the triage task (as v9 did).
    rng = random.Random(SEED)
    rng.shuffle(triage_pool)
    tri_val, tri_train = triage_pool[:150], triage_pool[150:]

    train = old_qa_train + vignettes * VIGN_OVERSAMPLE + tri_train
    val = old_qa_val + tri_val

    # (E2/E4) belt-and-braces: drop any row (incl. QA-reshaped) whose clinical case is a held-out gold.
    n0 = (len(train), len(val))
    train = [r for r in train if _key(_user(r)) not in gold_keys]
    val = [r for r in val if _key(_user(r)) not in gold_keys]
    leaked = (n0[0] - len(train), n0[1] - len(val))
    assert all(_key(_user(r)) not in gold_keys for r in train + val), "eval-gold leak remains"

    rng.shuffle(train)
    rng.shuffle(val)
    V10.mkdir(parents=True, exist_ok=True)
    dump(train, "sft_train.jsonl")
    dump(val, "sft_val.jsonl")
    # Copy the rest of the dataset unchanged so the folder is a complete, uploadable set.
    for name in ("dpo_train.jsonl", "dpo_val.jsonl", "triage_eval_gold.jsonl"):
        shutil.copy(V9 / name, V10 / name)
    if (V9 / "dataset-metadata.json").exists():
        shutil.copy(V9 / "dataset-metadata.json", V10 / "dataset-metadata.json")

    def urg(a: str) -> str:
        return next((lv for lv in URGENCY_LEVELS if lv in a), "?")

    def tri_urg(rows: list[dict]) -> dict:
        return dict(collections.Counter(
            urg(r["messages"][2]["content"]) for r in rows if r.get("kind") == "triage"))

    comp = lambda rows: dict(collections.Counter(r.get("source") for r in rows))  # noqa: E731
    lang = lambda rows: dict(collections.Counter(r.get("lang") for r in rows))  # noqa: E731
    kind = lambda rows: dict(collections.Counter(r.get("kind") for r in rows))  # noqa: E731
    return {
        "seed": SEED, "max_maximale_cap": MAX_MAXIMALE, "val_fraction_note": VAL_FRACTION,
        "train": {"n": len(train), "lang": lang(train), "kind": kind(train),
                  "source": comp(train), "triage_urgency": tri_urg(train)},
        "val": {"n": len(val), "lang": lang(val), "kind": kind(val),
                "source": comp(val), "triage_urgency": tri_urg(val)},
        "triage_pool_total": len(triage_pool),
        "eval_gold_leak_dropped": {"train": leaked[0], "val": leaked[1]},
    }


def main() -> None:
    rep = build()
    print(json.dumps(rep, indent=2, ensure_ascii=False))
    tri = rep["train"]["triage_urgency"]
    print("\nv10 TRAIN triage urgency vs v9 (1098/689/254):")
    for lv in URGENCY_LEVELS:
        print(f"  {lv:18s}: {tri.get(lv, 0)}")
    print("\nleak assert: PASSED (0 held-out eval-gold cases in train/val)")
    print(f"files written under {V10}:")
    for f in sorted(V10.glob("*")):
        print(f"  {f.name}")
    # sample v10 triage row (real justification)
    for r in load(V10 / "sft_train.jsonl"):
        if r.get("kind") == "triage" and r.get("source") == "llm_triage":
            a = r["messages"][2]["content"]
            if CANNED_FALLBACK_FR.get("urgence maximale", "___") not in a \
                    and CANNED_FALLBACK_FR.get("urgence modérée", "___") not in a \
                    and CANNED_FALLBACK_FR.get("urgence différée", "___") not in a \
                    and DISCLAIMER["fr"] in a:
                print("\nsample v10 triage row (real per-case justification):")
                print("  USER :", r["messages"][1]["content"][:120].replace("\n", " "), "...")
                print("  ASST :", a.replace("\n", " | "))
                break


if __name__ == "__main__":
    main()
