"""Sanity checks on the SHIPPED (kaggle_upload) datasets — the artifact actually trained on.

These would have caught the audit's leakage/quality bugs. Skip cleanly when data isn't present (CI).
"""
import json

import pytest

from oc14_triage.config import ROOT, SYSTEM_PROMPT

UPLOAD = ROOT / "data" / "kaggle_upload"
SFT_TRAIN = UPLOAD / "sft_train.jsonl"
EVAL_GOLD = UPLOAD / "triage_eval_gold.jsonl"
DPO_TRAIN = UPLOAD / "dpo_train.jsonl"


def _load(path):
    # split on "\n" only — NOT splitlines() (which also splits on U+2028/U+0085 inside medical text)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]


def _user(r):
    m = r["messages"]
    return m[1]["content"] if len(m) > 1 else ""


def _key(t):
    return " ".join((t or "").split()).lower()[:80]


@pytest.mark.skipif(not SFT_TRAIN.exists(), reason="retrain SFT data not built (run build_retrain_sft)")
def test_sft_rows_wellformed():
    rows = _load(SFT_TRAIN)
    assert len(rows) > 1000
    for r in rows[:200]:
        assert [m["role"] for m in r["messages"]] == ["system", "user", "assistant"]
        assert r["messages"][0]["content"] == SYSTEM_PROMPT[r["lang"]]
        assert r["kind"] in ("qa", "triage")


@pytest.mark.skipif(not SFT_TRAIN.exists(), reason="SFT data not built")
def test_sft_has_meaningful_triage_share():
    rows = _load(SFT_TRAIN)
    assert sum(r["kind"] == "triage" for r in rows) / len(rows) > 0.15, "triage must be central (§0b)"


@pytest.mark.skipif(not (SFT_TRAIN.exists() and EVAL_GOLD.exists()), reason="data not built")
def test_no_eval_gold_leak_into_train():  # guards audit E2/E4
    train, gold = _load(SFT_TRAIN), _load(EVAL_GOLD)
    gold_keys = {_key(g["user"]) for g in gold}
    leaked = [r for r in train if _key(_user(r)) in gold_keys]
    assert not leaked, f"{len(leaked)} held-out eval-gold cases leaked into training"


@pytest.mark.skipif(not SFT_TRAIN.exists(), reason="SFT data not built")
def test_no_unintended_duplicate_examples():
    # vignettes are intentionally oversampled; check no *other* exact (user, assistant) dups
    keys = [(_user(r), r["messages"][2]["content"]) for r in _load(SFT_TRAIN) if r.get("source") != "vignette"]
    assert len(keys) == len(set(keys)), "duplicate (user, assistant) rows in train"


@pytest.mark.skipif(not SFT_TRAIN.exists(), reason="SFT data not built")
def test_all_handwritten_vignettes_present():  # guards audit E5
    from oc14_triage.data.vignettes import sft_triage_rows
    train_users = {_user(r) for r in _load(SFT_TRAIN)}
    missing = {v["messages"][1]["content"] for v in sft_triage_rows()} - train_users
    assert not missing, f"{len(missing)} hand-written vignettes missing from train"


@pytest.mark.skipif(not DPO_TRAIN.exists(), reason="DPO data not built (run build_dpo)")
def test_dpo_rows_wellformed():
    rows = _load(DPO_TRAIN)
    assert len(rows) > 100
    for r in rows[:200]:
        assert r["prompt"][0]["role"] == "system"
        assert r["chosen"][0]["role"] == "assistant"
        assert r["rejected"][0]["role"] == "assistant"
        assert r["chosen"][0]["content"] != r["rejected"][0]["content"]
