"""Sanity checks on the built datasets. Skip cleanly when data isn't present (CI has no data)."""
import json

import pytest

from oc14_triage.config import PROCESSED, SYSTEM_PROMPT

SFT_TRAIN = PROCESSED / "sft_train.jsonl"
DPO_TRAIN = PROCESSED / "dpo_train.jsonl"


def _load(path):
    # split on "\n" only — NOT splitlines(), which also splits on U+2028/U+0085 that
    # legitimately appear inside medical text values and would break a JSON line.
    text = path.read_text(encoding="utf-8")
    return [json.loads(line) for line in text.split("\n") if line.strip()]


@pytest.mark.skipif(not SFT_TRAIN.exists(), reason="SFT data not built (run build_sft)")
def test_sft_rows_wellformed():
    rows = _load(SFT_TRAIN)
    assert len(rows) > 1000
    for r in rows[:200]:
        roles = [m["role"] for m in r["messages"]]
        assert roles == ["system", "user", "assistant"]
        assert r["messages"][0]["content"] == SYSTEM_PROMPT[r["lang"]]
        assert r["kind"] in ("qa", "triage")


@pytest.mark.skipif(not SFT_TRAIN.exists(), reason="SFT data not built")
def test_sft_has_meaningful_triage_share():
    rows = _load(SFT_TRAIN)
    triage = sum(r["kind"] == "triage" for r in rows)
    assert triage / len(rows) > 0.15, "triage must be central, not a garnish (mentor §0b)"


@pytest.mark.skipif(not DPO_TRAIN.exists(), reason="DPO data not built (run build_dpo)")
def test_dpo_rows_wellformed():
    rows = _load(DPO_TRAIN)
    assert len(rows) > 100
    for r in rows[:200]:
        assert r["prompt"][0]["role"] == "system"
        assert r["chosen"][0]["role"] == "assistant"
        assert r["rejected"][0]["role"] == "assistant"
        assert r["chosen"][0]["content"] != r["rejected"][0]["content"]
