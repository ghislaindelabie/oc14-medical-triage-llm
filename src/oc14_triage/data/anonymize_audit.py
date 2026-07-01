"""Dataset-level Presidio audit — the RGPD verification deliverable (research doc 06).

Runs the anonymiser over the shipped corpus's free-text and reports, per record, a one-way
SHA-256 (NOT the raw text) plus the PII entities Presidio found. Aggregated counts go in the
data card. We DO NOT predeclare "~0 PII": we run it and report the real numbers.

    uv run --extra anon python -m oc14_triage.data.anonymize_audit
"""

from __future__ import annotations

import collections
import json
from datetime import UTC, datetime

from ..anonymization import anonymize, sha256_text
from ..config import CARDS, KAGGLE_UPLOAD


def audit_records(records: list[dict], *, text_key: str = "text", lang_key: str = "lang") -> dict:
    """Scan records for PII → aggregate counts + per-record {sha256, entities}. No raw text kept."""
    by_type: collections.Counter[str] = collections.Counter()
    per_record, n_with, engine, version = [], 0, "", ""
    for rec in records:
        text = rec.get(text_key, "") or ""
        res = anonymize(text, mode="dataset", lang=rec.get(lang_key, "fr"))
        engine, version = res.engine, res.engine_version
        for e in res.entities:
            by_type[e["type"]] += e["count"]
        if res.entities:
            n_with += 1
        per_record.append({"sha256": sha256_text(text), "entities": res.entities})
    return {
        "n_records": len(records), "n_records_with_pii": n_with,
        "entities_by_type": dict(sorted(by_type.items())), "engine": engine,
        "engine_version": version, "generated_utc": datetime.now(UTC).isoformat(),
        "per_record": per_record,
    }


def _user_text(rec: dict) -> str:
    """Extract the patient-facing free-text (where PII would live) from a shipped record."""
    if "messages" in rec:  # SFT: the user turn
        for m in rec["messages"]:
            if m.get("role") == "user":
                return m.get("content", "")
        return ""
    if "prompt" in rec:  # DPO
        p = rec["prompt"]
        return p if isinstance(p, str) else (p[-1].get("content", "") if p else "")
    return rec.get("user", "")  # eval-gold


def _read(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().split("\n") if line.strip()]


def main() -> None:
    files = ["sft_train.jsonl", "sft_val.jsonl", "dpo_train.jsonl", "dpo_val.jsonl",
             "triage_eval_gold.jsonl"]
    records = []
    for fname in files:
        for rec in _read(KAGGLE_UPLOAD / fname):
            records.append({"text": _user_text(rec), "lang": rec.get("lang", "fr")})
    audit = audit_records(records)
    audit["files"] = files
    summary = {k: v for k, v in audit.items() if k != "per_record"}
    out = CARDS / "anonymization_audit.json"
    out.write_text(json.dumps(audit, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    print(f"\nWrote {out} ({audit['n_records']} records)")


if __name__ == "__main__":
    main()
