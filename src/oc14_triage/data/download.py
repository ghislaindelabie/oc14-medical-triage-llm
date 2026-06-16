"""Download the registered source datasets and record their real schema.

Runs CPU-only on the P710. Saves each (config, split) as parquet under
data/raw/<name>/ and writes data/raw/_inventory.json (row counts + columns + a
truncated example) — this is the plan's Day-1 "load-smoke" that turns assumed
schemas into verified facts before any ETL is written.

Usage:
    uv run python -m oc14_triage.data.download              # all sources
    uv run python -m oc14_triage.data.download --only mediqal frenchmedmcqa
"""

from __future__ import annotations

import argparse
import json

from datasets import get_dataset_config_names, get_dataset_split_names, load_dataset

from ..config import RAW
from .sources import SOURCES, SOURCES_BY_NAME, Source


def _discover_configs(hf_id: str) -> list[str | None]:
    try:
        names = get_dataset_config_names(hf_id)
    except Exception:  # noqa: BLE001
        return [None]
    return [None] if not names or names == ["default"] else list(names)


def _safe_split_names(hf_id: str, config: str | None) -> list[str]:
    try:
        return list(get_dataset_split_names(hf_id, config))
    except Exception:  # noqa: BLE001 — fall back to the usual trio; failures get recorded
        return ["train", "validation", "test"]


def _truncate(value: object, limit: int = 200) -> object:
    if isinstance(value, str) and len(value) > limit:
        return value[:limit] + "…"
    if isinstance(value, list):
        return [_truncate(v, limit) for v in value[:3]]
    return value


def _fetch(source: Source) -> list[dict]:
    """Download every (config, split) of one source; return inventory records."""
    records: list[dict] = []
    configs: list[str | None] = list(source.configs) or _discover_configs(source.hf_id)
    out_dir = RAW / source.name
    out_dir.mkdir(parents=True, exist_ok=True)
    for config in configs:
        splits = list(source.splits) or _safe_split_names(source.hf_id, config)
        for split in splits:
            tag = f"{config or 'default'}__{split}"
            try:
                ds = load_dataset(source.hf_id, config, split=split)
            except Exception as exc:  # noqa: BLE001 — record the failure, keep going
                msg = str(exc).splitlines()[0][:200]
                print(f"  FAIL {source.name} {tag}: {type(exc).__name__}: {msg}")
                records.append({"source": source.name, "config": config, "split": split,
                                "error": f"{type(exc).__name__}: {msg}"})
                continue
            path = out_dir / f"{tag}.parquet"
            ds.to_parquet(str(path))
            example = {k: _truncate(v) for k, v in ds[0].items()} if len(ds) else {}
            records.append({
                "source": source.name, "hf_id": source.hf_id, "config": config,
                "split": split, "language": source.language, "license": source.license,
                "role": source.role, "num_rows": len(ds), "columns": list(ds.column_names),
                "example": example, "path": str(path.relative_to(RAW.parent)),
            })
            print(f"  OK   {source.name} {tag}: {len(ds):>7,} rows  cols={ds.column_names}")
    return records


def main() -> None:
    ap = argparse.ArgumentParser(description="Download OC14 source datasets.")
    ap.add_argument("--only", nargs="*", help="subset of source names")
    args = ap.parse_args()
    todo = [SOURCES_BY_NAME[n] for n in args.only] if args.only else [s for s in SOURCES if s.enabled]
    RAW.mkdir(parents=True, exist_ok=True)
    inventory: list[dict] = []
    for source in todo:
        print(f"[{source.name}] {source.hf_id}  ({source.language}, {source.license})")
        inventory.extend(_fetch(source))
    (RAW / "_inventory.json").write_text(json.dumps(inventory, indent=2, ensure_ascii=False))
    ok = sum(1 for r in inventory if "error" not in r)
    print(f"\nInventory: {ok}/{len(inventory)} (config,split) pairs OK → {RAW/'_inventory.json'}")


if __name__ == "__main__":
    main()
