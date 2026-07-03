"""W&B sweep configuration for the SFT+LoRA hyperparameter search — pure config builders.

The deliverable is a *small, fast, principled* sweep dashboard, not SOTA: a reduced grid over a few
interesting SFT knobs, each config a SHORT proxy run (few steps / data subset), with Hyperband
early-stopping so weak configs die quickly. Everything here is a pure function returning a plain
dict so we can unit-test the grid size, the search space and the early-stop policy without W&B.
"""
from __future__ import annotations

from typing import Any

# The interesting knobs, deliberately few. learning_rate is the highest-leverage LoRA hyperparam;
# lora_r/lora_alpha trade capacity vs overfit; warmup_ratio matters on tiny runs.
SEARCH_SPACE: dict[str, dict[str, Any]] = {
    "learning_rate": {"values": [1e-4, 2e-4, 3e-4]},
    "lora_r": {"values": [8, 16, 32]},
    "warmup_ratio": {"values": [0.03, 0.1]},
}

# Held fixed so runs stay comparable and short.
FIXED = {
    "model": "Qwen/Qwen3-1.7B-Base",
    "max_seq_len": 2048,
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 8,
    "max_steps": 60,          # short proxy run — enough to separate configs on val loss
    "train_subset": 4000,     # data subset to keep each run to a few minutes
    "seed": 3407,
    "metric": "eval_loss",
}


def build_sweep_config(program: str = "sweep_train.py", grid_cap: int = 6) -> dict:
    """Return a W&B sweep config dict.

    Uses `method: bayes` over the small space with Hyperband early-termination so the sweep agent
    both explores efficiently and kills weak configs early. `grid_cap` documents the intended
    run budget for the agent (a few configs), keeping wall-clock and spend bounded.
    """
    return {
        "program": program,
        "method": "bayes",
        "metric": {"name": FIXED["metric"], "goal": "minimize"},
        "parameters": {k: dict(v) for k, v in SEARCH_SPACE.items()},
        "early_terminate": {"type": "hyperband", "min_iter": 15, "eta": 2},
        # advisory: the agent is launched with `--count grid_cap` so it stops after a few configs.
        "run_cap": grid_cap,
    }


def grid_size() -> int:
    """Total number of distinct configs in the full grid (before early-stop / run_cap)."""
    n = 1
    for spec in SEARCH_SPACE.values():
        n *= len(spec["values"])
    return n


def enumerate_grid() -> list[dict]:
    """All (learning_rate, lora_r, warmup_ratio) combinations — used by a deterministic grid runner
    when we want reproducible coverage instead of the bayes agent."""
    import itertools

    keys = list(SEARCH_SPACE.keys())
    value_lists = [SEARCH_SPACE[k]["values"] for k in keys]
    return [dict(zip(keys, combo, strict=True)) for combo in itertools.product(*value_lists)]
