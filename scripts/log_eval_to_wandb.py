"""Log the OC14 triage eval results to Weights & Biases — one comparable run per model arm.

The experiment-tracking dashboard for the report. W&B can't auto-capture from the Kaggle eval kernels
without attaching WANDB_API_KEY as a Kaggle Secret (a per-notebook UI step), so this logs the recorded
eval metrics from P710 using the local key (personal entity ghislaindelabie). Each arm's numbers are the
verified `triage_report` output of its eval kernel (provenance in `config.kernel`). Re-run to refresh.

    uv run --with wandb python scripts/log_eval_to_wandb.py
"""
from __future__ import annotations

import wandb

PROJECT = "oc14-triage-eval"

ARMS = [
    {"name": "base-untrained", "tags": ["base"],
     "config": {"stage": "base", "kernel": "oc14-base-baseline-eval", "decoding": "greedy",
                "leak_free": True, "served": False},
     "metrics": {"macro_f1": 0.19, "accuracy": 0.25, "recall_maximale": 0.70, "recall_moderee": 0.05,
                 "recall_differee": 0.0, "format_rate": 0.68, "disclaimer_rate": 0.0, "no_think_rate": 1.0}},
    {"name": "sft-v8-INFLATED-retracted", "tags": ["sft", "retracted"],
     "config": {"stage": "sft-v8", "kernel": "oc14-sft-eval@v4", "decoding": "sampled(t=0.3)",
                "leak_free": False, "served": False, "note": "inflated by eval->train leak + sampling; retracted"},
     "metrics": {"macro_f1": 0.813, "macro_precision": 0.816, "macro_recall": 0.813,
                 "recall_maximale": 0.93, "cohen_kappa": 0.72}},
    {"name": "sft-v8-honest", "tags": ["sft"],
     "config": {"stage": "sft-v8", "kernel": "oc14-sft-eval", "decoding": "greedy", "leak_free": True,
                "served": False, "note": "leak-free + greedy; differee starved (E1 too aggressive)"},
     "metrics": {"macro_f1": 0.653, "macro_precision": 0.79, "macro_recall": 0.68, "recall_maximale": 0.91,
                 "recall_moderee": 0.85, "recall_differee": 0.28, "precision_maximale": 0.845,
                 "precision_moderee": 0.716, "precision_differee": 0.886, "format_rate": 1.0,
                 "disclaimer_rate": 1.0}},
    {"name": "sft-v9-SERVED", "tags": ["sft", "served"],
     "config": {"stage": "sft-v9", "kernel": "oc14-sft-eval", "decoding": "greedy", "leak_free": True,
                "served": True, "note": "differee-restored (relax E1 + vignettes x8); the deliverable"},
     "metrics": {"macro_f1": 0.822, "macro_precision": 0.84, "macro_recall": 0.82, "accuracy": 0.82,
                 "recall_maximale": 0.90, "recall_moderee": 0.85, "recall_differee": 0.71,
                 "precision_maximale": 0.88, "precision_moderee": 0.69, "precision_differee": 0.95,
                 "cohen_kappa": 0.73, "recall_maximale_ci_low": 0.83, "recall_maximale_ci_high": 0.95,
                 "format_rate": 1.0, "disclaimer_rate": 1.0, "no_think_rate": 1.0}},
    {"name": "sft-v9+dpo", "tags": ["dpo"],
     "config": {"stage": "dpo", "kernel": "oc14-dpo-eval", "decoding": "greedy", "leak_free": True,
                "served": False, "note": "sharpened extremes, collapsed middle; not shipped"},
     "metrics": {"macro_f1": 0.799, "macro_precision": 0.814, "macro_recall": 0.81,
                 "recall_maximale": 0.92, "recall_moderee": 0.55, "recall_differee": 0.96,
                 "precision_maximale": 0.82, "precision_moderee": 0.83, "precision_differee": 0.79,
                 "cohen_kappa": 0.715, "format_rate": 1.0, "disclaimer_rate": 1.0, "no_think_rate": 1.0}},
]

for arm in ARMS:
    run = wandb.init(project=PROJECT, name=arm["name"], config=arm["config"],
                     tags=["n300", "stratified-gold", *arm["tags"]], reinit=True)
    run.log(arm["metrics"])
    run.summary.update(arm["metrics"])
    run.finish()

print(f"logged {len(ARMS)} arms to W&B project '{PROJECT}'")
