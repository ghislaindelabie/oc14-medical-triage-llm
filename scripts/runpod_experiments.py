"""CLI to run the OC14 RunPod GPU-pod experiments — the mutualized entry point behind both the DPO
retrain (with rpo_alpha anti-collapse) and the SFT+LoRA W&B hyperparameter sweep.

This project historically trained on Kaggle; this is the RunPod GPU-*pod* path (RunPod was
previously only serverless inference). One launcher, two jobs, each on its own GPU pod so they run
in parallel. Pods self-report status/results into a private HF dataset (no SSH needed); the monitor
tears every pod down when jobs finish OR on timeout — a GPU is never left running.

Startup scripts live next to this file under `scripts/runpod_jobs/`. Secrets
(RUNPOD_API_KEY, WANDB_API_KEY, HF_TOKEN) are read from the env and forwarded to the pods; their
values are never printed.

Usage:
    python scripts/runpod_experiments.py launch [dpo|sweep|both]
    python scripts/runpod_experiments.py monitor --minutes 35   # polls + auto-terminates
    python scripts/runpod_experiments.py cleanup                 # terminate ALL my pods now
    python scripts/runpod_experiments.py status
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from oc14_triage.runpod import launcher  # noqa: E402

JOBS_DIR = Path(__file__).resolve().parent / "runpod_jobs"
STATE = Path(os.environ.get("OC14_RUNPOD_STATE", "/tmp/oc14_runpod"))
RESULTS_REPO = "ghislaindelabie/oc14-runpod-results"

# (state-key, startup script, pod name). DPO first (longer); both on a 24 GB RTX 3090.
JOB_SPECS = {
    "dpo": ("dpo_startup.sh", "oc14-dpo-rpo"),
    "sweep": ("sweep_startup.sh", "oc14-sft-sweep"),
}


def _api_key() -> str:
    k = os.environ.get("RUNPOD_API_KEY")
    if not k:
        sys.exit("RUNPOD_API_KEY not set")
    return k


def cmd_launch(which: str) -> None:
    STATE.mkdir(parents=True, exist_ok=True)
    api_key = _api_key()
    env = launcher.secrets_env()
    if "HF_TOKEN" not in env:
        sys.exit("HF_TOKEN required (pods report results via HF)")
    print("secrets forwarded:", sorted(env))
    keys = list(JOB_SPECS) if which == "both" else [which]
    for key in keys:
        script, name = JOB_SPECS[key]
        spec = launcher.PodSpec(
            name=name, startup_cmd=(JOBS_DIR / script).read_text(),
            gpu_type_id=launcher.GPU_3090, disk_gb=40, env=env,
            cloud_type="COMMUNITY", cost_ceiling_per_hr=0.35)
        pod = launcher.create_pod(spec, api_key, gpu_fallbacks=[launcher.GPU_4090],
                                  try_all_cloud=False)
        (STATE / f"{key}_pod_id.txt").write_text(pod["id"])
        print(f"LAUNCHED {key}: pod={pod['id']} costPerHr={pod.get('costPerHr')}")


def _pod_ids() -> dict[str, str]:
    return {k: (STATE / f"{k}_pod_id.txt").read_text().strip()
            for k in JOB_SPECS if (STATE / f"{k}_pod_id.txt").exists()}


def cmd_monitor(minutes: float) -> None:
    from huggingface_hub import HfApi

    api_key = _api_key()
    api = HfApi(token=os.environ["HF_TOKEN"])
    pods = _pod_ids()
    terminal = {k: {f"{k}/results.json", f"{k}/status_error.json"} for k in pods}
    deadline = time.time() + minutes * 60
    done: set[str] = set()
    try:
        while time.time() < deadline and done != set(pods):
            try:
                files = set(api.list_repo_files(RESULTS_REPO, repo_type="dataset"))
            except Exception:  # noqa: BLE001
                files = set()
            for k in pods:
                if k not in done and terminal[k] & files:
                    print(f"{k} terminal", flush=True)
                    done.add(k)
            time.sleep(20)
    finally:
        for k, pid in pods.items():
            launcher.terminate_pod(pid, api_key)
            print(f"terminated {k} {pid}", flush=True)
        time.sleep(5)
        print("PODS REMAINING:", [p["id"] for p in launcher.list_my_pods(api_key)])


def cmd_cleanup() -> None:
    api_key = _api_key()
    pods = launcher.list_my_pods(api_key)
    for p in pods:
        launcher.terminate_pod(p["id"], api_key)
        print("terminated", p["id"], p["name"])
    print("done; remaining:", [p["id"] for p in launcher.list_my_pods(api_key)])


def cmd_status() -> None:
    for p in launcher.list_my_pods(_api_key()):
        print(p["id"], p["name"], p["desiredStatus"], p.get("costPerHr"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_launch = sub.add_parser("launch")
    p_launch.add_argument("which", choices=["dpo", "sweep", "both"], default="both", nargs="?")
    p_mon = sub.add_parser("monitor")
    p_mon.add_argument("--minutes", type=float, default=35)
    sub.add_parser("cleanup")
    sub.add_parser("status")
    args = ap.parse_args()
    if args.cmd == "launch":
        cmd_launch(args.which)
    elif args.cmd == "monitor":
        cmd_monitor(args.minutes)
    elif args.cmd == "cleanup":
        cmd_cleanup()
    elif args.cmd == "status":
        cmd_status()


if __name__ == "__main__":
    main()
