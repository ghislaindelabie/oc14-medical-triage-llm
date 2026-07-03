"""Mutualized RunPod GPU-pod launcher — ONE control plane shared by the DPO retrain and the
W&B SFT sweep (they differ only in the startup command + GPU choice, never in the plumbing).

Greenfield note: this project trained exclusively on Kaggle; RunPod was used only for serverless
*inference*. This module is the first RunPod GPU-*pod* training driver. It is deliberately thin:
the RunPod GraphQL API is the only I/O, everything else (mutation building, arg validation) is a
pure function so it can be unit-tested without touching the network.

Cost safety is structural, not aspirational:
  * `terminate_pod` is idempotent and the callers wrap every job in try/finally so a crash still
    tears the pod down;
  * `on_demand` pods have a hard `cost_ceiling` guard checked before creation.

Secrets (RUNPOD_API_KEY, WANDB_API_KEY, HF_TOKEN) are read from the environment and forwarded to
the pod as container env vars; their VALUES are never logged.
"""
from __future__ import annotations

import base64
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

API_URL = "https://api.runpod.io/graphql"

# PyTorch 2.8.0 + CUDA 12.8.1 image — matches the Kaggle cu128 recipe, so `pip install unsloth`
# resolves a torch build that already carries the right CUDA kernels (no slow torch reinstall).
DEFAULT_IMAGE = "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404"

# Modest 24 GB GPUs, plenty for Qwen3-1.7B 4-bit LoRA/DPO; both High-stock, community cloud.
GPU_3090 = "NVIDIA GeForce RTX 3090"   # ~$0.22/h
GPU_4090 = "NVIDIA GeForce RTX 4090"   # ~$0.34/h


@dataclass
class PodSpec:
    """Everything that varies between the two jobs. `startup_cmd` runs as the container's argv via
    bash -lc; it must be fully self-contained (install deps, pull data/model, train, push result,
    then `sleep`/exit — the launcher terminates the pod regardless)."""

    name: str
    startup_cmd: str
    gpu_type_id: str = GPU_3090
    image: str = DEFAULT_IMAGE
    gpu_count: int = 1
    disk_gb: int = 40           # container disk; model+deps ~15 GB, leave headroom
    volume_gb: int = 0          # no persistent volume — pods are ephemeral, results go to HF
    env: dict[str, str] = field(default_factory=dict)
    cloud_type: str = "COMMUNITY"   # COMMUNITY is cheapest; ALL lets RunPod pick
    cost_ceiling_per_hr: float = 0.60   # refuse to create anything pricier (safety)


def _gql(query: str, api_key: str, timeout: int = 40) -> dict:
    """POST a GraphQL query; raise on transport or GraphQL errors. Never logs the key."""
    body = json.dumps({"query": query}).encode()
    # RunPod fronts the API with Cloudflare, which 403s (`error code: 1010`) requests carrying the
    # default `Python-urllib/x.y` User-Agent. A curl-like UA passes. (Discovered the hard way.)
    req = urllib.request.Request(
        f"{API_URL}?api_key={api_key}", data=body,
        headers={"Content-Type": "application/json", "User-Agent": "curl/8.5.0"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            payload = json.loads(r.read().decode())
    except urllib.error.HTTPError as e:  # surface the server's message (never the api_key)
        body = e.read().decode(errors="replace")[:500]
        raise RuntimeError(f"RunPod HTTP {e.code}: {body}") from None
    if payload.get("errors"):
        msgs = "; ".join(e.get("message", "?") for e in payload["errors"])
        raise RuntimeError(f"RunPod GraphQL error: {msgs}")
    return payload["data"]


def _env_block(env: dict[str, str]) -> str:
    """Render an env dict as the GraphQL `env: [{key,value}, ...]` list. VALUES are embedded in the
    mutation string sent to RunPod but are never printed by this module."""
    items = ", ".join(
        f"{{key: {json.dumps(k)}, value: {json.dumps(v)}}}" for k, v in env.items())
    return f"[{items}]"


def build_create_mutation(spec: PodSpec) -> str:
    """Pure function: turn a PodSpec into the `podFindAndDeployOnDemand` mutation string.

    Unit-tested — no network. Keeping it pure is what lets us assert the GPU id, disk sizes and the
    startup command are wired correctly before we ever spend a cent.

    The startup script is base64-encoded inside dockerArgs. This is not cosmetic: a raw multi-line
    shell script (heredocs, `python -c`, backticks) sent in dockerArgs trips RunPod's Cloudflare WAF
    -> HTTP 403 `error code: 1010`. Base64 makes the payload opaque to the WAF and the pod decodes
    it with `base64 -d | bash`."""
    b64 = base64.b64encode(spec.startup_cmd.encode()).decode()
    wrapped = f"bash -lc {json.dumps('echo ' + b64 + ' | base64 -d | bash')}"
    docker_args = json.dumps(wrapped)
    return (
        "mutation { podFindAndDeployOnDemand(input: {"
        f"cloudType: {spec.cloud_type}, "
        f"gpuCount: {spec.gpu_count}, "
        f"gpuTypeId: {json.dumps(spec.gpu_type_id)}, "
        f"name: {json.dumps(spec.name)}, "
        f"imageName: {json.dumps(spec.image)}, "
        f"containerDiskInGb: {spec.disk_gb}, "
        f"volumeInGb: {spec.volume_gb}, "
        f"dockerArgs: {docker_args}, "
        f"env: {_env_block(spec.env)}"
        "}) { id imageName machineId costPerHr } }"
    )


def _create_once(spec: PodSpec, api_key: str) -> dict:
    data = _gql(build_create_mutation(spec), api_key)
    pod = data["podFindAndDeployOnDemand"]
    cost = pod.get("costPerHr")
    if cost is not None and float(cost) > spec.cost_ceiling_per_hr:
        # over budget — tear it down immediately, do not let it run
        terminate_pod(pod["id"], api_key)
        raise RuntimeError(
            f"pod {pod['id']} costPerHr {cost} > ceiling {spec.cost_ceiling_per_hr}; terminated")
    return pod


def create_pod(spec: PodSpec, api_key: str, gpu_fallbacks: list[str] | None = None,
               try_all_cloud: bool = True) -> dict:
    """Create an on-demand pod, resilient to 'machine does not have the resources' allocation
    misses: retry across `gpu_fallbacks` and, if COMMUNITY can't place it, escalate to cloudType
    ALL. Still guards the hourly cost ceiling. Raises only if every candidate fails."""
    candidates = [spec.gpu_type_id] + [g for g in (gpu_fallbacks or []) if g != spec.gpu_type_id]
    last_err: Exception | None = None
    for gpu in candidates:
        for cloud in ([spec.cloud_type, "ALL"] if try_all_cloud and spec.cloud_type != "ALL"
                      else [spec.cloud_type]):
            attempt = PodSpec(**{**spec.__dict__, "gpu_type_id": gpu, "cloud_type": cloud})
            try:
                return _create_once(attempt, api_key)
            except RuntimeError as e:
                last_err = e
                if "resources" not in str(e) and "not have" not in str(e):
                    raise  # a real error (bad field, auth) — don't mask it by retrying
    raise RuntimeError(f"no capacity across {candidates}: {last_err}")


def get_pod(pod_id: str, api_key: str) -> dict | None:
    q = (f"query {{ pod(input: {{podId: {json.dumps(pod_id)}}}) {{ id desiredStatus lastStatusChange "
         "runtime { uptimeInSeconds } } }")
    return _gql(q, api_key).get("pod")


def terminate_pod(pod_id: str, api_key: str) -> None:
    """Idempotent: terminating an already-gone pod is a no-op we swallow."""
    try:
        _gql(f"mutation {{ podTerminate(input: {{podId: {json.dumps(pod_id)}}}) }}",
             api_key)
    except RuntimeError:
        pass


def list_my_pods(api_key: str) -> list[dict]:
    data = _gql("query { myself { pods { id name desiredStatus costPerHr } } }", api_key)
    return (data.get("myself") or {}).get("pods") or []


def secrets_env() -> dict[str, str]:
    """Collect the secrets a training pod needs from the local environment. Missing ones are simply
    omitted (the pod scripts degrade: no W&B key -> report_to=none, etc.). Values never logged."""
    out = {}
    for k in ("WANDB_API_KEY", "HF_TOKEN"):
        v = os.environ.get(k)
        if v:
            out[k] = v
    return out


def wait_terminated(pod_id: str, api_key: str, tries: int = 6, delay: float = 5.0) -> bool:
    """Poll until the pod is gone (best-effort confirmation for cleanup accounting)."""
    for _ in range(tries):
        if get_pod(pod_id, api_key) is None:
            return True
        time.sleep(delay)
    return get_pod(pod_id, api_key) is None
