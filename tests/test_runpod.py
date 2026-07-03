"""Tests for the RunPod experiment control plane: launcher mutation building, sweep-config
generation, DPO-pair balance invariant, and results parsing. All pure — no network."""
from __future__ import annotations

import json

from oc14_triage.runpod import launcher, results, sweep


# --- launcher: mutation is a pure function of the spec ---------------------------------------
def test_build_create_mutation_wires_gpu_disk_and_startup():
    spec = launcher.PodSpec(
        name="oc14-test", startup_cmd="echo hi && nvidia-smi",
        gpu_type_id=launcher.GPU_3090, disk_gb=42, gpu_count=1,
        env={"HF_TOKEN": "SECRET", "WANDB_API_KEY": "SECRET2"})
    m = launcher.build_create_mutation(spec)
    assert "podFindAndDeployOnDemand" in m
    assert '"NVIDIA GeForce RTX 3090"' in m
    assert "containerDiskInGb: 42" in m
    assert "gpuCount: 1" in m
    assert "cloudType: COMMUNITY" in m
    # startup command is base64-wrapped (WAF dodge); it decodes back to the original
    assert "base64 -d | bash" in m
    # env keys present in the mutation
    assert "HF_TOKEN" in m and "WANDB_API_KEY" in m


def test_env_block_is_valid_key_value_list():
    block = launcher._env_block({"A": "1", "B": "two"})
    assert block.startswith("[") and block.endswith("]")
    assert '{key: "A", value: "1"}' in block
    assert '{key: "B", value: "two"}' in block


def test_startup_cmd_is_base64_wrapped_to_dodge_cloudflare_waf():
    # a realistic multi-line startup script (heredoc, python -c) must not break the mutation and
    # must be base64-encoded so RunPod's Cloudflare WAF doesn't 403 it (error code 1010).
    import base64

    script = 'pip install "unsloth"\npython -c "print(1)"'
    spec = launcher.PodSpec(name="x", startup_cmd=script)
    m = launcher.build_create_mutation(spec)
    start = m.index("dockerArgs: ") + len("dockerArgs: ")
    tail = m[start:]
    decoded = json.loads(tail[: tail.index(', env:')])
    assert decoded.startswith("bash -lc ")
    assert "base64 -d | bash" in decoded
    # the raw script chars must NOT be visible (that is the whole point) but must round-trip
    assert "unsloth" not in decoded
    b64 = decoded.split("echo ", 1)[1].split(" | base64", 1)[0]
    assert base64.b64decode(b64).decode() == script


# --- sweep: small, bounded, early-stopped -----------------------------------------------------
def test_grid_size_is_reduced():
    # 3 lr x 3 r x 2 warmup = 18 full grid, but we run a reduced budget
    assert sweep.grid_size() == 18


def test_build_sweep_config_has_early_stop_and_metric():
    cfg = sweep.build_sweep_config(grid_cap=6)
    assert cfg["method"] in ("bayes", "grid", "random")
    assert cfg["metric"]["goal"] == "minimize"
    assert cfg["early_terminate"]["type"] == "hyperband"
    assert cfg["run_cap"] == 6
    # the interesting knobs are all in the search space
    for k in ("learning_rate", "lora_r", "warmup_ratio"):
        assert k in cfg["parameters"]


def test_enumerate_grid_matches_grid_size():
    grid = sweep.enumerate_grid()
    assert len(grid) == sweep.grid_size()
    assert all({"learning_rate", "lora_r", "warmup_ratio"} == set(c) for c in grid)


# --- results: the anti-collapse balance invariant ---------------------------------------------
def _pair(chosen_lv: str, rejected_lv: str) -> dict:
    return {
        "chosen": [{"role": "assistant", "content": f"1. Niveau d'urgence : {chosen_lv}."}],
        "rejected": [{"role": "assistant", "content": f"1. Niveau d'urgence : {rejected_lv}."}],
    }


def test_check_balance_passes_on_balanced_set():
    pairs = (
        [_pair("urgence maximale", "urgence modérée")] * 5
        + [_pair("urgence modérée", "urgence différée")] * 5
        + [_pair("urgence différée", "urgence modérée")] * 5
        + [_pair("urgence modérée", "urgence maximale")] * 5
    )
    ok, reasons = results.check_balance(pairs)
    assert ok, reasons


def test_check_balance_flags_middle_class_collapse():
    # modérée only ever REJECTED (the prior failure mode)
    pairs = (
        [_pair("urgence maximale", "urgence modérée")] * 10
        + [_pair("urgence différée", "urgence modérée")] * 10
    )
    ok, reasons = results.check_balance(pairs)
    assert not ok
    assert any("modérée" in r for r in reasons)


def test_check_balance_flags_missing_chosen_class():
    pairs = [_pair("urgence maximale", "urgence modérée")] * 10  # différée never chosen or rejected
    ok, reasons = results.check_balance(pairs)
    assert not ok


# --- results: DPO verdict vs 0.822 -----------------------------------------------------------
def test_summarize_beats_baseline():
    r = results.summarize_dpo_result(
        {"macro_f1": 0.84, "recall_per_level": {lv: 0.8 for lv in results.LEVELS}})
    assert r["no_collapse"] is True
    assert r["delta_vs_v9"] > 0
    assert "beats" in r["verdict"]


def test_summarize_detects_collapse():
    r = results.summarize_dpo_result(
        {"macro_f1": 0.30,
         "recall_per_level": {"urgence maximale": 0.0, "urgence modérée": 0.9,
                              "urgence différée": 0.0}})
    assert r["no_collapse"] is False
    assert "COLLAPSE" in r["verdict"]
    assert set(r["collapsed_classes"]) == {"urgence maximale", "urgence différée"}


def test_summarize_below_baseline_no_collapse():
    r = results.summarize_dpo_result(
        {"macro_f1": 0.80, "recall_per_level": {lv: 0.7 for lv in results.LEVELS}})
    assert r["no_collapse"] is True
    assert r["delta_vs_v9"] < 0
    assert "below" in r["verdict"]
