# OC14 — Implementation plan

Operational companion to `docs/research/00-OVERALL-APPROACH.md` (the *why*; read its §0 locked
decisions and §0b mentor integration first). This file is the *what / in what order / current
state*. Triage is the central task (not medical Q&A); `.md` is the source of truth.

---

## Current state (2026-06-29)

**Done since 2026-06-20 (LLM-consensus retrain supersedes the 2026-06-18→20 snapshot below):**
- ✅ **LLM-consensus triage labelling** (3 models, Fleiss κ≈0.67); 300-case stratified gold + LLM-labelled train.
- ✅ **Adversarial audit** (`docs/KNOWN_ISSUES.md`): leak-free, greedy, batched eval.
- ✅ **SFT v9 macro-F1 0.82 — SERVED deliverable.** Base baseline 0.19. DPO #2 0.80 — instructive negative, **NOT shipped**.
- ✅ **FastAPI `/triage` wrapper built + unit-tested**; Dockerfile + README in `src/oc14_triage/serving/`.
- ✅ **Repo PUBLIC** (default `main`, PR #2 squash-merged ~2026-06-27) — https://github.com/ghislaindelabie/oc14-medical-triage-llm.
- ✅ **W&B results-comparison dashboard** (5 arms, manual eval summaries — **NOT live curves**) — https://wandb.ai/ghislaindelabie/oc14-triage-eval.

**Done / scaffolded and runnable (CPU on P710, no credentials needed):**
- ✅ Repo scaffold: `uv` project, `src/oc14_triage`, ruff + pytest, `.gitignore` (data not committed).
- ✅ **Data collected**: MediQAl (FR, CC-BY-4.0 — mcqu/mcqm/oeq), MedQuAD (EN), UltraMedical-Preference
  (EN, MIT) downloaded to `data/raw/` with a verified-schema inventory (`data/raw/_inventory.json`).
  FrenchMedMCQA **disabled** (ships a loader script modern `datasets` rejects; MediQAl covers FR amply).
- ✅ **SFT dataset built** (shipped: `data/kaggle_upload/sft_{train,val}.jsonl`, via `scripts/build_retrain_sft.py`):
  **5,598 train / 562 val**, **82% FR**, **triage 2,041 / QA 3,557** (urgency→justification→recommendation);
  sources mediqal_mcqu 2,166 · llm_triage 1,953 · medquad 995 · mediqal_oeq 396 · vignette 88. *(The old
  v1 `data/processed/` 5,000/556 / 28%-triage set is retracted.)*
- ✅ **DPO dataset built** (shipped: `data/kaggle_upload/dpo_{train,val}.jsonl`, via `scripts/build_dpo_pairs.py`):
  **211 train / 24 val** = direction-balanced triage pairs, **no UltraMedical**. *(The old v1 1,350/150
  UltraMedical-heavy set is retracted.)*
- ✅ **Eval metrics** (`eval/metrics.py`): triage-first (urgency agreement + κ, red-flag escalation
  recall, disclaimer/format/no-`<think>`/language-match rates). Unit-tested.
- ✅ **GDPR/provenance card** generator (`data/card.py` → `data/cards/DATA_CARD.md`).
- ✅ **CI** (`.github/workflows/ci.yml`): ruff + pytest (data-dependent tests skip without data).
- ✅ **Credentials** (2026-06-18): Kaggle API token + W&B key stored in `~/.env`, both **verified live**. Data **staged** as a private Kaggle dataset `ghislaindelabie/oc14-triage-data`.
- ✅ **SFT-LoRA notebook VALIDATED on Kaggle T4** (`notebooks/oc14-sft-lora/`): 5-step smoke ran `COMPLETE` — data resolves, ChatML template applies, train-on-responses-only works, adapter saves, generation emits triage format (loss 1.78 @ step 5). Full run launched.
  - Kaggle landmines solved: push with `--accelerator NvidiaTeslaT4` (P100 default is too old); cu128 torch install; datasets mount at `/kaggle/input/datasets/<owner>/<slug>/` (glob to find); Qwen3-Base has **no** chat_template → set ChatML explicitly.

> ⚠️ The 2026-06-18→20 snapshot below (train_loss 0.845, DPO regressed 0.33 vs 0.67, Instruct 0.33 vs Base 0.67) is **superseded by the LLM-consensus retrain** (see "Done since 2026-06-20"). Kept for history.

- ✅ **Full SFT run (Base) COMPLETE** (2026-06-18): train_loss **0.845** (from ~1.78), 314 steps / 2 epochs, ~79 min on T4 (free). LoRA adapter (69 MB) saved + persisted at `models/sft-base-lora/` (gitignored; ChatML template propagated to the saved tokenizer). Push to HF pending `HF_TOKEN`.
  - ⚠️ **Inference config to fix before eval/serving:** generations must **stop on `<|im_end|>`** (eos is `<|endoftext|>`) or the small model runs past the answer into repetition; and eval must use the **full trained `SYSTEM_PROMPT_FR`** (a shortened prompt drifts the format). Both are the verified Decision-H items.

- ✅ **DPO run COMPLETE but REGRESSED** (2026-06-19): SFT+DPO scored 0.33 vs SFT 0.67 and **missed both emergencies**; root cause = DPO set was ~99% UltraMedical (1,489 vs 11 safety pairs) → optimised GPT verbosity. Merge succeeded (`models/dpo-merged-16bit/`). **Per Decision-C fallback, the SFT model is the served deliverable.** Fix path (later): more safety pairs + safety-weighted DPO mix.

- ✅ **SFT merged to 16-bit** (`models/sft-base-merged-16bit/`, 3.3 GB) — the vLLM-servable model.
- ✅ **Instruct comparison arm trained + eval'd** (2026-06-20): Instruct SFT **0.33** (missed both emergencies) vs **Base SFT 0.67** (both caught), near-identical train loss → **Base wins, validates the brief's Base choice**. Base SFT stays the deliverable.

- ✅ **Multi-LLM triage-labelling pipeline built + key-free tested** (`src/oc14_triage/labeling/`, `docs/TRIAGE_CRITERIA.md`): cited rubric → 3 LLMs label real MediQAl vignettes (3-level + ESI/call) → consensus + Fleiss κ + MCQU calibration; `label`/`build`/`calibrate` CLIs; 28 tests, ruff clean, mock end-to-end OK. **Waiting on:** 3 API keys + tier/size confirm → calibrate → label 3,075 → retrain SFT on LLM labels → eval gold (~300).

**Remaining work:**
- ⬜ **Live endpoint deploy** (needs RunPod/Modal key) — wrapper is built + unit-tested, only the live deploy is pending.
- ⬜ **W&B live training curves** (attach `WANDB_API_KEY` as a Kaggle Secret + re-run the SFT notebook).
- ⬜ **Presidio/GDPR verification pass** + audit log.
- ⬜ **Final ≤20-page report.**

---

## Deliverables → tasks

### 1. Dataset (bilingual, anonymised, GDPR-documented) — **mostly done**
- [x] Collect + verify schemas; [x] build SFT (triage-central) + DPO; [x] provenance/GDPR card.
- [ ] Presidio verification pass (`uv sync --extra anon`; FR `fr_core_news_md` + EN), audit log, append findings to the card.
- [ ] Grow EN triage vignettes to ~100–150 and keep the eval set bilingual (mentor §0b). *(needs review/clinical input)*
- [ ] (optional) Push the processed dataset to HF Hub. *(needs `HF_TOKEN`)*

### 2. Fine-tuned model (SFT+LoRA → DPO) — **done**
- [x] `notebooks/oc14-sft-lora/` (Unsloth, Qwen3-1.7B): apply chat template; **read `tokenizer.eos_token`**
  (it's `<|endoftext|>`; train to emit `<|im_end|>`), loss on assistant tokens only; `save_steps=50`; `set_seed(3407)`.
- [x] DPO on the **adapter-attached** SFT model (`ref_model=None`), then **merge once** — DPO #2 = 0.80, instructive negative, NOT shipped.
- [x] Run for **Base (primary)**; SFT v9 (LLM-consensus labels) = macro-F1 **0.82**, the SERVED deliverable.
- [x] Offline verify before push.

### 3. Cloud endpoint (vLLM) — **wrapper built; live deploy pending key**
- [ ] RunPod/Modal serverless, stock `worker-vllm`, `--default-chat-template-kwargs '{"enable_thinking": false}'`, vLLM ≥0.9.0 — **pending RunPod/Modal key**.
- [x] Thin FastAPI `/triage` wrapper (`src/oc14_triage/serving/`): inject system prompt + `enable_thinking=False` + audit log — built + unit-tested (Dockerfile + README).

### 4. CI/CD — **partially done**
- [x] lint + test. [ ] build/push **wrapper** image to GHCR. [ ] documented manual RunPod redeploy (+ optional `workflow_dispatch` live smoke).

### 5. Report (≤20 pages) — **start as a running log Day 1**
- [ ] Method (data + GDPR incl. MIETIC exclusion), SFT/DPO curves, serving + **cost (training + deployment)**, CI/CD, eval + safety + Goh et al. framing.

---

## Re-baselined schedule (~7 working days; see research §6 for detail)

| Day | Focus |
|---|---|
| ~~0~~ | ✅ Scaffold + data collection + SFT/DPO build + eval/CI scaffold (this session). |
| 1 | Accounts/secrets; Presidio pass; grow EN triage vignettes + bilingual eval set; start `REPORT.md` log. |
| 2 | Kaggle: Unsloth install smoke + `pip freeze` lockfile; SFT-LoRA run (Base). |
| 3 | DPO (Base) + merge + offline vLLM verify + push; SFT (Instruct, comparison). |
| 4 | RunPod serverless endpoint live (hard gate); FastAPI wrapper. |
| 5 | CI deploy (wrapper image) + documented redeploy; eval harness vs endpoint; cost calc. |
| 6 | Base-vs-Instruct comparison; safety review of emergency probes. |
| 7 + buffer | Report editing; clean-checkout demo; polish. |

---

## Credentials

| Credential | For | Status |
|---|---|---|
| `HF_TOKEN` (write) | rate-limit-free downloads; push dataset + model | optional |
| Kaggle (phone-verified + `kaggle.json`) | free T4 training | ✅ obtained / used (SFT + eval ran) |
| `WANDB_API_KEY` | experiment tracking | ✅ obtained (used for the results-comparison dashboard; live training curves still pending a Kaggle Secret re-run) |
| RunPod / Modal API key | serverless vLLM endpoint | ⬜ still pending (only blocker for the live endpoint) |
| GitHub repo | CI/CD (gh authed as ghislaindelabie) | ✅ **public** `oc14-medical-triage-llm` (default `main`) |

## Decisions (updated 2026-06-29)
1. ✅ **Base = primary, Instruct = comparison** (confirmed). The served/deployed model is Base
   (SFT+LoRA → DPO → merge); Instruct gets the same SFT + eval as the comparison arm.
2. ⏸ Rubric specifics (formal DPIA vs report RGPD section, numeric pass thresholds, DPO graded
   separately) — **deferred**, to revisit before the report.
3. Accounts: GitHub repo created + pushed. HF / Kaggle / W&B / RunPod pending for training + deploy.
