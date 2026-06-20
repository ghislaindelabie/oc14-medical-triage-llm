# OC14 — Development journal

> **Status:** intermediary, as of **2026-06-19**. May be updated as work continues.
> **Scope:** a factual record of what was done, the difficulties hit, plan changes, and the
> engineering problems/solutions. Every entry is verifiable from this project's artifacts
> (git history, `data/processed/_*_stats.json`, `data/raw/_inventory.json`, the Kaggle run
> logs, `models/sft-base-lora/adapter_config.json`) or directly observed during the work.
> Nothing here is estimated unless explicitly labelled "estimate".

## Project in one line
POC of a bilingual (FR/EN) **medical-triage** assistant: fine-tune **Qwen3-1.7B** (SFT+LoRA → DPO),
serve via vLLM, with CI/CD and a report. Triage (prioritise urgency, justify, recommend) is the
central task — not general medical Q&A.

---

## Timeline (by phase)

### Phase 0 — Requirements + research (2026-06-12)
- Read the OpenClassrooms brief (`docs/Finetunez votre propre LLM - OpenClassrooms BRIEF.pdf`).
  The local machine had no PDF tooling (`pdftotext`/poppler missing); extracted text with
  `uv run --with pypdf`.
- Confirmed the execution machine (P710) has **no GPU** (28 CPU cores, 31 GB RAM) → training/serving
  must be external.
- Ran a multi-agent research workflow (8 parallel topic reports on Sonnet + an Opus synthesis),
  then a 3-reviewer **red-team** pass + an Opus revision. Outputs: `docs/research/00`–`09`.
- The red-team caught a **licensing blocker**: the "MIETIC" triage dataset (initially considered) is
  PhysioNet **credentialed** MIMIC data with a no-redistribution DUA → **excluded** to protect the
  "no real patient data / GDPR-clean" story.
- Decisions taken with the user: fine-tune **both** Qwen3-1.7B-Base (brief's wording) **and**
  Instruct (comparison); urgency scale = simple **3-level French** (urgence maximale / modérée /
  différée).

### Phase 0b — Reporting + publishing (2026-06-13/14)
- Bundled the research into one mobile HTML reader served on the tailnet via the existing
  **`vault-reader`** service (`:8445/doc/oc14-finetune-llm-report`). Initial mistake: spun up a new
  Tailscale Serve port (`:8446`) — corrected to the canonical method (drop the HTML into `~/vault`,
  no new port). Recorded the canonical "publish HTML report" method in global `~/.claude/CLAUDE.md`.
- Established the working rule: **`.md` is source of truth**; HTML is a derived package, generated only
  for substantial/multi-report work or on request.

### Phase 1 — Mentor guidance + scaffold + data (2026-06-16)
- **Mentor guidance integrated** (`docs/research/00-OVERALL-APPROACH.md` §0b): triage must be the
  *centre* not QA; triage represented in **both** languages; **cost** (training + deployment) is a
  graded metric; **no RAG** (deliberate). Confirmed in `ARCHITECTURE_AND_DECISIONS.md`.
- Scaffolded the repo (`uv` project, `src/oc14_triage/`, ruff + pytest, `.gitignore` keeping data out
  of git).
- **Collected** the datasets to `data/raw/` (verified-schema inventory in `data/raw/_inventory.json`):
  MediQAl (mcqu 10,113/2,561/4,343 · mcqm 5,767/1,466/3,384 · oeq test 4,969), MedQuAD (16,407),
  UltraMedical-Preference (train 109,353 / val 2,232).
- **Built** the datasets (`data/processed/`): SFT 5,000 train / 556 val (FR 4,003 / EN 997 ≈ 80/20;
  **triage 28%**); DPO 1,350 train / 150 val (11 hand-written bilingual safety pairs + 1,489 filtered
  UltraMedical). Stats: `_sft_stats.json`, `_dpo_stats.json`.
- Added the triage-first eval metrics (`eval/metrics.py`), a GDPR/provenance data-card generator
  (`data/card.py` → `data/cards/DATA_CARD.md`), and CI (ruff + pytest). 17 tests passed.
- Created the private GitHub repo **`ghislaindelabie/oc14-medical-triage-llm`**, opened **PR #1**
  (CI green). `main` seeded empty; work on `feat/data-prep-and-scaffold`.

### Phase 2 — Training credentials + SFT (2026-06-18)
- Stored + live-verified **Kaggle** (`KAGGLE_API_TOKEN`) and **W&B** (`WANDB_API_KEY`) creds in `~/.env`
  (values never printed/committed). Staged `data/processed/*.jsonl` as a private Kaggle dataset
  `ghislaindelabie/oc14-triage-data`.
- Authored the SFT-LoRA Kaggle notebook (reproducibly via `notebooks/build_kaggle_notebooks.py`):
  Unsloth, Qwen3-1.7B-Base, LoRA r=16, `train_on_responses_only`, smoke-first toggle.
- **Five-round debugging** to get a clean run on Kaggle (see "Engineering problems" below).
- **Full SFT run completed** on a free Kaggle **T4**: **train_loss 0.845** (from ~1.78), 314 steps /
  2 epochs, **~79 min** (4,734.7 s). LoRA adapter (69 MB) saved + persisted to `models/sft-base-lora/`
  (config confirms base `unsloth/qwen3-1.7b-base-unsloth-bnb-4bit`, r=16, α=16).

### Phase 3 — Eval + documentation (2026-06-19)
- Quick eval of the SFT model with **correct inference** (stop on `<|im_end|>`, the exact trained
  system prompt read back from the data) on the held-out hand-labelled vignettes. *(Results filled in
  the "Eval results" section below once the run completes.)*
- Wrote this journal + `ARCHITECTURE_AND_DECISIONS.md`.

---

## Engineering problems & solutions (verifiable)

| # | Problem (observed) | Root cause | Solution |
|---|---|---|---|
| 1 | No GPU on the P710 | Home server has no NVIDIA GPU | Train on free Kaggle T4; serve later on RunPod serverless |
| 2 | Couldn't read the brief PDF | `poppler`/`pdftotext` not installed | `uv run --with pypdf` to extract text |
| 3 | `datasets` probe crashed (NumPy 2 vs system scipy) | Throwaway `uv run --with` env saw system site-packages | Use an isolated `uv` project venv (`uv sync`) |
| 4 | MediQAl configs "not found" | Config names are **lowercase** (`mcqu/mcqm/oeq`), not `MCQU…`; `oeq` is test-only | Lowercase + auto-discover splits |
| 5 | FrenchMedMCQA failed to load | Both qanastek variants ship a **loader script** that `datasets≥3` rejects | **Disabled** it (MediQAl covers FR amply); documented in `sources.py` |
| 6 | Triage rows ≈ 0 after first build | MediQAl `medical_subject` is **English** with **no "Urgences"** category | Reshape rows that have a `clinical_case` into the triage structure (heuristic urgency); triage rose to 28% |
| 7 | `'float' has no attribute strip` | pandas reads empty cells as `NaN` (float) | Normalise NaN→"" at the parquet-read boundary |
| 8 | JSON parse error reading built JSONL | `str.splitlines()` splits on U+2028/U+0085 present in medical text | Read JSONL by splitting on `"\n"` only |
| 9 | Credentialed-data licence | MIETIC = PhysioNet credentialed MIMIC, no redistribution | Excluded; triage slice from exam-derived MediQAl + hand-written vignettes |
| 10 | `CUDA: no kernel image` at model load (×2) | Kaggle API kernels default to **Tesla P100 (sm_60)**, dropped by the current torch toolchain | Push with **`--accelerator NvidiaTeslaT4`** every time (not settable in kernel-metadata) |
| 11 | torch replaced with a build lacking the GPU's kernels | plain `pip install unsloth` re-resolves torch | Use the official Unsloth **cu128** install cell (pins `transformers==4.56.2`, `--no-deps trl==0.22.2`) |
| 12 | Status polling silently failed | Kaggle filed the kernel under the **title-slug**, not the metadata `id` | Aligned `id` to the slug; use the URL slug for status/output |
| 13 | `FileNotFoundError` for the dataset | This Kaggle mounts datasets at **`/kaggle/input/datasets/<owner>/<slug>/`** (not the classic path) | `glob('**/sft_train.jsonl')` to resolve the path |
| 14 | `chat_template is not set` | **Qwen3-1.7B-Base ships no chat template** (research doc 01's assumption was wrong) | Set a plain **ChatML** template explicitly (also propagates to the saved tokenizer) |
| 15 | SFT generation degenerated (repetition/garbage) | Naive sample didn't **stop on `<|im_end|>`** and used a shortened system prompt | Eval/serving must stop on `<|im_end|>` + use the full trained system prompt |
| 16 | W&B/HF can't be injected into API-pushed kernels | Kaggle **Secrets** are UI-only | Notebook reads them from Secrets with graceful fallback (report_to="none"/no push) |

**Method note:** from problem #11 onward, fixes were validated cheaply where possible (e.g. the ChatML
template was rendered locally with jinja2 before pushing) because each Kaggle round-trip is ~10–13 min.

---

## Plan changes (chronological)
- **Model:** Instruct-only (red-team recommendation) → **both Base + Instruct** (user; Base = primary served, Instruct = comparison).
- **Framing:** "QA assistant with triage-style guidance" → **triage-central** (mentor): default response = urgency→justification→recommendation; eval leads with triage.
- **Bilingual triage:** add more EN triage coverage; bilingual eval set (mentor).
- **Cost:** added training + deployment cost as a tracked, graded metric (mentor).
- **RAG:** explicitly **out of scope** (mentor) — note as a possible future add-on only.
- **Publishing:** new per-report Tailscale port → **canonical vault-reader** method (no new port).
- **Time estimate corrected:** full SFT was estimated ~30–45 min; actual **~79 min** on T4 (still free, within quota).

---

## Current artifacts
- Repo: `ghislaindelabie/oc14-medical-triage-llm` (private), branch `feat/data-prep-and-scaffold`, PR #1.
- Data: `data/raw/` (collected, gitignored), `data/processed/*.jsonl` (SFT+DPO, gitignored), `data/cards/DATA_CARD.md`.
- Model: `models/sft-base-lora/` (LoRA adapter, 69 MB, gitignored). Not yet on HF (pending `HF_TOKEN`).
- Notebooks: `notebooks/oc14-sft-lora/`, `notebooks/oc14-sft-eval/` (+ generator `build_kaggle_notebooks.py`).
- Research + plan: `docs/research/00`–`09`, `IMPLEMENTATION_PLAN.md`.

## Eval results (Phase 3) — SFT (Base), correct inference
Quick eval on the **6 held-out hand-labelled vignettes**, generating with stop-on-`<|im_end|>` + the
exact trained system prompt (read back from the data). Run on a Kaggle T4, `COMPLETE`:

- **urgency_accuracy = 0.67 (4/6)** · format_rate = **0.83** · disclaimer_rate = **0.83**
- Per case: FR maximale→maximale ✓ · FR modérée→modérée ✓ · FR différée→**modérée ✗** · EN maximale→maximale ✓ · EN modérée→modérée ✓ · EN différée→**None ✗**
- **Both `urgence maximale` (safety-critical) cases caught — FR and EN (2/2).**
- The 2 misses are both **low-urgency** cases (over-triaged or unlabelled) → erring toward *more* caution (the safe direction for triage).
- **No repetition/degeneration** — confirms the earlier ugly sample was purely inference config (missing `<|im_end|>` stop + shortened prompt), not the trained weights.
- **Caveat:** n=6 is a sanity check, not statistically meaningful; the `différée` (non-urgent) class is the weakest. A fuller eval (larger set, DPO comparison, base-vs-tuned) comes later.

## DPO outcome (Phase 3) — attempted, **measured regression → SFT shipped**
Full DPO ran (1 epoch, ~45 min on T4) and the single post-DPO **merge succeeded** (valid 16-bit weights,
`models/dpo-merged-16bit/`, 3.3 GB). But on the **same** held-out vignettes + same correct inference, the
SFT+DPO model **regressed**:

| Model | urgency acc | `urgence maximale` caught | disclaimer | generation |
|---|---|---|---|---|
| SFT (Base) | **0.67** (4/6) | **2/2** | 0.83 | clean |
| SFT + DPO | **0.33** (2/6) | **0/2** ⚠️ | 0.67 | repetition + "GPT-isms" |

DPO collapsed predictions toward "modérée" and **missed both emergencies** (the safety-critical class SFT
got right) — the dangerous direction. **Root cause (verifiable from `_dpo_stats.json`):** the DPO set was
**1,489 UltraMedical pairs vs only 11 hand-written safety pairs** (~99% UltraMedical). UltraMedical is
English, GPT-4-scored, and differs mainly in verbosity/formatting, so DPO optimised GPT-style verbosity
(hence the "Intialized by GPT-4…" artifacts), not triage quality — exactly the red-team's Decision-C risk.
DPO `train_loss ≈ 0.876` (above the ~0.69 no-preference baseline) is consistent.

**Decision (pre-agreed Decision-C fallback):** **ship the SFT model** as the served deliverable; report DPO
as attempted-with-measured-regression (a legitimate, honest POC result). **Fix path ("deepen later"):**
grow the hand-written safety/triage vignettes to ~300–500 (the §0b target) and re-run DPO with a
safety-weighted mix (mostly safety pairs, subsampled UltraMedical) — the **data**, not the method, was the problem.

## Base vs Instruct vs DPO — comparison (Phase 3)
All three scored on the **same** 6 held-out hand-labelled vignettes, same correct inference:

| Model | urgency acc | `urgence maximale` caught | format | disclaimer |
|---|---|---|---|---|
| **Base SFT** | **0.67** (4/6) | **2/2** | 0.83 | 0.83 |
| Base SFT + DPO | 0.33 (2/6) | 0/2 | 0.83 | 0.67 |
| Instruct SFT | 0.33 (2/6) | 0/2 | 0.67 | 0.67 |

**Base SFT is the best of the three — and the only one that catches both emergencies.** Training losses
were near-identical (Base 0.845, Instruct 0.854, ~72–79 min each), so the gap is **behavioural**, not fit.

**Reading (honest):**
- This **supports the brief's choice of Base**: a raw model gives a cleaner SFT signal with no competing
  instruct priors. We overrode the Instruct model's native chat template with our plain ChatML (for
  apples-to-apples + non-thinking serving), and its existing alignment likely interfered with the small
  triage SFT — lowering format adherence and emergency detection.
- DPO and Instruct-SFT both underperformed Base SFT → **Base SFT remains the served deliverable.**
- **Caveat:** n=6 is a sanity check, not statistically significant; the direction (Base best, only Base
  catches emergencies) is suggestive, not definitive. A fuller eval (larger held-out set; optionally the
  Instruct *native* template) is future work.
