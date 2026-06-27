# OC14 — Architecture & decisions (mentor-facing)

> **Status:** intermediary, as of **2026-06-19**. May be updated as work continues.
> **Purpose:** explain *what* was built and *why*, with the criterion behind each choice, so each
> step can be defended to the mentor. Terms are defined on first use. Numbers are the real,
> measured ones (sources in `DEVELOPMENT_JOURNAL.md`); deeper detail lives in `docs/research/00`–`09`.

## 1. Goal & framing
Build a **proof-of-concept medical-triage assistant** for a (fictional) French ED. The graded core is
**LLM specialisation** (SFT + LoRA + DPO + evaluation + deployment), not a RAG system. Per mentor
guidance the **central task is triage** — given a patient situation, output (1) an **urgency level**
(*urgence maximale / modérée / différée*), (2) a **clinical justification**, (3) a **recommendation**
with red-flag escalation — in **French or English**. We deliberately do **not** build RAG (noted as a
possible future add-on only).

## 2. End-to-end architecture
```
 Public datasets (HF)          ETL (CPU, P710)                  Train (Kaggle T4, free)
 MediQAl FR · MedQuAD EN  ─►  filter + ChatML wrap + triage  ─►  SFT + LoRA  ─►  DPO  ─►  merge once
 UltraMedical-Pref EN          reshape + safety pairs            (adapter)      (pref)   (16-bit)
        │                              │                                                   │
        ▼                              ▼                                                   ▼
  GDPR/provenance card      data/processed/*.jsonl                         Serve: vLLM on RunPod
  (no patient data)         (SFT 5k + DPO 1.5k)                            serverless + FastAPI wrapper
                                                                                  │
                              Eval (triage-first metrics) ◄───────────────────────┤
                              CI/CD (GitHub Actions: ruff + pytest)  ──────────────┘
```
**Why this shape:** it matches the brief's required techniques and keeps every stage on free/cheap,
reproducible infrastructure. Data prep is CPU-only (runs on the home server); only training and
serving need a GPU.

## 3. Model choice
- **What:** start from **Qwen3-1.7B-Base** (a small open model under Apache-2.0). "Base" = a raw
  next-token predictor with no instruction-following; we add that behaviour ourselves via fine-tuning.
- **Why Base:** the brief names it explicitly, and it gives a clean "we specialised a raw model" story.
- **Why also Instruct (comparison arm):** an *Instruct* model already has safety/refusal behaviour; for
  a clinical-safety POC that head-start matters. So we fine-tune **both** on the same data and compare;
  **Base is the primary** model we DPO, merge, serve and deploy, **Instruct is the comparison**.
- **Criterion:** honour the brief while showing the Base-vs-Instruct safety trade-off the mentor/reviewers value.
- **Empirical result (small eval, n=6):** **Base SFT 0.67 (2/2 emergencies) > Instruct SFT.** The template
  confound was tested and **confirmed real**: Instruct on its **native** template recovered from 0.33→**0.50**
  (format 0.67→0.83, emergencies 0→1/2) — so ~half the original gap was our forced ChatML. **But Base still
  leads even with the confound controlled** (0.67 vs 0.50, 2/2 vs 1/2). Net: the Base-over-Instruct claim is
  **weakly supported, not proven** (margin small, n=6) — the **n=500 `medical-triage-500` eval** is needed to
  settle it. **Served deliverable = Base SFT** (best measured). Full analysis in `DEVELOPMENT_JOURNAL.md`.

## 4. Fine-tuning
- **SFT (Supervised Fine-Tuning):** show the model good (instruction → response) examples so it learns
  our triage response format and persona. **LoRA (Low-Rank Adaptation)** trains a small add-on
  (~0.3% of weights) instead of all 1.7B — so it fits a free 16 GB T4. Config: **r=16, α=16**,
  4-bit, 2 epochs. **Result (Base): train_loss 0.845**, ~79 min on a T4.
- **DPO (Direct Preference Optimization):** show the model pairs of (better, worse) answers so it
  prefers the better one — a cheap alignment method with no separate reward model. We use it as
  **(a) a technique demonstration** and **(b) a safety lever** (escalate-vs-reassure pairs), not as a
  clinical-quality claim. Mix: ~hand-written bilingual safety pairs + filtered UltraMedical.
- **Ordering invariant:** DPO runs on the SFT model **with the LoRA adapter still attached**; the
  adapter is **merged into the base weights exactly once, after DPO**, never between the stages.
- **Measured outcome (2026-06-19):** DPO **regressed** the model (urgency 0.33 vs SFT 0.67; **missed
  both emergencies**; repetition/GPT-isms). Root cause: the built DPO set was ~99% UltraMedical (1,489
  vs 11 safety pairs), so DPO optimised GPT-4 verbosity, not triage. **→ We ship the SFT model**; DPO is
  reported as an honest negative result. Fix path: more hand-written safety pairs + a safety-weighted mix.
- **Tooling:** **Unsloth** (a training wrapper with custom GPU kernels — ~2× faster, less VRAM) on top
  of Hugging Face **TRL/PEFT**. Plain TRL+PEFT is the documented fallback.

## 5. Data strategy
**Sources & licences** (all public; no real patient records):

| Source | Lang | Licence | Role |
|---|---|---|---|
| ANR-MALADES/MediQAl | FR | CC-BY-4.0 | FR medical QA + (clinical-case rows →) triage |
| keivalya/MedQuad-MedicalQnADataset (MedQuAD) | EN | CC-BY-4.0 (per original) | EN medical QA |
| TsinghuaC3I/UltraMedical-Preference | EN | MIT | DPO preference pairs |
| qanastek FrenchMedMCQA | FR | Apache-2.0 | **disabled** (loader-script incompatible) |
| MIETIC / MIMIC triage | EN | PhysioNet credentialed | **excluded** (no-redistribution licence) |

**Construction recipe (built, verifiable):** SFT = **5,000 train / 556 val**, ≈ **80% FR / 20% EN**,
**28% triage** (1,418 rows). Triage rows = hand-written bilingual vignettes + MediQAl clinical-case
rows reshaped into the urgency→justification→recommendation structure. The remaining QA rows give the
model the medical knowledge it needs to justify a triage. DPO = **1,350 train / 150 val**.

**Honest caveat (must state to mentor):** urgency labels on the QA-derived triage rows are a documented
**heuristic** (red-flag keywords), not clinically validated. The **held-out eval vignettes are
hand-labelled by a different process**, so the triage metric is not circular.

**GDPR/RGPD:** sources are exam questions / public NIH text / synthetic data → **no personal data** →
out of GDPR scope (Recital 26). The MIETIC exclusion is documented as a maturity point. A Microsoft
Presidio verification pass is wired as a *hypothesis test* (expect ~0 PII; report actual) — pending.

## 6. Tool stack (why each)
| Tool | Why |
|---|---|
| Qwen3-1.7B (Base + Instruct) | Apache-2.0, small enough for a free T4, strong multilingual |
| Unsloth + TRL/PEFT | ~2× faster / less VRAM on the T4; standard SFT+DPO+LoRA APIs |
| Hugging Face Datasets | load + ETL the public corpora (CPU-only) |
| Microsoft Presidio | documented, reproducible GDPR PII verification |
| vLLM | fast inference server with an OpenAI-compatible API |
| FastAPI wrapper | force the safety system prompt + `enable_thinking=False` + audit log |
| RunPod serverless | cheapest GPU serving; scale-to-zero |
| Kaggle (free T4) | free training GPU; ~30 h/week, 12 h/session |
| GitHub Actions | CI (lint + tests); deploy step on the wrapper |
| Weights & Biases | experiment tracking / loss curves (reproducibility) |

## 7. Infrastructure & cost
- **Training:** free **Kaggle T4**. Measured: full SFT ~79 min (≈ **€0**; ~1.3 of the ~30 free GPU-h/week).
  Equivalent rented cost would be ~€0.3–0.7/h on an L4/A10 — reported for honesty.
- **Serving:** **RunPod serverless** (pay-per-second, scale-to-zero) — deployment cost to be measured
  (€/1k requests + idle). Cost is tracked as a **graded metric** (mentor).

## 8. Evaluation
- **Triage-first metrics** (`eval/metrics.py`): urgency agreement vs hand-labelled gold (+ per-level
  recall, esp. *urgence maximale* = safety-critical), red-flag escalation, disclaimer presence, format
  adherence, language match, no-`<think>`. **MCQA accuracy is a secondary "didn't lose knowledge" check.**
- **Inference config that matters (learned the hard way):** the small model must **stop on `<|im_end|>`**
  (its `eos` is `<|endoftext|>`) and use the **full trained system prompt**, or it degenerates into
  repetition. This is also exactly how we'll serve it via vLLM.
- **Quick SFT eval result (Base, 6 held-out vignettes, correct inference):** urgency accuracy **0.67
  (4/6)**, format **0.83**, disclaimer **0.83**; **both safety-critical `urgence maximale` cases caught
  (2/2)**; the 2 misses were low-urgency (over-cautious — safe direction); no degeneration. n=6 is a
  sanity check, not statistically meaningful (details in `DEVELOPMENT_JOURNAL.md`).

## 9. Key decisions (the genuine forks)
| Decision | Choice | Why |
|---|---|---|
| Model | Base (primary) + Instruct (comparison) | Honour brief + show safety trade-off |
| Urgency scale | 3-level FR | Brief wording; simplest to label/eval |
| Training framework | Unsloth (+TRL fallback) | Speed/VRAM on a free T4 |
| Bridge QA→triage | Reframe + reshape clinical cases + vignettes | No triage-labelled open data; keep it honest |
| DPO | Attempted → **SFT shipped** | DPO regressed on a ~99% UltraMedical set (missed emergencies); honest negative result, fix = more safety pairs |
| Serve | Merged 16-bit weights | Simplest single-model serving |
| Endpoint liveness | Serverless scale-to-zero | Cheapest; cold-start reported honestly |
| Experiment tracking | W&B | One-line, hosted curves, reproducibility |
| RAG | Excluded | Not the graded core; protects scope |
| Triage data licence | Exclude MIETIC/MIMIC | Credentialed no-redistribution licence |

## 10. Status
**Done:** dataset built + documented; **SFT (Base) trained + eval'd** (0.67, both emergencies caught) —
**served deliverable**, merged to 16-bit (`models/sft-base-merged-16bit/`); **DPO attempted but regressed**
(→ SFT shipped); **Instruct comparison arm trained + eval'd** (0.33 — Base wins, validates the brief);
eval harness, CI, repo + PR; all four creds (Kaggle/W&B/HF/GitLab) set. **Pending:** vLLM serving (RunPod
*or* Modal) + FastAPI wrapper on the merged SFT model; CI deploy step; Presidio pass; fuller eval; report;
**HF publish (deferred until user's full review)**. Optional later: safety-weighted DPO retry.
