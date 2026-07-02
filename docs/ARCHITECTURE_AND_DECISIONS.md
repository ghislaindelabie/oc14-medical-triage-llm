# OC14 — Architecture & decisions (mentor-facing)

> **Status:** intermediary, as of **2026-06-29**. May be updated as work continues.
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
- **Empirical result (stratified n=300, greedy, leak-free):** Base (untrained) macro-F1 0.19 → SFT v9 0.82
  (served). The earlier n=6 Base-vs-Instruct comparison + template-confound analysis are in
  `DEVELOPMENT_JOURNAL.md` (now superseded as the headline).

## 4. Fine-tuning
- **SFT (Supervised Fine-Tuning):** show the model good (instruction → response) examples so it learns
  our triage response format and persona. **LoRA (Low-Rank Adaptation)** trains a small add-on
  (~0.3% of weights) instead of all 1.7B — so it fits a free 16 GB T4. Config: **r=16, α=16**,
  4-bit, 2 epochs. **Result (Base): train_loss ~0.869 on the LLM-labelled set** (the heuristic-labelled v1
  was 0.845), ~79 min on a T4.
- **DPO (Direct Preference Optimization):** show the model pairs of (better, worse) answers so it
  prefers the better one — a cheap alignment method with no separate reward model. We use it as
  **(a) a technique demonstration** and **(b) a safety lever** (escalate-vs-reassure pairs), not as a
  clinical-quality claim.
- **Ordering invariant:** DPO runs on the SFT model **with the LoRA adapter still attached**; the
  adapter is **merged into the base weights exactly once, after DPO**, never between the stages.
- **Data + outcome:** Shipped DPO = **direction-balanced triage-preference pairs** (211 train / 24 val:
  under 103 / mod 50 / over 48 / safety 10; UltraMedical removed). **Outcome (DPO #2):** macro-F1 0.80 vs
  SFT v9 0.82 — sharpened extremes (différée recall 0.71→0.96) but collapsed *modérée* (0.85→0.55). Honest
  negative; **SFT v9 shipped**. Mechanism: adjacent-level pairs make *modérée* the rejected level on both
  sides. (DPO #1 on a ~99% UltraMedical set also regressed — see journal.)
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

**Construction recipe (shipped, verifiable):** SFT = **5,598 train / 562 val** (fr 4,571 / en 1,027;
triage 2,041 / qa 3,557; sources llm_triage 1,953 · mediqal_mcqu 2,166 · mediqal_oeq 396 · medquad 995 ·
vignette 88). Triage rows = hand-written bilingual vignettes + MediQAl clinical-case rows reshaped into
the urgency→justification→recommendation structure. The remaining QA rows give the model the medical
knowledge it needs to justify a triage. DPO = **211 train / 24 val** direction-balanced
triage-preference pairs.

**Triage labelling (core architectural change):** Triage labels = **3-LLM consensus** (GPT-5.4 +
Mistral-Medium-3.5 + Sonnet-4.6, Fleiss κ≈0.67), replacing the v1 red-flag heuristic; eval-gold
(n=300, stratified 100/100/100) is held-out consensus, disjoint from train — so the triage metric is
not circular.

**Honest caveat (must state to mentor):** the consensus is a **silver standard** (LLM-as-annotator),
not clinical validation; source cases are real French *exam* vignettes (good provenance, exam-style).
See `DEVELOPMENT_JOURNAL.md` for the full limitations discussion.

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
- **SFT v9 eval result (stratified n=300, greedy, leak-free):** **macro-F1 0.82**; per-class recall
  *urgence maximale* **0.90** [CI 0.83, 0.95] · *modérée* 0.85 · *différée* 0.71; format/disclaimer
  **1.00**. The naive **Base (no fine-tuning)** floor on the same gold/harness is **macro-F1 0.19** — so
  fine-tuning bought 0.19 → 0.82 and taught the format + disclaimer from scratch. The earlier n=6 sanity
  number (0.67) is superseded (details in `DEVELOPMENT_JOURNAL.md`).

## 9. Key decisions (the genuine forks)
| Decision | Choice | Why |
|---|---|---|
| Model | Base (primary) + Instruct (comparison) | Honour brief + show safety trade-off |
| Urgency scale | 3-level FR | Brief wording; simplest to label/eval |
| Training framework | Unsloth (+TRL fallback) | Speed/VRAM on a free T4 |
| Bridge QA→triage | Reframe + reshape clinical cases + vignettes | No triage-labelled open data; keep it honest |
| DPO | Attempted twice → **SFT v9 shipped** | DPO #1 regressed (UltraMedical-heavy); DPO #2 (direction-balanced) collapsed the middle class (0.80<0.82) — honest negative |
| Serve | Merged 16-bit weights | Simplest single-model serving |
| Endpoint liveness | Serverless scale-to-zero | Cheapest; cold-start reported honestly |
| Experiment tracking | W&B | One-line, hosted curves, reproducibility |
| RAG | Excluded | Not the graded core; protects scope |
| Triage data licence | Exclude MIETIC/MIMIC | Credentialed no-redistribution licence |

## 10. Status
**Done:** 3-LLM consensus triage labelling (κ≈0.67) replacing the heuristic; adversarial audit fixes
(leak-free, consensus-clean eval); **SFT v9 trained + eval'd** (stratified n=300, macro-F1 **0.82**) — the
**served deliverable**; **naive-Base baseline** (macro-F1 0.19 — the honest progress floor); **DPO #2**
attempted (direction-balanced — macro-F1 0.80, analysed, **not shipped**); FastAPI `/triage` **serving
wrapper built + unit-tested** (mocked vLLM) with Dockerfile + README; repo made **public** (PR #2
squash-merged to `main`); **W&B results-comparison dashboard** (5 manually-logged eval summaries).
**Pending:** live vLLM endpoint (RunPod *or* Modal — needs a credential); W&B **live training curves**
(needs a `WANDB_API_KEY` Kaggle Secret + re-run); Presidio/GDPR pass; final report.
