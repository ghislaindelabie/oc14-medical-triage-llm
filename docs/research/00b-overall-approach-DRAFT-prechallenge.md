# CHSA Medical Triage Assistant — Overall Approach (2-Week POC)

> **Who this is for.** You, a solo learner, with two weeks, no local GPU, and free Kaggle/Colab training. This document ties together the eight research reports into one simple, executable plan. It defines terms on first use and explains *why* behind each choice so you can adapt it. Every recommendation cites the report it draws from: **[Model]**, **[Unsloth]**, **[TRL]**, **[Data]**, **[Serving]**, **[GDPR]**, **[Refs]**, **[Eval]**.

---

## 1. Executive summary

You will fine-tune a small open model (Qwen3-1.7B-Base, Apache-2.0) into a **bilingual French/English medical triage assistant** and ship it end-to-end. First, build a ~5,000-pair instruction dataset by template-wrapping existing public French and English medical Q&A datasets, plus a small "triage-flavored" slice and an off-the-shelf English preference set for alignment — all from data with no real patient information, so it is GDPR-compliant by construction **[Data]**. Run Microsoft Presidio over everything anyway, mostly to *document* a rigorous anonymization process (you will find near-zero PII, and that is the correct, expected result) **[GDPR]**. Train in two stages on a free Kaggle T4 GPU: supervised fine-tuning (SFT) with LoRA to teach the response format, then Direct Preference Optimization (DPO) to nudge the model toward safer, more complete answers **[Unsloth][TRL]**. Use **Unsloth** as the training framework because it is roughly 2× faster and uses ~70% less memory, which buys you comfortable headroom on the T4 **[Unsloth]**. Merge the trained LoRA adapter into the base weights and serve the merged model with **vLLM** behind RunPod's serverless GPU (scale-to-zero, pay-per-second), which gives you an OpenAI-compatible API for almost no idle cost **[Serving]**. Wrap it in a thin FastAPI layer only to inject the safety system prompt and log each request, and automate lint/test/build/deploy with a three-stage GitHub Actions pipeline **[Serving]**. Bake clinical safety into the system prompt *and* the DPO preference pairs — hard-escalate red-flag symptoms, always show a disclaimer, never diagnose or prescribe — and verify it with a ~100-case evaluation harness that checks MCQA accuracy, triage agreement, emergency recall, and latency **[Eval]**. The goal is a *working, honest, well-documented pipeline*, not state-of-the-art accuracy — frame the deliverable accordingly in the 20-page report.

---

## 2. The big picture

```
                          ┌─────────────────────────────────────────────────────┐
                          │  STAGE 0 — DATA (CPU only, ~1 day, Kaggle/Colab CPU)  │
                          └─────────────────────────────────────────────────────┘
   Public datasets (HF)            ETL: filter + template-wrap        Presidio pass
   ┌──────────────────┐            ┌────────────────────────┐         ┌──────────────┐
   │ MediQAl (FR)     │──┐         │ wrap in Qwen3 ChatML    │         │ analyze +    │
   │ FrenchMedMCQA(FR)│  ├────────▶│ system/user/assistant   │────────▶│ anonymize +  │
   │ MedQuAD (EN)     │  │         │ + triage-flavor slice   │         │ audit log    │
   │ UltraMed-Pref(EN)│──┘         └────────────────────────┘         └──────┬───────┘
   └──────────────────┘                                                       │
                                                                              ▼
        ┌─────────── DELIVERABLE 1: ~5,000 SFT pairs + ~3,000 DPO pairs (JSONL) + data sheet ───────────┐
        │                                                                                               │
        ▼                                                                                               ▼
┌───────────────────────────────────┐                                            ┌──────────────────────────────┐
│ STAGE 1 — SFT  (Kaggle T4, ~3-4h)  │                                            │  STAGE 2 — DPO (Kaggle T4,~1h)│
│ Unsloth FastLanguageModel (4-bit)  │                                            │  PatchDPOTrainer() + DPOTrainer│
│ + LoRA r=16 on Qwen3-1.7B-Base     │───── SFT LoRA adapter ────────────────────▶│  ref_model=None, beta=0.1     │
│ TRL SFTTrainer.train()             │                                            │  on SFT checkpoint            │
└───────────────────────────────────┘                                            └───────────────┬──────────────┘
                                                                                                  │
                                          DELIVERABLE 2: merged_16bit weights ◀───── merge LoRA ──┘
                                                   │  push to HuggingFace Hub
                                                   ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 3 — SERVE                                                                                         │
│  RunPod Serverless ── worker-vllm image ── pulls merged model from HF Hub ── OpenAI-compatible API      │
│  (optional) thin FastAPI wrapper: inject safety system prompt + structured request logging             │
│                                                          │                                              │
│                                       DELIVERABLE 3: live cloud demo endpoint                           │
└──────────────────────────────────────────────────────────┼─────────────────────────────────────────────┘
                                                            │
┌───────────────────────────────────────────────────────────▼─────────────────────────────────────────────┐
│ STAGE 4 — CI/CD (GitHub Actions)                                                                          │
│  push → lint (ruff) + unit tests (mock vLLM) → build+push Docker image (GHCR) → update RunPod endpoint    │
│                                                          │                                                │
│                                       DELIVERABLE 4: CI/CD pipeline                                       │
└──────────────────────────────────────────────────────────┼─────────────────────────────────────────────┘
                                                            │
                            ┌───────────────────────────────▼───────────────────────────────┐
                            │ EVAL HARNESS (~100 cases): MCQA acc, triage agreement,         │
                            │ emergency recall, disclaimer/escalation, p95 latency           │
                            └───────────────────────────────┬───────────────────────────────┘
                                                            ▼
                                       DELIVERABLE 5: 20-page technical report
```

**How the pieces connect, in one sentence each:**
- The **dataset** is the input to SFT and the source of held-out test cases for the eval harness.
- **SFT** teaches the model *how to respond* (format, persona, language); **DPO** teaches it *which response is better* (safer, more complete) — DPO always runs on the SFT checkpoint, never on the raw base model **[Model][TRL]**.
- **Merging** turns the tiny LoRA adapter + base model into one ordinary weights folder that vLLM can serve with zero special flags **[Serving][Refs]**.
- **vLLM on RunPod** turns those weights into an HTTP API; **FastAPI** (optional) adds the safety prompt and logging; **GitHub Actions** rebuilds and redeploys on every merge to main.
- The **eval harness** runs against the live endpoint and produces the numbers your **report** quotes.

---

## 3. Key decisions (with options + recommendation)

### Decision A — Training framework: Unsloth vs plain TRL+PEFT

**The choice.** Both produce the *same* trained model; the difference is speed, memory, and how easy it is to debug.

- **Option 1 — Unsloth + TRL.** Unsloth wraps Hugging Face Transformers/TRL with custom GPU kernels: ~2× faster, ~70% less VRAM on Qwen3 **[Unsloth]**. Drop-in `FastLanguageModel.from_pretrained` → `get_peft_model` → `SFTTrainer.train`. There is no Qwen3-1.7B-specific notebook, but the 4B/14B notebook works unchanged with the model name swapped **[Unsloth][Refs]**.
- **Option 2 — Plain TRL + PEFT.** Official Hugging Face libraries, maximum transparency and documentation, no custom-kernel dependency. ~2–4× slower, more VRAM, but errors are standard PyTorch and easy to debug. Switching to Unsloth later is near-drop-in **[TRL]**.

**Recommendation: Unsloth + TRL**, with plain TRL+PEFT as a documented fallback. **Criteria:** on a free T4 with a frequent-disconnect risk, *training speed and VRAM headroom directly reduce your risk of losing a run* — a 1.7B SFT finishes in ~1–2h on Unsloth vs longer on plain TRL **[Unsloth][Data]**. The one real cost is version fragility: Unsloth breaks after Transformers updates, so you **must copy the exact pins from the current official notebook** (e.g. `transformers==4.56.2`, `trl==0.22.2` as of mid-2026) rather than from memory **[Unsloth]**. Fallback trigger: if Unsloth fails to install/compile on the Kaggle T4 runtime after ~30 minutes of trying, switch to plain TRL+PEFT — it will work and the report narrative is unaffected.

### Decision B — Bridging the QA→triage data gap

**The problem (stated honestly).** None of the four core datasets has urgency labels or a triage-dialogue structure. The product brief asks for triage; the data is medical Q&A. This is a real gap, not a formatting nuisance **[Data]**.

- **Option 1 — Reframe as "bilingual medical QA assistant with triage-style guidance."** Keep datasets as-is; a system prompt makes the assistant *act* like a triage helper (acknowledge symptoms → ask one follow-up → give a preliminary urgency indication). Low effort, fully honest **[Data]**.
- **Option 2 — Synthesize a triage-flavored slice.** Take ~300 emergency-subject MediQAl rows + ~200 already-instruction-formatted rows from MIETIC (real ED triage, CC-BY-NC-SA, HF copy is public), and template them into a 3-part triage response with heuristic urgency labels. Medium effort; demonstrates real data-engineering skill (a graded deliverable) **[Data]**.
- **Option 3 — Train primarily on a real triage dataset (MIETIC / FedMML).** Highest fidelity but FedMML is tabular (needs heavy ETL) and MIETIC is English-only and non-commercial; building the whole pipeline on it is disproportionate for a POC **[Data]**.

**Recommendation: Option 1 as the frame + Option 2 as a ~500-row slice.** **Criteria:** the grader evaluates your *pipeline and data-engineering*, not your ER protocol. Reframing keeps you honest about what the model actually learned; the small synthesized slice gives genuine triage behavior to demonstrate and a legitimate data-engineering story — while you explicitly flag in the report that urgency labels are heuristic, not clinically validated **[Data][Eval]**.

### Decision C — Run DPO, or skip it?

- **Option 1 — Do DPO.** It is an explicit graded deliverable, it is cheap (~1h on the T4, 1 epoch, low learning rate), and it is the *correct place to encode safety behavior* (hard-escalation, disclaimers) so it survives even if the system prompt is bypassed **[TRL][Eval]**. Off-the-shelf preference data exists: a 3,000-row slice of UltraMedical-Preference (MIT license, GPT-4-scored chosen/rejected pairs) needs zero generation work **[Data]**.
- **Option 2 — Skip DPO, SFT only.** Simpler, one fewer thing to break. But you drop a required deliverable and lose the cleanest mechanism for baking in safety.

**Recommendation: Do DPO.** **Criteria:** it is required, it is genuinely cheap given an off-the-shelf preference set, and it is where safety alignment belongs. Add ~50–100 hand-written safety preference pairs (chosen = escalate + disclaimer; rejected = reassure-and-continue) to the UltraMedical slice — the Eval report notes even this small number measurably shifts behavior **[Eval]**. Minimum-viable fallback: if DPO destabilizes (reward margin not climbing, model degrades), ship the SFT-only model and document DPO as "attempted, with these observations" — a POC can honestly report a negative result.

### Decision D — Serve merged weights vs runtime LoRA

- **Option 1 — Merge LoRA into base, serve as ordinary 16-bit weights.** `save_pretrained_merged(..., "merged_16bit")` → a standard ~3.4 GB HuggingFace folder → `vllm serve` with no special flags **[Unsloth][Serving]**.
- **Option 2 — Keep the adapter separate, serve with vLLM `--enable-lora`.** Lets you hot-swap many adapters on one GPU — useful only if you serve multiple fine-tunes at once.

**Recommendation: Merge.** **Criteria:** simplicity and foolproofness for a single-model POC. Runtime LoRA adds configuration and failure modes for a benefit you do not need. Do **not** serve the raw 4-bit QLoRA checkpoint directly — that path is officially discouraged and fiddly **[Unsloth][Serving]**.

### Decision E — How "live" must the endpoint be?

- **Option 1 — RunPod serverless, scale-to-zero, short idle timeout (30–60s).** Near-zero idle cost; a cold start (~5–15s with model caching) on the first request after idle **[Serving]**.
- **Option 2 — RunPod serverless with a longer idle timeout (300–600s).** Stays warm longer (better demo feel) but burns money while idle (~$0.11 per 10-min idle window) **[Serving]**.
- **Option 3 — Always-on dedicated pod.** Best responsiveness, worst cost ($0.19–0.50+/hr even idle) — wrong for a student demo that is idle 95% of the time **[Serving]**.

**Recommendation: Serverless with a short idle timeout + model caching, and a documented "warm-up" call.** **Criteria:** cost dominates for a rarely-used demo. Accept the cold start, enable RunPod model caching to keep it to seconds, and in your README/report tell the grader to send one warm-up request (or hit it once yourself right before they review). This is the cheapest credible option and satisfies the "vLLM + Docker" deliverable that HF Inference Endpoints (which uses TGI, not vLLM) would not **[Serving]**.

### Decision F — Do you even need the FastAPI wrapper?

- **Option 1 — No wrapper.** vLLM's built-in server already gives OpenAI-compatible endpoints, API-key auth, streaming, and Prometheus metrics. Inject the system prompt in your demo client **[Serving]**.
- **Option 2 — Thin FastAPI wrapper.** Auto-injects the triage safety system prompt (so callers send only the patient message) and adds structured request logging for the GDPR audit trail **[Serving][Eval]**.

**Recommendation: Thin wrapper.** **Criteria:** the safety prompt must not be optional, and the audit log is part of the GDPR/traceability story the grader is looking for. Keep it tiny: one `/triage` POST that forwards to local vLLM, injects the system prompt, and logs the fixed audit schema. This also gives you real, mockable code to unit-test in CI **[Serving][Eval]**.

### Decision G — Experiment tracking: W&B vs MLflow vs none

- **Option 1 — Weights & Biases (W&B).** Free tier, one-line `report_to="wandb"` in TRL/Unsloth configs, hosted dashboards for loss curves and DPO reward margins — nice screenshots for the report **[Unsloth][TRL]**.
- **Option 2 — MLflow.** Open-source, self-hostable, but needs a tracking server or local store — more setup on an ephemeral Kaggle session.
- **Option 3 — None / TensorBoard / CSV logs.** Zero dependency; `report_to="none"` and read the printed loss.

**Recommendation: W&B.** **Criteria:** lowest setup on ephemeral notebooks and it directly produces the training/alignment plots your report needs (SFT loss, DPO `rewards/margins` and `rewards/accuracies`). It is the canonical tracker in the Unsloth/TRL examples, so it is well-trodden **[Unsloth][TRL]**. Minimum-viable fallback: `report_to="none"` and screenshot the printed metrics — acceptable if W&B auth is a hassle.

---

## 4. Recommended tool stack

| Tool | Role | Why chosen (criteria) |
|---|---|---|
| **Qwen3-1.7B-Base** | Base model | Apache-2.0 (commercial + medical OK), fits 4-bit on a T4, ships a ChatML template even on Base; starting from Base means *you* own the format with no conflicting alignment **[Model]** |
| **Unsloth + TRL** | SFT + DPO training | ~2× faster / ~70% less VRAM on the free T4; near-drop-in TRL fallback if it breaks **[Unsloth][TRL]** |
| **LoRA (PEFT), r=16** | Parameter-efficient fine-tuning | Trains ~0.3% of params; sufficient for 1.7B domain adaptation on ~5k samples; low overfitting risk **[TRL][Model]** |
| **DPO (TRL DPOTrainer)** | Preference alignment | Offline (fast on one GPU), no reward model, off-the-shelf preference data exists; correct place for safety **[TRL][Data][Eval]** |
| **Hugging Face Datasets** | Source data + ETL | All four core datasets are public; ETL is CPU-only, minutes to run **[Data]** |
| **Microsoft Presidio** | Anonymization + audit | Demonstrates a documented, reproducible GDPR process; bilingual via `fr_core_news_md` + `en_core_web_lg` **[GDPR]** |
| **vLLM (`vllm/vllm-openai`)** | Inference server | PagedAttention throughput + OpenAI-compatible API out of the box; pin ≥0.9.0 so `enable_thinking=False` works **[Serving][Model]** |
| **FastAPI (thin wrapper)** | Prompt injection + logging | Forces the safety system prompt; produces the GDPR audit log; gives testable code for CI **[Serving][Eval]** |
| **Docker (`vllm/vllm-openai` base)** | Packaging | Official image; do NOT bake the 3.4 GB weights into it — pull from HF Hub at start **[Serving]** |
| **RunPod Serverless + worker-vllm** | Hosting | Cheapest GPU, scale-to-zero, official vLLM template, confirmed Qwen3 support **[Serving][Refs]** |
| **GitHub Actions** | CI/CD | Free runner minutes; lint+test → build+push (GHCR) → deploy is proportionate to a POC **[Serving]** |
| **Weights & Biases** | Experiment tracking | One-line integration, hosted plots for the report **[Unsloth][TRL]** |
| **scikit-learn + numpy** | Eval harness | Computes accuracy, Cohen's κ, per-class recall, p95 latency from ~100 cases **[Eval]** |
| **Kaggle (T4, 30h/week)** | Free training GPU | No idle disconnect (vs Colab's ~90 min), reliable allocation, fast HF downloads **[Unsloth]** |

---

## 5. The triage-data strategy

**Target: ~5,000 SFT instruction-response pairs (≈80% FR / 20% EN) + ~3,000 DPO preference pairs**, all with no real patient data. The whole build is CPU-only and runs in 1–2 hours **[Data]**.

**Verified ground truth (checked against HuggingFace, June 2026):** MediQAl is public, CC-BY-4.0, with configs MCQU (17,017), MCQM (10,617), OEQ (4,969). This is the backbone of the French data.

**Step 1 — French SFT core (~4,000 rows) [Data]**
- **MediQAl OEQ** (4,969 open-ended FR rows): question→free-text answer maps directly to SFT. Filter `len(answer) > 50`; take ~3,500 for training.
- **MediQAl MCQU "Reasoning" subset:** convert MCQ→instruction, response = "La réponse est {lettre}. {answer_text}." Take ~500 (cap MCQ-style to avoid over-representation).
- **FrenchMedMCQA** (Apache-2.0, ~2,171 train): same MCQ→instruction template, single-answer rows; take ~1,800 (pharmacy vocabulary MediQAl lacks).

**Step 2 — Triage-flavor injection (~500 rows, FR+EN) [Data]** — *this is Decision B, Option 2.*
- ~300 MediQAl rows where `medical_subject == "Urgences"`, wrapped in a 3-part triage template: (1) one clarifying follow-up question, (2) urgency level (urgence maximale / modérée / différée), (3) recommended action. Urgency is a heuristic from subject + question type — **flag as heuristic, not clinical** in the report.
- ~200 rows from MIETIC (`jackf7499/MIMIC-IV-Ext_Triage_Instruction_Corpus`, CC-BY-NC-SA, HF copy public) — already instruction-formatted real ED triage, gives English grounding.

**Step 3 — English SFT (~1,000 rows) [Data]**
- **MedQuAD** (`keivalya/...`; treat as CC-BY-4.0 per the original NIH source, or use the better-documented `lavita/MedQuAD` if the grader wants an explicit card). Filter `qtype ∈ {symptoms, treatment, exams and tests, complications, prevention}` and `len(Answer) > 100`; stratified sample 1,000.

**Running total: ~5,500 rows → reserve ~500 for validation → ~5,000 SFT train.** Language mix lands ≈82% FR / 18% EN, satisfying "bilingual" with no machine translation needed **[Data]**.

**Step 4 — DPO preference set (~3,000 rows) [Data][Eval]**
- **UltraMedical-Preference** (MIT, ~110k train): sample 3,000 where `chosen_score ≥ 4.5` and `rejected_score ≤ 4.0` (meaningful gap); exclude `prompt_id` starting `"MedQuad"` to avoid SFT overlap; load only `train`/`validation` (the `test` split has a schema mismatch). Map `chosen[1].content`→`chosen`, `rejected[1].content`→`rejected`.
- **Add ~50–100 hand-written safety pairs:** for each red-flag scenario (chest pain, stroke, severe dyspnea, anaphylaxis, self-harm), chosen = hard-escalate + disclaimer; rejected = reassure-and-continue **[Eval]**.

**Step 5 — Anonymization + audit (GDPR deliverable) [GDPR]**
Run Presidio (bilingual `fr_core_news_md` + `en_core_web_lg`, `replace` operators, threshold 0.5, custom recognizers for French NIR/IPP) over every record. **Expect near-zero findings — that is the correct result**, because the sources are exam questions and public NIH text. Produce: a per-source provenance card (license, URL, PII status), a JSON audit log (source, SHA-256 of raw text, Presidio version, entities found, timestamp), an automated re-scan asserting 0 residual PII, and a 2% manual spot-check. This four-part story (legal basis → process applied → audit trail → QC) *is* what the GDPR grading criterion rewards.

**Eval/test sets (built once, ~2–3h) [Eval]:** ~50 held-out MedQA/FrenchMedMCQA MCQA items + ~30 hand-written FR triage vignettes with reference urgency labels + ~20 adversarial safety probes (red-flag escalation, disclaimer presence, diagnosis-elicitation). Written de novo / held out — no leakage, no PII.

**Critical chat-template rule across all of this [Model]:** wrap every example in the Qwen3 ChatML template with `enable_thinking=False` (exclude `<think>` blocks entirely — simpler, and 1.7B "thinking" is unreliable for triage), set `eos_token="<|im_end|>"`, compute loss on assistant tokens only, and add a system-prompt line "respond in the same language as the question" to curb language mixing.

---

## 6. Day-by-day plan over ~2 weeks

Assume ~half-time effort; each day lists a **checkpoint** and a **minimum-viable fallback (MVP)** so no single setback sinks a deliverable.

| Day | Focus | Checkpoint | Minimum-viable fallback |
|---|---|---|---|
| **1** | Repo + accounts. Create GitHub repo, RunPod + HF + W&B accounts/tokens, Kaggle Secrets. Skim the Unsloth Qwen3 notebook + the Kaggle Qwen3-1.7B community notebook **[Refs]**. | Empty repo with README, tokens stored as secrets, notebook opens on Kaggle. | — |
| **2** | Data ETL. Load the four datasets, filter, template-wrap to ChatML, build ~5,000 SFT + ~3,000 DPO JSONL. Verify MediQAl loads. | JSONL files exist; print 5 random samples per source; language mix ≈80/20. | Drop the triage-flavor slice; ship pure QA SFT (still ~5k). |
| **3** | GDPR pass. Run Presidio, write audit log + provenance cards + re-scan + 2% spot-check. Draft the report's RGPD section. | Audit JSONs saved; re-scan shows ~0 residual PII; 4 provenance cards. | Presidio on FR only + document EN sources as "already de-identified public text." |
| **4** | SFT smoke test. Unsloth load 4-bit Qwen3-1.7B, LoRA r=16, **train on 200 rows / `max_steps=30`** to confirm the pipeline + VRAM headroom. | Loss decreases; no OOM; sample generation is coherent. | Switch to plain TRL+PEFT if Unsloth won't install in ~30 min **[TRL]**. |
| **5** | Full SFT. Train ~5,000 rows, 3 epochs, `save_steps=50`, `report_to="wandb"`. Quick bilingual generation test. | SFT adapter saved; FR and EN prompts both answer in the right language. | 1–2 epochs if time/quota is tight; smaller train set. |
| **6** | DPO. `PatchDPOTrainer()`, DPOTrainer on the SFT checkpoint, `ref_model=None`, `beta=0.1`, 1 epoch, lr 5e-6. Watch `rewards/margins` climb. | DPO loss drops, margins positive and rising, accuracies ≥0.7. | Ship SFT-only; document DPO attempt honestly (Decision C MVP). |
| **7** | Merge + publish. `save_pretrained_merged(merged_16bit)`, `push_to_hub_merged`. Sanity-load with a local/Colab `vllm serve`. | Merged model on HF Hub loads in vLLM and answers. | Keep weights private on HF + a Google Drive backup. |
| **8** | Serve on RunPod. Create serverless endpoint via worker-vllm: `MODEL_NAME`, `MAX_MODEL_LEN=2048`, `DTYPE=bfloat16`, `HF_TOKEN` secret, model caching on, idle timeout 60s. | `curl`/OpenAI SDK gets a triage response from the live endpoint. | If worker-vllm misbehaves, fall back to a one-GPU RunPod pod running `vllm/vllm-openai` directly **[Serving]**. |
| **9** | FastAPI wrapper + Docker. `/triage` injects the safety system prompt + logs the audit schema; `FROM vllm/vllm-openai:<pinned>`, no weights baked. | Local Docker container serves `/triage`; logs show the audit fields. | Skip the wrapper, inject the prompt in the demo client (Decision F MVP). |
| **10** | CI/CD. GitHub Actions: ruff + pytest (mock vLLM) → build+push to GHCR → deploy step (community action or RunPod GitHub integration). | Green pipeline on a push; image in GHCR; endpoint updates. | Stop at build+push; do the RunPod redeploy manually and document it. |
| **11** | Eval harness. Run the ~100-case set against the live endpoint: MCQA accuracy, triage agreement + per-class emergency recall, disclaimer/escalation rate, regex danger check, p95 latency. | Metrics table produced; emergency recall and escalation specifically reported. | Run against a local vLLM if the endpoint is flaky; smaller case set. |
| **12** | Safety + base-vs-tuned comparison. Compare base Qwen3-1.7B vs your SFT+DPO model on the same eval set (shows fine-tuning *improved* things — the real POC goal). | Before/after numbers; safety thresholds (100% disclaimer, ≥90% emergency recall, 0% danger patterns) checked. | Report whatever you measured honestly, including failures. |
| **13** | Report. Write the 20 pages: data + GDPR, SFT/DPO method + curves, serving architecture + cost, CI/CD, eval + safety + JAMA framing (assistant value is in *guiding*, not raw capability **[Eval]**). | Draft covers all 5 deliverables. | Bullet-point appendices over polished prose if short on time. |
| **14** | Buffer + polish. Re-run the demo end-to-end, warm the endpoint, fix README, final report pass. | Everything runs from a clean checkout; demo works. | — |

**Hard checkpoints (do-or-pivot):** end of Day 4 (pipeline proven) and end of Day 8 (a *live* endpoint exists, even if rough). If either slips, invoke the MVP path immediately rather than chasing quality.

---

## 7. Risks & how we keep it simple

| Risk | Why it bites a POC | The decision that keeps it proportionate |
|---|---|---|
| **Data/triage gap** — the data is QA, the brief says triage | Could tempt you into building a real clinical triage corpus (weeks of work) | Reframe as "QA assistant with triage-style guidance" + a small heuristic triage slice; **state the limitation in writing** (Decision B) **[Data]** |
| **Free-GPU limits** — T4 VRAM, session caps, Colab idle disconnects, version breakage | A lost 3-hour run or a broken install can eat a day | Kaggle over Colab; Unsloth for headroom; `save_steps=50`; **copy version pins from the live official notebook**; Day-4 smoke test before the full run **[Unsloth][Model]** |
| **Scope creep** — GRPO, thinking mode, multi-adapter serving, custom kernels, large datasets | Each is a plausible-sounding rabbit hole with no POC payoff | Explicit "no" list: no GRPO (DPO suffices), no `<think>` blocks, no runtime LoRA (merge), no machine translation (data is already ~80% FR), subsample big datasets to ~5k **[TRL][Model][Data]** |
| **Clinical safety** — model gives dangerous advice or misses an emergency | The one place a POC can do real harm if demoed naively | Safety in the system prompt **and** DPO pairs (hard-escalate red flags, mandatory disclaimer, no diagnosis/prescription); eval enforces 100% disclaimer + ≥90% emergency recall + 0% danger patterns as CI gates **[Eval]** |
| **Cost overrun** — GPU left running | Idle serverless or a forgotten pod silently bills | Scale-to-zero, 60s idle timeout, model caching; warm only at demo time; ~$0.50–1.00 per 1,000 requests, ~$0 idle **[Serving]** |
| **Over-claiming results** — quoting accuracy as if it were production-grade | A 1.7B model won't hit GPT-4 numbers; pretending otherwise is dishonest and the grader will notice | Frame every metric as "POC, shows fine-tuning improved over base," cite the JAMA finding that the bottleneck is human-AI *collaboration*, not raw capability **[Eval]** |
| **GDPR misframing** — implying you scrubbed lots of PII | The sources have ~none; claiming heavy redaction is wrong | Document the *process* with near-zero findings as the correct, expected outcome; cite Recital 26 (anonymous data is out of scope) **[GDPR]** |

**The simplicity through-line:** every fork above resolves toward *the option with the fewest moving parts that still satisfies the deliverable*. That is the right default for a 2-week solo POC graded on a working end-to-end pipeline.

---

## 8. What I need from you

Please confirm or decide these before implementation starts — each changes a concrete step:

1. **Triage framing & urgency taxonomy.** OK to reframe as "bilingual medical QA assistant with triage-style guidance" (Decision B)? And which urgency scale should templates/eval use — the simple 3-level (maximale / modérée / différée) I've assumed, or a specific CHSA/CCMU/Manchester scale? This sets the prompt templates and the eval reference labels **[Data][Eval]**.
2. **Grading rubric specifics.** Does the rubric (a) require a formal DPIA or is a report RGPD section enough; (b) set a numeric pass threshold for triage agreement (I've assumed ≥70% / ≥90% emergency recall); (c) treat the DPO set as a *separate* GDPR deliverable or bundled with SFT? **[GDPR][Eval]**
3. **DPO go/no-go.** Confirm we run DPO with the UltraMedical slice + ~50–100 hand-written safety pairs (recommended), vs SFT-only fallback **[Data][Eval]**.
4. **Endpoint liveness expectation.** Is a scale-to-zero endpoint with a ~5–15s cold start (warmed at demo time) acceptable, or does the grader expect instant always-on responses? This is the single biggest cost lever **[Serving]**.
5. **License acceptability.** Confirm these are fine for an educational POC: Unsloth (Apache-2.0 core), MIETIC (CC-BY-NC-SA, non-commercial), and — if you want a native-French augmentation later — MedInjection-FR (CC-BY-NC-ND). I've avoided ND-blocked redistribution in the core plan **[Unsloth][Data][Refs]**.
6. **Accounts/credits ready?** HuggingFace token, RunPod account with a few dollars of credit, Kaggle phone-verified for GPU, W&B account — so Day 1 isn't spent on signups.

Once 1–4 are confirmed, the Day-1/Day-2 work can start immediately.
