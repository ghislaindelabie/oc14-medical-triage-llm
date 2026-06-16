# CHSA Medical Triage Assistant — Overall Approach (2-Week POC) — Hardened v2

> **Who this is for.** You, a solo learner, with two weeks, no local GPU, and free Kaggle/Colab training. This document ties the eight research reports plus three red-team reviews into one simple, executable plan. It defines terms on first use and explains *why* behind each choice so you can adapt it. Recommendations cite the report they draw from: **[Model]**, **[Unsloth]**, **[TRL]**, **[Data]**, **[Serving]**, **[GDPR]**, **[Refs]**, **[Eval]**. Facts re-verified against primary sources in June 2026 are marked **[verified]**.
>
> **This v2 is the post-red-team version.** Three independent reviews (feasibility, data-clinical, technical) found one true blocker and a cluster of real bugs and over-claims. Every blocker and high-severity item is resolved below; see the new section **"What the red-team changed"** for the audit trail. The shape of the plan is unchanged — it was already well-scoped — but several recommendations and facts are corrected.

---

## 0. Decisions locked by Ghislain (2026-06-12)

These override the corresponding recommendations in the body below — read the body with these in mind:

1. **Model — DO BOTH, compare (not Instruct-only as §C2/Decision 2 recommends).** We fine-tune **Qwen3-1.7B-Base** (per the brief's explicit wording) *and* **Qwen3-1.7B Instruct**, and compare them in the report. **Proposed cost-control split (to confirm): Base is the PRIMARY deliverable** — the model we run DPO on, merge, serve on RunPod, and deploy; **Instruct is the COMPARISON arm** — same dataset, SFT + evaluation, with DPO only if GPU budget allows. The live endpoint serves one model (Base). This honours the brief, adds the safety-motivated Instruct comparison the red-team wanted, and bounds training to ~2–3 SFT runs.
2. **Urgency scale — simple 3-level French:** *urgence maximale / urgence modérée / urgence différée* (exactly the brief's wording). All triage templates and eval reference labels use this; no CCMU/ESI 5-level scale.
3. **MIETIC stays excluded** (credentialed-data blocker, §B/§5/§7) — pending only a final OK from Ghislain on the licence list (§9 Q6).

Still open for Ghislain after reading: rubric specifics (DPIA vs report section; numeric thresholds; DPO as separate deliverable), DPO go/no-go + mix, endpoint-liveness cost lever, accounts readiness (§9 Q3–Q7).

---

## 0b. Mentor guidance (2026-06-16) — integrated

The mentor reviewed the approach and mostly confirms it. Where it sharpens the plan, the points below **override the body** (same as §0):

1. **Triage is the CENTER — not general medical Q&A.** The known failure mode on this exact project is shipping a medical-QA bot wearing a triage label. So the bar moves from *"QA assistant with triage-style guidance"* (the body's Decision B framing) to *"a triage system that happens to know medicine."* Concretely: **(a)** the assistant's **default response format IS the triage structure** — *urgency level (maximale / modérée / différée) → clinical justification → adapted recommendation, with red-flag escalation* — applied to every symptom/clinical-case prompt, not free-form answers; **(b)** grow the **triage-structured SFT share from ~500 to ~1,200–1,500 rows** (MediQAl clinical-case + `Urgences` rows reshaped into that structure, plus hand-written bilingual vignettes), keeping the remaining QA pairs explicitly as *underlying medical knowledge the model needs to justify a triage decision*; **(c)** **evaluation LEADS with triage quality** — urgency agreement, justification present & clinically relevant, recommendation appropriate, red-flag escalation — and demotes MCQA to a "didn't lose medical knowledge" sanity check. *Refines Decision B, §5, §8.*

2. **Triage represented in BOTH FR and EN.** The business context is a French hospital (FR must be well covered), but triage cases must exist in **both** languages. Hand-write **~100–150 EN triage vignettes** (up from 30–50), make the **clinical-eval set bilingual** (FR + EN), keep DPO safety pairs ~half FR, and **document the FR/EN split per data slice** (not just overall). *Refines §5 + eval set.*

3. **Cost is a graded, first-class metric — TRAINING and DEPLOYMENT.** Track and report both: **training** = GPU-hours per run + cost (*"$0 on Kaggle free tier; ≈ $X if the same run were on a RunPod L4 at $Y/hr"* — honest equivalent), and **deployment** = RunPod serverless $/1k requests + idle cost. Log a cost line alongside W&B runs; add a cost subsection to the report. *New tool-stack line + a Day-10 task.*

4. **No RAG — deliberate.** The graded core is LLM specialization (SFT + LoRA + DPO + eval + deploy); fine-tuning is mandatory and central. We **exclude RAG on purpose** to protect scope, and note in the report that a retrieval layer (e.g. grounding answers in CHSA protocols) is a sensible *future* add-on, not part of this POC.

**Already aligned, mentor confirms (no change):** experiment tracking (W&B), reproducible metrics (seed `3407` + committed lockfile), evaluation scripts (the eval harness), full experiment traceability, infra freedom (Kaggle training + RunPod serving via the Alien account), and bilingual FR/EN with a documented split.

---

## 1. Executive summary

You will fine-tune a small open model into a **bilingual French/English medical assistant that gives triage-style guidance** and ship it end-to-end. First, build a ~5,000-pair instruction dataset by template-wrapping existing public French and English medical Q&A datasets, plus a small triage-flavored slice built **only from exam-derived sources** and a license-clean English preference set for alignment — all from data with **no real patient records**, so it is GDPR-out-of-scope by construction (GDPR Recital 26: truly anonymous/non-personal data falls outside the regulation) **[Data][GDPR]**. Run Microsoft Presidio over everything as a *verification step with a stated hypothesis* (we expect minimal PII because the sources are exam questions and public text — and we report whatever we actually find) **[GDPR]**.

Train in two stages on a free Kaggle T4 GPU: **supervised fine-tuning (SFT)** — teaching the model the response format and persona — with **LoRA** (Low-Rank Adaptation: train a tiny add-on instead of all weights), then **DPO (Direct Preference Optimization)** — a cheap alignment method that nudges the model toward "better" answers using chosen/rejected pairs, no reward model needed **[Unsloth][TRL]**. Use **Unsloth** (a training wrapper with custom GPU kernels) because it is roughly 2× faster and uses ~70% less GPU memory, buying headroom on the 16 GB T4 **[Unsloth]**. **Critical ordering invariant [verified]:** DPO runs on the SFT model *with the LoRA adapter still attached* (`ref_model=None` works by temporarily switching the adapter off to get the reference policy); you merge the adapter into the base weights **exactly once, after DPO**, never between the two stages.

Serve the single merged model with **vLLM** (a fast inference server with an OpenAI-compatible API) behind **RunPod serverless** (rent a GPU per-second, scale to zero when idle) **[Serving]**. Wrap it in a thin **FastAPI** layer to force the safety system prompt and write a structured request log, and automate lint/test/build with **GitHub Actions** **[Serving]**. Bake clinical safety into the system prompt *and* a bilingual, safety-weighted DPO slice — escalate red-flag symptoms, always show a disclaimer, never diagnose or prescribe — and verify it with a ~100-case evaluation harness whose emergency cases are **manually safety-reviewed**, not just regex-checked **[Eval]**. The goal is a *working, honest, well-documented pipeline*, not state-of-the-art accuracy — and the report says so plainly.

**The single biggest schedule change from v1:** the 14 "calendar days" are half-time, i.e. ~7 working days. We therefore (a) treat *"a live endpoint exists"* as a Day-8 → pulled-to-Day-6 hard gate, (b) start the report on Day 1 as a running build-log so the 20 pages become editing not writing, and (c) front-load the riskiest verifications (Unsloth install, dataset loads, template correctness) to Day 1–2.

---

## 2. The big picture

```
                          ┌─────────────────────────────────────────────────────┐
                          │  STAGE 0 — DATA (CPU only, ~1.5-2 days)               │
                          └─────────────────────────────────────────────────────┘
   Public EXAM/PUBLIC datasets (HF)     ETL: filter + template-wrap       Presidio (verify)
   ┌──────────────────────┐            ┌────────────────────────┐         ┌──────────────┐
   │ MediQAl (FR, CC-BY)  │──┐         │ wrap in Qwen3 ChatML    │         │ analyze +    │
   │ FrenchMedMCQA (FR)   │  ├────────▶│ system/user/assistant   │────────▶│ report ACTUAL│
   │ MedQuAD (EN, carded) │  │         │ + triage slice from     │         │ findings +   │
   │ UltraMed-Pref (EN)   │──┘         │   MediQAl "Urgences"    │         │ audit log    │
   │ (NO MIETIC/MIMIC)    │            │   + hand-written EN     │         └──────┬───────┘
   └──────────────────────┘            └────────────────────────┘                │
                                                                                  ▼
        ┌────── DELIVERABLE 1: ~5,000 SFT pairs + ~1,500 DPO pairs (JSONL) + data/provenance sheet ──────┐
        ▼                                                                                                 ▼
┌───────────────────────────────────┐                                          ┌────────────────────────────────┐
│ STAGE 1 — SFT  (Kaggle T4, 1-2 ep) │                                          │  STAGE 2 — DPO (Kaggle T4, ~1h)│
│ Unsloth FastLanguageModel (4-bit)  │                                          │  DPOTrainer on SFT model with  │
│ + LoRA r=16 on Qwen3-1.7B          │── adapter (DO NOT MERGE YET) ───────────▶│  ADAPTER STILL ATTACHED        │
│ TRL SFTTrainer.train()             │                                          │  ref_model=None, beta=0.1      │
│ push adapter to HF Hub @ save_steps│                                          │  bilingual safety-weighted set │
└───────────────────────────────────┘                                          └───────────────┬────────────────┘
                                                                                                │
                                       DELIVERABLE 2: merged_16bit weights  ◀──── MERGE ONCE ───┘
                                          (verify offline w/ vLLM LLM() in-notebook BEFORE push)
                                                   │  push to HuggingFace Hub (tagged revision)
                                                   ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 3 — SERVE                                                                                         │
│  RunPod Serverless ── stock worker-vllm image (NO custom image) ── env: MODEL_NAME=<HF repo> ──         │
│  ── --default-chat-template-kwargs '{"enable_thinking": false}' ── OpenAI-compatible API                │
│  thin FastAPI wrapper (separate, GHCR-built): injects safety prompt + enable_thinking=False + audit log │
│                                       DELIVERABLE 3: live cloud demo endpoint                           │
└──────────────────────────────────────────────────────────┼─────────────────────────────────────────────┘
                                                            │
┌───────────────────────────────────────────────────────────▼─────────────────────────────────────────────┐
│ STAGE 4 — CI/CD (GitHub Actions)  — on the FastAPI WRAPPER only                                           │
│  push → ruff lint + pytest (mocked vLLM) → build+push WRAPPER image to GHCR → [manual] RunPod redeploy     │
│  (optional, behind workflow_dispatch) post-deploy smoke: 1 request to live URL asserts HTTP 200 + text    │
│                                       DELIVERABLE 4: CI/CD pipeline                                       │
└──────────────────────────────────────────────────────────┼─────────────────────────────────────────────┘
                                                            │
                            ┌───────────────────────────────▼───────────────────────────────┐
                            │ EVAL HARNESS (~100 cases): behavioral metrics FIRST            │
                            │ (lang-match, disclaimer, escalation, format, no-<think>),      │
                            │ MCQA as a "didn't break knowledge" sanity check, p95 + COLD     │
                            │ latency reported SEPARATELY; every emergency probe hand-reviewed│
                            └───────────────────────────────┬───────────────────────────────┘
                                                            ▼
                                       DELIVERABLE 5: 20-page report (built incrementally from Day 1)
```

**How the pieces connect, one sentence each:**
- The **dataset** feeds SFT and supplies held-out test cases for eval; the eval emergency vignettes are hand-labelled **independently** of the training-time heuristic labels (no circularity).
- **SFT** teaches *how to respond* (format, persona, language); **DPO** teaches *which response is safer/better* — DPO runs on the SFT model with the adapter attached, never on the raw base model and never after merge **[TRL][verified]**.
- **Merging** (once, after DPO) turns adapter + base into one ordinary weights folder vLLM serves with no special flags **[Serving]**.
- **vLLM on RunPod** turns weights into an HTTP API; the **FastAPI wrapper** forces the safety prompt, disables thinking-mode, and logs; **GitHub Actions** lints/tests/builds the wrapper on every push.
- The **eval harness** produces the numbers the **report** quotes — framed honestly as a POC.

---

## 3. Key decisions (with options + recommendation)

### Decision A — Training framework: Unsloth vs plain TRL+PEFT

**The choice.** Both produce the *same* trained model; the difference is speed, memory, and debuggability.

- **Option 1 — Unsloth + TRL.** Custom GPU kernels: ~2× faster, ~70% less VRAM on Qwen3 **[Unsloth]**. The Qwen3 4B/14B notebook works with the model name swapped **[Refs]**.
- **Option 2 — Plain TRL + PEFT.** Official Hugging Face libraries, maximum transparency, no custom-kernel dependency; ~2–4× slower, more VRAM, but errors are standard PyTorch. Switching from one to the other is near-drop-in **[TRL]**.

**Recommendation: Unsloth + TRL**, with plain TRL+PEFT as a documented fallback. **Criteria:** on a free T4 with disconnect risk, speed and VRAM headroom directly reduce the chance of losing a run.

**Red-team correction (was a memory-fact error in v1):** do **not** copy hard version numbers from this document. The earlier draft cited `transformers==4.56.2, trl==0.22.2`; by mid-2026 the official Unsloth installer pulls `transformers`/`unsloth_zoo` from GitHub main, and `unsloth_zoo` carries an explicit *blocklist* of incompatible `transformers` versions — pasting a stale pin can land you inside a hole and break on import after you've spent quota. **What to actually do:** on Day 1, open the *current* official Unsloth Qwen3 notebook, run its exact install cell, then `pip freeze > requirements-train.lock.txt` and commit that lockfile — *that* is your reproducible pin. Add one assert that the installed `transformers` version is not in the blocklist. Install your explicit `transformers` pin *after* unsloth so it overrides Kaggle's preinstalled version. **Fallback trigger:** if Unsloth won't install/compile on the Kaggle T4 within ~30 min, switch to plain TRL+PEFT — report narrative unaffected. *(Sources: unsloth.ai/docs/get-started/install; github.com/unslothai/unsloth/issues/5237, /3211.)*

### Decision B — Bridging the QA→triage data gap

**The problem (stated honestly).** None of the core datasets has clinically-validated urgency labels or triage-dialogue structure. The brief asks for triage; the data is medical Q&A. This is a real gap **[Data]**.

- **Option 1 — Reframe as "bilingual medical QA assistant with triage-style guidance."** A system prompt makes it *act* like a triage helper (acknowledge symptoms → ask one follow-up → give a preliminary urgency indication). Low effort, fully honest **[Data]**.
- **Option 2 — Synthesize a small triage-flavored slice.** ~300–500 rows, template a 3-part triage response with heuristic urgency labels. Demonstrates real data-engineering (a graded skill) **[Data]**.
- **Option 3 — Train primarily on a real triage dataset (MIETIC/FedMML).** Highest fidelity but disproportionate ETL, and — see the red-team blocker below — license-incompatible.

**Recommendation: Option 1 as the frame + Option 2 as a ~300–500-row slice built ONLY from exam-derived sources.**

**Red-team BLOCKER resolved — MIETIC is removed entirely. [verified]** v1 sourced ~200 triage rows from `jackf7499/MIMIC-IV-Ext_Triage_Instruction_Corpus`, a Hugging Face re-upload self-labelled `cc-by-nc-sa-4.0`. The authoritative source is MIETIC on PhysioNet under the **PhysioNet Credentialed Health Data License 1.5.0**: it requires CITI/HIPAA credentialing, forbids sharing access with third parties, and forbids redistributing derivatives anywhere except PhysioNet under the same DUA. It is **de-identified real ED patient data from MIMIC-IV-ED.** The HF re-uploader had no right to relicense it. Baking it into a publicly-published dataset (Deliverable 1) and into pushed model weights (Deliverable 2) would (a) use credentialed data without credentialing and (b) violate the no-redistribution DUA — detonating the project's "no real patient data, GDPR-by-construction" narrative exactly where it claims to be strongest. *(Verified June 2026: physionet.org/content/mietic/1.0.0/, /view-dua/1.0.0/, /view-required-training/1.0.0/.)*

**Fix (turns a liability into evidence of maturity):** build the triage slice **only** from MediQAl `medical_subject == "Urgences"` rows (CC-BY-4.0, exam-derived, no patient data). For English triage grounding, **hand-write ~30–50 synthetic vignettes** (no real patients). Put this sentence in the report: *"We deliberately excluded MIMIC-derived triage corpora (e.g. MIETIC) because their PhysioNet credentialed license forbids redistribution and is incompatible with an openly-published POC dataset."* That single sentence is a GDPR-maturity win.

**And close the circularity (high-severity):** urgency labels in the *training* slice are heuristic (derived from subject + question type) and are honestly flagged as such. For the **eval** vignettes, assign urgency labels **by hand from the symptom description, by a different process than the training heuristic**, so train and test labels do not come from the same fiction. Report triage agreement only on those hand-labelled vignettes, with the caveat: *"urgency labels are author-assigned, not clinically validated; this measures behavioral consistency, not triage safety."*

### Decision C — Run DPO, or skip it?

- **Option 1 — Do DPO.** Required graded deliverable, cheap (~1h on the T4), and the *correct place to encode safety behavior* so it survives even if the system prompt is bypassed **[TRL][Eval]**.
- **Option 2 — Skip DPO, SFT only.** Simpler, but drops a required deliverable and the cleanest safety mechanism.

**Recommendation: Do DPO.** Keep SFT-only as a fallback if DPO destabilizes (reward margin not climbing, model degrades).

**Red-team correction — right-size the claim and rebalance the mix (high-severity).** UltraMedical-Preference is **English-only, GPT-4-scored** (its own card admits self-preference bias toward GPT-4 outputs), and its pairs differ mostly in completeness/formatting, not clinical correctness. So DPO on it does **not** teach "clinical quality" — at best it is a no-op on French, at worst it imports GPT-4 verbosity. **What changes:**
- DPO is framed in the report as **(1) a technique demonstration** (the graded deliverable) **and (2) a safety-behavior lever** — *not* as "improved clinical quality."
- **Rebalance the mix toward safety and bilingualism:** ~300–500 **hand-written bilingual safety pairs** (about half French) + a **subsampled ~1,000 rows** of UltraMedical (down from 3,000), for ~1,500 total. The safety pairs are now a *meaningful fraction*, not a 50-of-3000 garnish. Template them aggressively: one canonical "chosen" escalation skeleton (acknowledge → escalate → disclaimer → no diagnosis), slot in each red-flag symptom; "rejected" = reassure-and-continue. **Reuse the same red-flag scenarios across DPO pairs, eval vignettes, and adversarial probes** so the manual authoring is written once and amortized.
- Filter UltraMedical sensibly (`chosen_score ≥ 4.5`, `rejected_score ≤ 4.0`; exclude `prompt_id` starting `"MedQuad"` to avoid SFT overlap; load only `train`/`validation` — the `test` split has a schema mismatch). Map `chosen[1].content`→`chosen`, `rejected[1].content`→`rejected`.
- **Report DPO success as "shifted escalation/disclaimer behavior on the safety-probe set,"** measured before vs after DPO.

**Fallback:** if DPO destabilizes, ship SFT-only and document "DPO attempted, here are the observations" — a POC can honestly report a negative result.

### Decision C2 (NEW) — Base vs Instruct model

The red-team (data-clinical) argued the v1 choice of **Qwen3-1.7B-Base** trades a safety head-start for a "cleaner format narrative," the wrong trade for a *clinical-safety-graded* POC, because a Base model has no instruction-following, no refusal behavior, and no safety priors — every guardrail must be installed from scratch.

- **Option 1 — Qwen3-1.7B-Base.** "You own the format," but you build all safety behavior yourself from ~5k SFT + a small DPO set + system prompt.
- **Option 2 — Qwen3-1.7B (Instruct).** Still Apache-2.0, still fits the T4, and **inherits refusal/safety behavior** you would otherwise recreate; SFT adapts format/domain on top.

**Recommendation: switch the SFT base to Qwen3-1.7B (Instruct).** **Criteria:** for a two-week clinical-safety POC, a safety head-start is worth more than format purity, and the "I own the format" benefit is largely illusory — **[verified]** the Base model already ships the full ChatML chat_template (with `<think>` logic), so you do not actually get a blank-slate format either way. We are not over-acting on this: it is a one-line model-name change, costs nothing, and reduces the chance of a base-derived model emitting confident dangerous text. **Document the Base-vs-Instruct safety tradeoff in the report explicitly** — reviewers reward seeing it named. *(If you prefer Base for a learning narrative, that is defensible, but then over-weight safety examples in the SFT data and say so.)*

### Decision D — Serve merged weights vs runtime LoRA

- **Option 1 — Merge LoRA into base, serve as ordinary 16-bit weights.** `save_pretrained_merged(..., "merged_16bit")` → a standard ~3.4 GB folder → vLLM with no special flags **[Unsloth][Serving]**.
- **Option 2 — Keep the adapter separate, serve with `--enable-lora`.** Hot-swap many adapters — useful only when serving multiple fine-tunes at once.

**Recommendation: Merge (once, after DPO).** **Criteria:** simplicity for a single-model POC. Do not serve the raw 4-bit QLoRA checkpoint directly — officially discouraged.

**Red-team corrections (medium):** (1) The DPO ordering invariant above means **merge happens exactly once, after DPO**, never between SFT and DPO. (2) Unsloth's `save_pretrained_merged` has documented footguns: it re-downloads full FP16 base weights into a new `.cache` inside the output dir (disk + time blowup on Kaggle's ~20 GB `/kaggle/working`), and in some containers it finishes without error but writes **no files**. **Mitigation:** clear the HF cache before merging if disk is tight; after merge, assert the output dir contains `*.safetensors` + `config.json` + tokenizer files + `tokenizer_config.json` with the ChatML template, and total size ≈3–3.4 GB; then load it once with vLLM's **offline engine** `from vllm import LLM; LLM(path).generate(...)` *in the same notebook cell* — no server/port needed — to catch dtype/tokenizer/template breakage before pushing. Keep a private HF + Google Drive backup. *(Sources: github.com/unslothai/unsloth/issues/3633, /3882; docs.unsloth.ai/basics/running-and-saving-models/saving-to-vllm.)*

### Decision E — How "live" must the endpoint be?

- **Option 1 — Serverless, scale-to-zero, short idle timeout (60s).** Near-zero idle cost; a cold start on the first request after idle.
- **Option 2 — Serverless, long idle timeout (300–600s).** Warmer, but burns money idle (~$0.11 per 10-min idle window) **[Serving]**.
- **Option 3 — Always-on pod.** Instant, worst cost ($0.19–0.50+/hr even idle) — wrong for a demo idle 95% of the time **[Serving]**.

**Recommendation: serverless, scale-to-zero, 60s idle timeout + model caching/FlashBoot — PLUS set `min-workers=1` (or "active worker = 1") for the ~1-hour grading/demo window, then scale back to zero.**

**Red-team correction (two reviewers, high/medium) — the cold-start number in v1 was optimistic.** The "5–15s" figure is FlashBoot *warm revival*, which applies only under consistent traffic — exactly *not* a student demo. **[verified]** RunPod's 190ms–5s numbers are the weight-*load* step on an already-warm-ish host; a truly cold worker after scale-to-zero must also cold-pull/start the container, init the vLLM engine, allocate the KV cache, and capture CUDA graphs — realistically **tens of seconds to ~2 minutes** on the first request. **What changes:**
- For grading/demo, do not rely on cold-start magic: spin up **1 active worker for the review window** (costs cents/hour), warm it with 2–3 requests, then scale back to zero. Document this in the README.
- In the eval harness and README, send **2–3 warm-up requests** and report **"cold-start latency (first request): ~X s" SEPARATELY from "warm p95."** Set a **generous client timeout** (≥180s) so a cold request doesn't read as a crash.
- Report cold start honestly in the report: *"tens of seconds to ~2 min on first request after idle; <2s warm."* This turns an ugly number into an expected POC finding instead of a fire drill.

This still satisfies the required **vLLM + Docker** stack (HF Inference Endpoints uses TGI, not vLLM, so it would not). *(Sources: docs.runpod.io/serverless/development/optimization; blog.runpod.io/run-larger-llms-on-runpod-serverless-than-ever-before/.)*

### Decision F — Do you even need the FastAPI wrapper?

- **Option 1 — No wrapper.** vLLM's built-in server already gives OpenAI-compatible endpoints, auth, streaming, metrics; inject the prompt client-side **[Serving]**.
- **Option 2 — Thin FastAPI wrapper.** Auto-injects the safety system prompt, forces `enable_thinking=False`, and writes the GDPR audit log **[Serving][Eval]**.

**Recommendation: thin wrapper, one `/triage` POST.** **Criteria:** the safety prompt must not be optional, the audit log is part of the GDPR/traceability story, and the wrapper gives real, mockable code to unit-test in CI. **Red-team addition (high):** the wrapper must **also always inject `chat_template_kwargs={"enable_thinking": false}`** so a caller cannot accidentally re-enable reasoning (see Decision H).

### Decision G — Experiment tracking: W&B vs MLflow vs none

- **Option 1 — Weights & Biases (W&B).** Free tier, one-line `report_to="wandb"`, hosted loss/reward-margin dashboards — good report screenshots **[Unsloth][TRL]**.
- **Option 2 — MLflow.** Open-source but needs a tracking server/store — extra friction on ephemeral Kaggle sessions.
- **Option 3 — None / CSV.** Zero dependency.

**Recommendation: W&B**, with `report_to="none"` + screenshots as the fallback if auth is a hassle. **Criteria:** lowest setup on ephemeral notebooks, directly produces the SFT-loss and DPO `rewards/margins` plots the report needs, and **log the W&B run id in the report** for reproducibility (see Decision I). We are not acting on the reviewers here beyond reproducibility — W&B is low-risk and well-trodden.

### Decision H (NEW) — Thinking-mode: a serve-time decision, not a train-time one

**Red-team correction (high, technical) — v1 misunderstood this. [verified]** v1 implied that stripping `<think>` blocks during SFT and setting a tokenizer flag would stop the model thinking. Reality: `enable_thinking=False` must be supplied **at request time** via `chat_template_kwargs={"enable_thinking": false}` (or server-wide via `--default-chat-template-kwargs '{"enable_thinking": false}'`), it only works cleanly on **vLLM ≥ 0.9.0**, and when you merge, the model's copied chat_template **still defaults to thinking-ON** — so a caller who forgets the flag gets `<think>` blocks even though you never trained on them. If the wrapper or eval harness forgets it, the live demo emits visible reasoning, latency balloons, and the disclaimer/escalation regex may miss content buried in reasoning.

**Recommendation (defense in depth — make non-thinking the default in two places):**
1. Launch the RunPod worker with `--reasoning-parser qwen3 --default-chat-template-kwargs '{"enable_thinking": false}'` and pin **vLLM ≥ 0.9.0**.
2. Have the FastAPI wrapper **always** inject `chat_template_kwargs={"enable_thinking": false}` per request.
3. Add one eval probe asserting no `<think>` string appears in any output.

*(Sources: docs.vllm.ai/en/latest/features/reasoning_outputs/; github.com/modelscope/ms-swift/issues/5836.)*

### Decision I (NEW) — The CI/CD deliverable must match what is actually deployed

**Red-team correction (high, technical) — v1 had an architectural contradiction.** v1's CI/CD built+pushed a Docker image to GHCR *and* recommended serving via the **prebuilt stock worker-vllm image configured by env vars** (`MODEL_NAME`, `HF_TOKEN`). In that serving path you **never build or push a custom model image**, so the GHCR build step deploys nothing the endpoint uses — a "deploy step" that is theater, and a graded deliverable that doesn't hold up to one click.

**Recommendation — pick one coherent story:**
- **Model serving:** stock `runpod-workers/worker-vllm` pointed at your HF repo via env vars. No custom model image.
- **CI/CD:** lints (ruff) + tests (pytest, **mocked vLLM**) + builds/pushes the **FastAPI wrapper image** to GHCR — the wrapper is the only custom code you own, so it is the honest thing to build. That is a complete, demonstrable, green pipeline.
- **RunPod redeploy:** a **manual, documented step** (a one-line `curl` to the RunPod REST API to update the endpoint's env, plus a screenshot in the report). Auto-deploy-to-serverless from GH Actions is the most fragile, least-documented link in the chain and a near-certain rabbit hole; a POC's deliverable is *"a CI/CD pipeline exists,"* not *"continuous deployment to GPU serverless works flawlessly."* **Adopt the manual redeploy as the plan, not the contingency.**
- **(Optional, behind `workflow_dispatch`):** a post-deploy smoke step that sends ONE OpenAI-format request to the live RunPod URL and asserts HTTP 200 + non-empty text. This closes the "green CI, dead endpoint" gap — the one integration CI on a CPU runner otherwise cannot see (bad `MODEL_NAME`, gated-repo 401, dtype/VRAM mismatch). Run it manually before grading to avoid paying for a cold start on every push.

*(Sources: github.com/runpod-workers/worker-vllm; docs.runpod.io/serverless/workers/github-integration; runpod.io/articles/guides/integrating-runpod-with-ci-cd-pipelines.)*

---

## 4. Recommended tool stack

| Tool | Role | Why chosen (criteria) |
|---|---|---|
| **Qwen3-1.7B (Instruct)** | Base for SFT | Apache-2.0 (commercial + medical OK), fits 4-bit on a T4, **inherits refusal/safety priors** worth more than format purity for a safety-graded POC; ships ChatML template either way **[Model] + red-team C2** |
| **Unsloth + TRL** | SFT + DPO training | ~2× faster / ~70% less VRAM on the free T4; near-drop-in TRL fallback if it breaks **[Unsloth][TRL]** |
| **LoRA (PEFT), r=16** | Parameter-efficient fine-tuning | Trains ~0.3% of params; enough for 1.7B domain adaptation on ~5k samples; low overfitting risk **[TRL][Model]** |
| **DPO (TRL DPOTrainer)** | Preference alignment + safety behavior | Offline, no reward model; **run on the adapter-attached SFT model, merge once after** **[TRL][verified]** |
| **Hugging Face Datasets** | Source data + ETL | Public sources; ETL is CPU-only **[Data]** |
| **Microsoft Presidio** | PII verification + audit | Documented, reproducible GDPR check; bilingual `fr_core_news_md` + `en_core_web_lg`; report ACTUAL findings **[GDPR]** |
| **vLLM (`vllm/vllm-openai`), ≥0.9.0** | Inference server | PagedAttention throughput + OpenAI API; **≥0.9.0 required** for clean `enable_thinking=False` **[Serving][verified]** |
| **FastAPI (thin wrapper)** | Prompt injection + thinking-off + logging | Forces safety prompt + `enable_thinking=False`; produces the audit log; testable code for CI **[Serving][Eval]** |
| **Docker (`vllm/vllm-openai` base)** | Packaging the WRAPPER | Build/push only the wrapper image; the model is served by stock worker-vllm via env vars **[Serving] + red-team I** |
| **RunPod Serverless + stock worker-vllm** | Hosting | Cheapest GPU, scale-to-zero; **`min-workers=1` for the grading window**; honest cold-start reporting **[Serving]** |
| **GitHub Actions** | CI/CD (wrapper) | ruff + pytest(mock) + build/push GHCR; manual documented RunPod redeploy **[Serving] + red-team I** |
| **Weights & Biases** | Experiment tracking | One-line integration, hosted plots; log run id for reproducibility **[Unsloth][TRL]** |
| **scikit-learn + numpy** | Eval harness | Accuracy, Cohen's κ, per-class recall, p95 latency; **report raw confusion counts too** on the small vignette set **[Eval]** |
| **Kaggle (T4, 30h/week, 12h/session)** | Free training GPU | No idle disconnect (vs Colab); note the **12h per-session cap** and checkpoint accordingly **[Unsloth]** |

---

## 5. The triage-data strategy

**Target: ~5,000 SFT pairs (≈80% FR / 20% EN) + ~1,500 DPO pairs (safety-weighted, ~half FR), all with no real patient data.** **Budget 1.5–2 days, not 1** — the ETL *runs* in minutes but getting it *correct* (schema surprises, ChatML templating, language balancing, dedup against the DPO set) is the classic hidden time-sink, and v1 listed four known data caveats, each a small debugging session.

**Day-1 10-minute load-smoke (do this BEFORE Day-2 ETL):** `load_dataset()` each source, print one row + config names + row count, and confirm the **license on the card**. This catches a gated/removed/renamed dataset with 13 days left, not 10.

**Verified ground truth (HuggingFace, June 2026):** MediQAl is public, CC-BY-4.0, configs MCQU (17,017), MCQM (10,617), OEQ (4,969).

**Step 1 — French SFT core (~4,000 rows) [Data]**
- **MediQAl OEQ** (4,969 open-ended FR): question→free-text answer maps directly to SFT. Filter `len(answer) > 50`; take ~3,500.
- **MediQAl MCQU "Reasoning" subset:** MCQ→instruction, response `"La réponse est {lettre}. {answer_text}."` Take ~500 (cap MCQ-style).
- **FrenchMedMCQA** (Apache-2.0, ~2,171 train): same template, single-answer rows; take ~1,800 (pharmacy vocabulary).

**Step 2 — Triage-flavor slice (~300–500 rows) [Data] — Decision B, MIETIC removed**
- ~300 MediQAl `medical_subject == "Urgences"` rows (CC-BY-4.0, exam-derived) → 3-part triage template: (1) one clarifying follow-up, (2) urgency level (urgence maximale / modérée / différée), (3) recommended action. Urgency is a **heuristic from subject + question type — flagged as heuristic, not clinical** in the report.
- ~30–50 **hand-written synthetic English triage vignettes** (no real patients) for EN grounding — replaces the dropped MIETIC rows.

**Step 3 — English SFT (~1,000 rows) [Data]**
- **MedQuAD:** **do not "treat as CC-BY-4.0."** Use a HF copy that ships an **explicit license/dataset card** (e.g. `lavita/MedQuAD`) and record the *actual stated* license in the provenance card; original NIH sources had per-source usage restrictions, so a blanket assertion is unsafe. If no clean card exists, downscope EN-from-MedQuAD and lean more on FrenchMedMCQA (Apache-2.0, unambiguous). Filter `qtype ∈ {symptoms, treatment, exams and tests, complications, prevention}`, `len(Answer) > 100`; stratified sample ~1,000.

**Running total: ~5,300–5,500 rows → reserve ~500 for validation → ~5,000 SFT train.** Language mix ≈80/20, no machine translation **[Data]**.

**Step 4 — DPO preference set (~1,500 rows, safety-weighted) [Data][Eval] — Decision C rebalanced**
- **~300–500 hand-written bilingual safety pairs** (≈half FR): chosen = acknowledge → escalate → disclaimer → no diagnosis; rejected = reassure-and-continue. Reuse the same red-flag scenarios as the eval probes.
- **~1,000 rows of UltraMedical-Preference** (MIT), filtered as in Decision C. Framed as a technique demonstration, not a clinical-quality signal.

**Step 5 — Anonymization + audit (GDPR deliverable) [GDPR]** — **reframed as a test with a hypothesis, not a foregone conclusion.** Run Presidio (bilingual, `replace` operators, threshold 0.5, custom recognizers for French NIR/IPP) over every record. State: *"Sources are exam questions / public NIH text and contain no patient records, so we hypothesize minimal PII; we ran Presidio to verify and report actual findings"* — then **report the real numbers, whatever they are.** Because MIETIC is gone, near-zero PII is now a *defensible genuine expectation* rather than a naive pre-declaration. Produce: per-source provenance cards (license, URL, PII status), a JSON audit log (source, SHA-256 of raw text, Presidio version, entities found, timestamp), an automated re-scan, and a 2% manual spot-check. Cite **GDPR Recital 26** (anonymous/non-personal data is out of scope) for exam-derived data; **never claim anonymity for any patient-derived source** (there are none, by design). This four-part story (legal basis → process → audit trail → QC) is what the GDPR criterion rewards.

**Eval/test sets (built once, ~3–4h incl. manual labelling) [Eval]:** ~50 held-out MCQA items + ~15–30 **hand-labelled** FR triage vignettes (urgency assigned by hand from symptoms, **independent of training heuristics**) + ~20 adversarial safety probes (reusing the red-flag scenarios). Written de novo / held out — no leakage, no PII. **This is 4–8 hours of careful manual authoring — it is on the critical path; do not rush it, or the safety story and headline metrics become meaningless.**

**Critical chat-template rules [Model][verified] — corrected from v1:**
- Wrap every example in the Qwen3 ChatML template; the assistant turn ends with `<|im_end|>`; **compute loss on assistant tokens only.**
- **Do NOT hard-code `eos_token="<|im_end|>"`.** **[verified]** Qwen3-1.7B's configured `eos_token` is `<|endoftext|>` (id 151643), tokenizer class `Qwen2Tokenizer`; `<|im_end|>` is the ChatML *turn terminator inside the template*, not the model's EOS. After loading, read `tokenizer.eos_token` and use it; train so the model emits `<|im_end|>` at end of the assistant turn, and at inference add `stop=["<|im_end|>"]` to vLLM while leaving the model's real `eos_token` alone. **Verify on Day 2** by decoding 5 fully-templated rows back through the tokenizer to confirm only the assistant span is the label and the stop token is correct — catching this on 50 rows costs minutes; catching it post-SFT costs a training run. *(Source: huggingface.co/Qwen/Qwen3-1.7B-Base/raw/main/tokenizer_config.json.)*
- `enable_thinking=False` is a **serve-time** flag (Decision H), not solved at training time. Add a system-prompt line "respond in the same language as the question" to curb language mixing.

---

## 6. Day-by-day plan over ~2 weeks (re-baselined to ~7 working days + named buffer)

**Read this first:** the "days" are half-time, so this is ~7 working days. Two structural rules from the feasibility review: **(1) start the report on Day 1 as a running build-log** (paste decisions, screenshots, error messages, metrics as they happen into one markdown file — the 20 pages become editing); **(2) protect 2 full days of buffer between "endpoint live" and "report due," not one shared day.**

| Day | Focus | Checkpoint | Minimum-viable fallback |
|---|---|---|---|
| **1** | Accounts + the two riskiest smoke tests, immediately. Create GitHub repo (start `REPORT.md` build-log), RunPod + HF + W&B + Kaggle Secrets. **(a) Unsloth GPU smoke:** `import unsloth; FastLanguageModel.from_pretrained(Qwen3-1.7B 4-bit)` + 5 training steps + `df -h /kaggle/working`; `pip freeze > requirements-train.lock.txt` and commit. **(b) Data load-smoke:** `load_dataset()` all sources, print row+config+row count, confirm license on card. | Unsloth runs 5 steps on the current Kaggle image; all datasets load; lockfile committed. | If Unsloth churns >30 min, decide TRL+PEFT **now** (13 days left). If a dataset is gated/removed, drop the triage slice. |
| **2** | Data ETL. Filter, template-wrap to ChatML, build ~5,000 SFT + ~1,500 DPO JSONL. **Verify templating on 50 rows: decode back, confirm assistant-only labels + correct stop token + system prompt present.** | JSONL exists; 5 samples/source printed; language mix ≈80/20; template verification passes. | Drop triage slice; ship pure QA SFT. |
| **3** | Manual authoring + GDPR. Write the ~300–500 bilingual safety pairs + ~15–30 hand-labelled triage vignettes + ~20 probes (reuse scenarios across all three). Run Presidio; write audit log + provenance cards + re-scan + 2% spot-check; draft REPORT RGPD section. | Hand-built sets exist; audit JSONs saved; real PII numbers reported; provenance cards done. | Cut to ~30 safety pairs + ~15 vignettes (templated). Presidio FR-first if time-tight. |
| **4** | Full SFT. Train ~5,000 rows, **1–2 epochs (default, not 3)**, `save_steps=50`, **push LoRA adapter to HF Hub at each save interval** (so a killed 12h session never costs >50 steps), `report_to="wandb"`, `set_seed(3407)` + seeded shuffle. Bilingual generation test. | Adapter saved + on Hub; FR and EN prompts answer in the right language; no OOM. | 1 epoch / smaller train set if quota tight; resume from the pushed checkpoint. |
| **5** | **HARD GATE moved up: DPO + merge + offline verify.** DPO on the **adapter-attached** SFT model, `ref_model=None`, `beta=0.1`, 1 epoch, lr 5e-6, watch `rewards/margins`. Then **merge ONCE**, assert files+size, **verify with in-notebook `vllm.LLM().generate()`**, push tagged revision. | Margins positive/rising; merged folder has safetensors+config+tokenizer_config (ChatML); offline vLLM generates a clean triage answer ending at `<|im_end|>` with no `<think>`. | Ship SFT-only if DPO destabilizes; keep Drive + private-HF backup of weights. |
| **6** | **HARD GATE: a live endpoint exists.** RunPod serverless via stock worker-vllm: `MODEL_NAME`, `MAX_MODEL_LEN=2048`, `DTYPE=bfloat16`, `HF_TOKEN`, `--reasoning-parser qwen3 --default-chat-template-kwargs '{"enable_thinking": false}'`, model caching on, idle 60s. | `curl` gets a triage response (allow ≥180s on the cold first call); no `<think>` in output. | Fall back to a one-GPU RunPod pod running `vllm/vllm-openai` directly. |
| **7** | FastAPI wrapper + Docker. `/triage` injects safety prompt + `enable_thinking=False` + audit log; `FROM vllm/vllm-openai:<pinned>`, **wrapper image only, no weights baked.** | Local Docker serves `/triage`; logs show audit fields; thinking disabled. | Skip wrapper, inject prompt client-side (Decision F MVP). |
| **8** | CI/CD (wrapper). GitHub Actions: ruff + pytest(mock vLLM) → build+push wrapper to GHCR. Document the manual `curl` RunPod redeploy + screenshot. Optional `workflow_dispatch` post-deploy smoke. | Green pipeline on push; wrapper image in GHCR; redeploy documented. | Stop at build+push; redeploy manually (this IS the plan). |
| **9** | Eval harness. Run ~100 cases against the live endpoint with a generous timeout: behavioral metrics first (lang-match, disclaimer rate, escalation rate, format, no-`<think>`), MCQA as a sanity check, triage agreement on hand-labelled vignettes (κ + raw confusion counts), **cold vs warm latency reported separately.** **Hand-review every emergency probe** against a short rubric (escalated? disclaimer? no diagnosis/dose? plausible-but-wrong?). | Metrics table + manual safety-review notes produced. | Run against local vLLM if endpoint flaky; smaller case set. |
| **10** | Base-vs-tuned comparison + safety write-up. Compare base Qwen3-1.7B vs your SFT+DPO model on the same set; lead with behavioral deltas, present MCQA honestly even if flat/down ("expected: domain/format adaptation, not benchmark maximization"). | Before/after numbers; safety metrics reported as **measured numbers + manual review**, not as pass/fail "gates." | Report whatever you measured honestly, including failures. |
| **11–12** | **Report (protected buffer + writing).** Edit the Day-1 build-log into 20 pages: data + GDPR (incl. the MIETIC-exclusion sentence), SFT/DPO method + curves, serving architecture + cost + honest cold-start, CI/CD, eval + safety + the correctly-cited Goh framing. | Draft covers all 5 deliverables. | Bullet-point appendices over prose if short. |
| **13–14** | **Buffer + polish.** Re-run demo end-to-end from a clean checkout (using the committed lockfile), `min-workers=1` + warm-up before any review, fix README, final report pass. | Everything runs from a clean checkout; warmed demo works. | — |

**Hard checkpoints (do-or-pivot):** end of Day 1 (Unsloth + data proven), **end of Day 5 (a tested merged model exists)**, **end of Day 6 (a live endpoint exists, however rough)**. If any slips, invoke the MVP path immediately.

---

## 7. Risks & how we keep it simple

| Risk | Why it bites a POC | The decision that keeps it proportionate |
|---|---|---|
| **Credentialed-data license (MIETIC/MIMIC)** | Re-distributing PhysioNet credentialed patient data in a public dataset/weights violates the DUA and breaks the GDPR story | **Drop MIETIC entirely**; triage slice from MediQAl "Urgences" + hand-written synthetic EN vignettes; turn the exclusion into a documented GDPR-maturity point (Decision B) **[verified]** |
| **Half-time timeline reads as 14 days but is ~7** | Back-half cloud/devops tasks + a 20-page report get crushed into a panic | Re-baseline to ~7 working days; report as a Day-1 build-log; endpoint-live gate at Day 6; 2 protected buffer days (Day-by-day) |
| **Wrong eos_token / thinking-mode leaks** | Model runs past the answer or emits visible `<think>` reasoning in the live demo | Read `tokenizer.eos_token` (it's `<|endoftext|>`), train on `<|im_end|>` turn terminator + vLLM `stop=["<|im_end|>"]`; force `enable_thinking=False` at server AND wrapper; assert no `<think>` in eval (Decisions H + §5) **[verified]** |
| **DPO ordering** | Merging before DPO makes `ref_model=None`'s implicit reference undefined → silently wrong training or T4 OOM | Run DPO on the adapter-attached SFT model; assert it's a `PeftModel` before `DPOTrainer`; merge once after (Decisions C/D) **[verified]** |
| **CI/CD theater** | A green pipeline that builds an image the endpoint never uses fails inspection | CI builds/tests only the wrapper; model served by stock worker-vllm via env; manual documented redeploy + optional live smoke (Decision I) |
| **Free-GPU limits** — T4 VRAM, 12h-session + 30h/week caps, version breakage | A lost run or broken install eats a day | Kaggle over Colab; Unsloth headroom; **1–2 epochs**; `save_steps=50` + push adapter to Hub every interval; commit lockfile; Day-1 smoke (§6) |
| **Cold-start over-promise** | Grader hits a cold endpoint, waits ~2 min, sees "broken" | `min-workers=1` for the review window; warm-up calls; report cold vs warm latency separately; ≥180s client timeout (Decision E) **[verified]** |
| **Over-claiming results** | A 1.7B model won't hit GPT-4 numbers; MCQA may even drop after fine-tuning | Lead with behavioral metrics; MCQA as a "didn't break knowledge" sanity check, reported honestly either way; safety as measured numbers + manual review, not fake "100%/90% gates" (Decisions C/§9) |
| **Circular triage metric** | Eval against the same heuristic labels the model trained on measures a fiction | Eval vignettes hand-labelled by a different process; report "agreement with our heuristic scale" + raw confusion counts + caveat (Decision B) |
| **Clinical safety** | The one place a POC can do real harm if demoed naively | Safety in system prompt **and** a larger bilingual DPO safety slice; mandatory disclaimer, no diagnosis/prescription; every emergency probe hand-reviewed **[Eval]** |
| **Scope creep** — GRPO, thinking mode, multi-adapter serving, auto-deploy | Plausible rabbit holes with no POC payoff | Explicit "no" list: no GRPO, no thinking mode, no runtime LoRA (merge), no auto-deploy (manual redeploy), no MT, subsample big sets |

**The simplicity through-line:** every fork resolves toward *the option with the fewest moving parts that still satisfies the deliverable*. The red-team did **not** push us into over-engineering; where it suggested complexity (auto-deploy CI, per-record clinical validation), we explicitly declined and kept the POC-proportionate path.

---

## What the red-team changed

Severity uses the reviewers' own labels. Every **blocker** and **high** is resolved; mediums are folded into the relevant sections.

| # | Blindspot (reviewer) | Severity | What the final plan does |
|---|---|---|---|
| 1 | **MIETIC is PhysioNet *credentialed* MIMIC data, not CC-BY-NC-SA** — redistributing it breaks the DUA and the GDPR narrative | **blocker** | **Removed MIETIC entirely** (verified June 2026). Triage slice from MediQAl "Urgences" + hand-written synthetic EN vignettes. Exclusion documented as a GDPR-maturity point. (Decision B, §5, §7) |
| 2 | **Wrong eos_token** — v1 said `<|im_end|>`; actual is `<|endoftext|>` | **high** | **Verified and corrected.** Don't hard-code; read `tokenizer.eos_token`; train on `<|im_end|>` turn terminator + vLLM `stop=["<|im_end|>"]`; verify a clean stop. (§5, Decision H) |
| 3 | **Thinking-mode is a serve-time flag, not train-time** — merged model defaults to thinking-ON | **high** | New Decision H: force `enable_thinking=False` at server (`--default-chat-template-kwargs`, vLLM ≥0.9.0) AND in the wrapper; eval probe asserts no `<think>`. |
| 4 | **Stale hard version pins** (`transformers==4.56.2` etc.) | **high** | **Deleted the numbers.** Day-1: run the current official notebook's install cell, `pip freeze` a committed lockfile, assert not in unsloth blocklist. (Decision A) |
| 5 | **CI/CD conflates two deploy models** — GHCR image the endpoint never uses | **high** | New Decision I: CI builds/tests the **wrapper** only; model served by stock worker-vllm via env; manual documented redeploy + optional live smoke. |
| 6 | **Half-time "14 days" = ~7 working days; back-half overloaded; report compressed** | **high** | Re-baselined §6: report as a Day-1 build-log; endpoint-live gate pulled to Day 6; 2 protected buffer days. |
| 7 | **DPO on UltraMedical doesn't teach clinical quality** (English, GPT-4-biased) | **high** | Decision C rebalanced: ~300–500 bilingual safety pairs + ~1,000 UltraMedical (~1,500 total); reframed as technique demo + safety lever, not clinical quality. |
| 8 | **Safety "100%/≥90% CI gates" can't be backed by a ~20-probe regex harness** | **high** | Demoted to **reported metrics + manual safety review** of every emergency probe with a short rubric. No fake guarantees. (§9, §6 Day 9) |
| 9 | **Circular triage metric** (eval reuses training heuristic labels) | **high** | Eval vignettes hand-labelled by a different process; report "agreement with our heuristic scale" + raw confusion counts + caveat. (Decision B) |
| 10 | **RunPod cold start ≫ "5–15s"** (≈ tens of s to ~2 min after scale-to-zero) | **high/med** | Decision E: `min-workers=1` for the review window; warm-ups; cold vs warm latency reported separately; ≥180s client timeout. **Verified.** |
| 11 | **DPO-after-merge ordering** breaks `ref_model=None` | **medium** | Stated as an invariant: DPO on adapter-attached model, assert `PeftModel`, merge once after. **Verified.** (Decisions C/D) |
| 12 | **`save_pretrained_merged` footguns** (disk blowup; silent empty write) on Kaggle | **medium** | Clear cache, assert files+size, offline `vllm.LLM()` verify in-notebook before push, Drive/HF backup. (Decision D, Day 5) |
| 13 | **No GPU in CI → model path untested** | **medium** | Optional `workflow_dispatch` post-deploy smoke hitting the live URL (HTTP 200 + non-empty). (Decision I) |
| 14 | **Manual authoring (safety pairs, vignettes, labels) hidden in the schedule** | **medium** | Surfaced as a dedicated Day-3 task; counts right-sized; scenarios reused across DPO/vignettes/probes. (§5, §6) |
| 15 | **Base vs Instruct safety tradeoff never weighed** | **medium** | New Decision C2: **switch to Qwen3-1.7B Instruct** for inherited safety priors; document the tradeoff. **Verified** Base ships a ChatML template anyway. |
| 16 | **Data budgeted at 1 day** despite four known caveats | **low→med** | Re-budgeted to 1.5–2 days; Day-1 load-smoke; Day-2 50-row template verification. (§5, §6) |
| 17 | **Kaggle 12h-session cap + checkpoint resume not addressed** | **medium** | Default 1–2 epochs; push adapter to Hub at each `save_steps`; resume-from-checkpoint; disk check. (Day 4) |
| 18 | **Merged-model sanity-load assumed a local GPU the learner lacks** | **medium** | Use vLLM **offline engine** in the same notebook cell — no server/port. (Decision D, Day 5) |
| 19 | **JAMA citation miscited/over-read** | **low** | **Corrected (verified):** Goh et al., *JAMA Network Open* 2024;7(10):e2440969 — adding an LLM to physicians' usual resources did **not** significantly improve diagnostic reasoning (LLM-alone scored ~16% higher than physicians). Drawn modestly: capability isn't the only bottleneck; integration/over-reliance are real → hence the disclaimer framing. |
| 20 | **MedQuAD license hand-waved** | **low** | Use a HF copy with an explicit card; record the *actual* stated license; else lean on FrenchMedMCQA. (§5) |
| 21 | **Presidio "expect ~0 PII" pre-declared as a foregone conclusion** | **medium** | Reframed as a hypothesis-tested verification; report actual numbers; near-zero is now genuinely defensible since MIETIC is gone. (§5) |
| 22 | **Reproducibility under-specified** (seeds, model versioning, lockfile) | **low** | `set_seed(3407)` + seeded shuffle; committed lockfile; tagged HF revision; dataset SHA-256 (reuse GDPR per-record SHAs); log W&B run id. |
| 23 | **MCQA could anchor a "fine-tuning improved things" claim it may contradict** | **medium** | Lead eval with behavioral metrics; MCQA is a secondary "didn't destroy knowledge" check, pre-committed to honest reporting either way. (§9, Day 10) |

**One reviewer claim we did NOT fully act on, and why:** the suggestion to add per-probe *clinical* validation / a clinician-grade safety guarantee. For a solo 2-week POC with no clinician in the loop, that is disproportionate and unachievable; instead we *demote the claim* (manual author review + explicit "not a clinical validation" caveat) rather than build machinery we cannot staff. That is the honest, proportionate response and itself scores well on the rigor dimension.

---

## 8. Evaluation framing (clarified)

Lead the eval with **behavioral metrics that fine-tuning genuinely changes** — language-match rate, disclaimer presence, escalation rate on red-flag probes, format adherence, and a no-`<think>` check. Present **MCQA accuracy as a secondary "sanity check that we did not destroy medical knowledge,"** pre-committed to reporting it honestly even if flat or down (domain/format adaptation often trades benchmark accuracy for format adherence). Report **triage agreement only on the independently hand-labelled vignettes**, as "agreement with our heuristic urgency scale," with raw confusion counts alongside Cohen's κ (a single label flip is visible on ~15–30 vignettes). Treat the safety numbers (disclaimer present, emergency escalation) as **measured results plus a manual safety review of every emergency probe** — not as pass/fail CI gates, which a small author-written regex harness cannot honestly back. Cite **Goh et al., *JAMA Network Open* 2024;7(10):e2440969** precisely (adding an LLM to physicians' usual resources did not significantly improve diagnostic reasoning) to support the modest, accurate inference that the assistant's value is in *guiding with guardrails*, not raw capability.

---

## 9. What I need from you

Please confirm or decide these before implementation — each changes a concrete step:

1. **Base model: Instruct (now recommended) vs Base.** I propose switching to **Qwen3-1.7B Instruct** for inherited safety priors (Decision C2). OK, or do you want to keep -Base for the learning narrative (then we over-weight safety in SFT)? **[Model]**
2. **Triage framing & urgency taxonomy.** OK to reframe as "bilingual medical QA assistant with triage-style guidance" + a small heuristic slice? And which urgency scale — the simple 3-level (maximale / modérée / différée), or a specific CHSA/CCMU/Manchester scale? This sets templates and eval reference labels. **[Data][Eval]**
3. **Grading rubric specifics.** Does the rubric (a) require a formal DPIA or is a report RGPD section enough; (b) set a numeric pass threshold for triage agreement / emergency recall (I'm now reporting these as *measured* numbers + manual review, not as gates); (c) treat the DPO set as a *separate* GDPR deliverable or bundled with SFT? **[GDPR][Eval]**
4. **DPO go/no-go + mix.** Confirm DPO with ~300–500 bilingual safety pairs + ~1,000 UltraMedical (~1,500 total), framed as technique-demo + safety lever, vs SFT-only fallback. **[Data][Eval]**
5. **Endpoint liveness.** Acceptable to use scale-to-zero with `min-workers=1` warmed for the grading window and an honestly-reported ~tens-of-seconds-to-2-min cold start? This is the single biggest cost lever. **[Serving]**
6. **License acceptability.** Confirm fine for an educational POC: Unsloth (Apache-2.0), UltraMedical-Preference (MIT), MediQAl (CC-BY-4.0), FrenchMedMCQA (Apache-2.0), MedQuAD (whichever carded copy we pick). **Confirm you're OK that MIETIC/MIMIC is excluded** (it's a credentialed-data blocker). **[Data]**
7. **Accounts/credits ready?** HuggingFace token, RunPod with a few dollars of credit, Kaggle phone-verified for GPU, W&B account — so Day 1 is smoke tests, not signups.

Once 1–5 are confirmed, Day-1 work can start immediately.

---

### Confidence & what I could not verify
- **High confidence (re-verified against primary sources, June 2026):** eos_token = `<|endoftext|>` and Base ships a ChatML+thinking template (HF tokenizer_config.json); MIETIC = PhysioNet credentialed, no-redistribution (PhysioNet DUA/required-training pages); `enable_thinking=False` is a serve-time flag needing vLLM ≥0.9.0 (vLLM docs); DPO `ref_model=None` recovers the reference by disabling the adapter so DPO must precede the single merge (TRL docs/issues); Goh et al. JAMA Network Open 2024;7(10):e2440969 finding.
- **Medium confidence (reported in the source reports, plausible but version-sensitive):** exact RunPod cold-start seconds (varies by GPU class/caching); Unsloth speed/VRAM multipliers; MediQAl/UltraMedical row counts and split schemas — all to be re-confirmed by the Day-1 load-smoke.
- **Could not verify here:** the OpenClassrooms rubric specifics (DPIA requirement, numeric thresholds, DPO-as-separate-deliverable) — these are in question 3 to you; and the current exact Unsloth/transformers pins (deliberately not pinned in this doc — captured live via lockfile on Day 1).

---

## Appendix — Decision points (structured)

### Decision 1: Training framework: Unsloth vs plain TRL+PEFT

**Options:** Unsloth + TRL (custom kernels, ~2x faster, ~70% less VRAM, but version-fragile) · Plain TRL + PEFT (official HF, fully debuggable, ~2-4x slower, more VRAM)

**Recommendation:** Unsloth + TRL, with plain TRL+PEFT as a documented fallback if Unsloth fails to install on the Kaggle T4 within ~30 minutes. Do NOT copy hard version pins from any document; on Day 1 run the current official Unsloth Qwen3 notebook's install cell, pip freeze a committed lockfile, and assert the transformers version is not in unsloth_zoo's blocklist.

**Why:** On a free T4 with disconnect risk, training speed and VRAM headroom directly reduce the chance of losing a run. The only real cost is version fragility. The red-team showed the v1 pins (transformers==4.56.2, trl==0.22.2) are stale and that the official installer now pulls from GitHub main with an explicit incompatible-version blocklist, so the safe move is a live-captured lockfile, not memorized numbers. Switching to TRL later is near-drop-in and the report narrative is unaffected.

### Decision 2: Base vs Instruct model for SFT (NEW, raised by red-team)

**Options:** Qwen3-1.7B-Base (you build all instruction-following/refusal/safety from scratch via SFT+DPO+prompt) · Qwen3-1.7B Instruct (Apache-2.0, fits the T4, inherits refusal/safety priors; SFT adapts format/domain on top)

**Recommendation:** Switch the SFT base to Qwen3-1.7B Instruct, and document the Base-vs-Instruct safety tradeoff in the report. If keeping Base for a learning narrative, over-weight safety examples in the SFT data and say so.

**Why:** For a two-week clinical-safety-graded POC, an inherited safety head-start (refusal behavior, no-diagnosis priors) is worth more than format purity. The 'I own the format' benefit of Base is largely illusory: verified that Qwen3-1.7B-Base already ships the full ChatML chat_template (with <think> logic), so you do not get a blank slate either way. It is a one-line model-name change at zero cost that reduces the chance of confident dangerous output.

### Decision 3: How to bridge the QA->triage data gap (MIETIC must be dropped)

**Options:** Reframe as bilingual medical QA assistant with triage-style guidance via system prompt (low effort, honest) · Add a ~300-500 row triage-flavored slice from EXAM-DERIVED MediQAl 'Urgences' rows plus hand-written synthetic English vignettes (no real patients) · Train primarily on a real triage dataset like MIETIC/FedMML (disproportionate ETL AND license-blocked: MIETIC is PhysioNet credentialed MIMIC data)

**Recommendation:** Use the reframe as the overall frame PLUS a ~300-500 row triage slice built ONLY from MediQAl 'Urgences' (CC-BY-4.0) and ~30-50 hand-written synthetic English vignettes. Drop MIETIC entirely and document the exclusion. Hand-label eval vignettes by a separate process from the training heuristic to avoid circular validity.

**Why:** Verified June 2026 that MIETIC is PhysioNet Credentialed Health Data License 1.5.0 (CITI training, no third-party sharing, no redistribution off PhysioNet) derived from de-identified real MIMIC-IV-ED records, not the CC-BY-NC-SA its HF re-upload claims. Baking it into a publicly-published dataset and weights would violate the DUA and detonate the 'no real patient data, GDPR-by-construction' narrative. Exam-derived sources keep clean provenance, and the explicit exclusion sentence turns a blocker into evidence of GDPR maturity.

### Decision 4: Run DPO, and with what preference mix?

**Options:** Run DPO on the adapter-attached SFT model with ~300-500 bilingual hand-written safety pairs + ~1,000 subsampled UltraMedical-Preference rows (~1,500 total) · Run DPO on a 3,000-row UltraMedical slice + ~50-100 safety pairs (the v1 mix) · Skip DPO, ship SFT-only

**Recommendation:** Run DPO with the rebalanced ~1,500-row safety-weighted bilingual mix, on the SFT model WITH the LoRA adapter still attached (ref_model=None), then merge once after. Frame it as a technique demonstration plus a safety-behavior lever, NOT as 'improved clinical quality.' Keep SFT-only as a fallback if DPO destabilizes.

**Why:** DPO is a required graded deliverable and the right place to bake in safety so it survives a bypassed system prompt. But UltraMedical is English-only and GPT-4-scored (self-preference bias) and differs mostly in formatting, so it does not teach clinical quality; over-weighting hand-written bilingual safety pairs is where the genuine value is. Verified that with PEFT, ref_model=None recovers the reference by temporarily disabling the adapter, so DPO must run before the single post-DPO merge or the reference policy is undefined / OOMs the T4.

### Decision 5: How live must the demo endpoint be (cost vs responsiveness)?

**Options:** RunPod serverless, scale-to-zero, 60s idle, accept a cold start · RunPod serverless with min-workers=1 only during the grading/demo window, scale back to zero after · Always-on dedicated pod (instant, bills continuously)

**Recommendation:** Serverless scale-to-zero with model caching/FlashBoot for normal operation, PLUS set min-workers=1 for the ~1-hour grading window (warm it with 2-3 requests, then scale back). Report cold vs warm latency separately and set a >=180s client timeout. Document a warm-up call in the README.

**Why:** For a demo idle ~95% of the time, cost dominates, so scale-to-zero is right. But the v1 '5-15s cold start' was verified optimistic: a truly cold worker after scale-to-zero re-pulls the container, inits vLLM, allocates KV cache and captures CUDA graphs, realistically tens of seconds to ~2 min. Relying on cold-start magic risks a 2-minute hang that reads as 'broken' to a grader. min-workers=1 for the review window costs cents and removes the risk; honest separate latency reporting turns the number into an expected POC finding.

### Decision 6: Thin FastAPI wrapper + thinking-mode handling (expanded)

**Options:** No wrapper: vLLM built-in server, inject prompt and enable_thinking=False client-side · Thin FastAPI /triage wrapper that injects the safety system prompt, ALWAYS forces chat_template_kwargs enable_thinking=False, and writes the GDPR audit log

**Recommendation:** Add the thin FastAPI wrapper. Force enable_thinking=False in TWO places (server via --default-chat-template-kwargs on vLLM >=0.9.0, AND in the wrapper per request), and add an eval probe asserting no <think> appears in output.

**Why:** The safety prompt must not be optional and the audit log is part of the GDPR/traceability grading story; the wrapper also gives mockable code for CI. Verified that enable_thinking=False is a request/serve-time flag (not solved at training time) and that the merged model's copied chat_template still defaults to thinking-ON, so without defense-in-depth a forgotten flag makes the live demo emit visible <think> reasoning, balloon latency, and hide content from the disclaimer/escalation checks.

### Decision 7: CI/CD scope so the pipeline matches what is actually deployed (NEW)

**Options:** Build+push a model Docker image to GHCR and auto-deploy it to RunPod from GitHub Actions · CI lints+tests+builds the FastAPI WRAPPER image to GHCR; the model is served by stock worker-vllm via env vars; RunPod redeploy is a manual documented step (optional live smoke behind workflow_dispatch)

**Recommendation:** Adopt the wrapper-only CI with a manual, documented RunPod redeploy as the PLAN (not a contingency). Optionally add a workflow_dispatch post-deploy smoke that hits the live URL and asserts HTTP 200 + non-empty text.

**Why:** The v1 plan built a GHCR image while serving via the stock worker-vllm configured by env vars, so the image was never used by the endpoint, a deploy step that is theater and fails inspection. Building/testing only the wrapper (the one piece of custom code) is the honest, complete, green pipeline. Auto-deploy-to-serverless is the most fragile, least-documented link and a near-certain rabbit hole for a deliverable that only needs to 'exist'; a manual curl redeploy plus screenshot satisfies it at zero risk.

### Decision 8: Experiment tracking: W&B vs MLflow vs none

**Options:** Weights & Biases (one-line report_to='wandb', hosted dashboards) · MLflow (open-source, needs a tracking server/store on ephemeral notebooks) · None / CSV (zero dependency)

**Recommendation:** Weights & Biases, with report_to='none' + screenshots as the fallback if auth is a hassle. Log the W&B run id in the report and pair it with a committed pip-freeze lockfile, set_seed(3407), a tagged HF model revision, and the dataset SHA-256 for reproducibility.

**Why:** Lowest setup on ephemeral Kaggle sessions and it directly produces the SFT-loss and DPO reward-margin plots the report needs; it is the canonical tracker in the Unsloth/TRL examples. MLflow's server requirement is extra friction for no POC benefit. The red-team's reproducibility gap is closed cheaply by logging the run id alongside the lockfile, seed, model revision, and dataset hashes rather than by changing trackers.
