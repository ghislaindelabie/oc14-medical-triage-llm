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

**Reading (honest — corrected after checking the literature):**
- The result is **consistent with** the brief's Base rationale (clean slate → cleaner SFT signal for a
  strict custom format; the literature favours Base specifically when a specialised output format is
  needed), but it is **not proof** — for two reasons:
  - **Confound:** we trained *and* served the Instruct model with our plain ChatML, **overriding its native
    template**. Template mismatch is a documented, often-large performance hit for instruct models, so part
    of the gap is plausibly "we mistreated Instruct," not "Base is inherently better."
  - **Underpowered:** n=6 → Base 0.67 vs Instruct 0.33 is a **2-vignette** difference, within noise.
  - Note: the *common* community default is actually that **Instruct is the better starting point**; Base
    wins mainly for custom-format/neutral cases (which is ours). So our result is in the minority direction
    → the confound + small-n are the most credible drivers.
  - The Instruct failures were mostly **format breakdowns** (no parseable urgency level), consistent with
    its instruct priors reasserting at inference.
- DPO and Instruct-SFT both underperformed Base SFT → **Base SFT remains the served deliverable** (best measured).
- **Clean test to settle it (future work):** re-run Instruct on its **native** Qwen3 template + a larger eval set.
- Sources: Ithy (base-vs-instruct), arXiv 2411.02688 (instruction-tuning forgetting), arXiv 2406.14972
  (base vs instruct in RAG), Predibase / HF-forum (chat-template mismatch).

## Follow-ups (2026-06-22) — addressing the confound + the small eval
**(1) Instruct on its NATIVE template (controls the confound).** New arm `oc14-sft-instruct-native`:
keeps Qwen3-1.7B-Instruct's **native chat template** (no ChatML override), `enable_thinking=False`
(Qwen3's non-thinking format = an empty `<think></think>` wrapper, by design — verified locally). This
isolates "is Base really better?" from "did we mistreat Instruct?" (train_loss 0.830.)

**Result (n=6):** Instruct-native scored **0.50** acc, format **0.83**, **1/2** emergencies — up from
Instruct-ChatML (0.33 / 0.67 / 0/2). So the **template confound was real and material**: roughly half the
original Base-vs-Instruct gap was the forced ChatML. **But even on its native template, Instruct-SFT
(0.50, 1/2 emergencies) still trails Base-SFT (0.67, 2/2).** Full table:

| Model | template | urgency acc | emergencies | format | disclaimer |
|---|---|---|---|---|---|
| Base SFT | ChatML | **0.67** | **2/2** | 0.83 | 0.83 |
| Instruct SFT | ChatML (forced) | 0.33 | 0/2 | 0.67 | 0.67 |
| Instruct SFT | native | 0.50 | 1/2 | 0.83 | 0.67 |
| Base SFT + DPO | ChatML | 0.33 | 0/2 | 0.83 | 0.67 |

**Refined conclusion:** the Base-over-Instruct claim is **weakly supported** — Base still leads with the
confound controlled, but the margin (0.67 vs 0.50) is small and within n=6 noise. A firm statement needs
the n=500 eval (below). Base SFT remains the served deliverable (best measured, only one catching both
emergencies).

**(2) Larger eval set — solved.** `syntech-ai/medical-triage-500` (CC-BY-NC, **n=500**, synthetic, English,
never trained on) has a clean 3-class label that maps 1:1 to ours:
`immediate (230) → urgence maximale · urgent (195) → urgence modérée · routine (75) → urgence différée`
(its `risk_level` high/medium/low mirrors it). This gives a statistically meaningful held-out triage eval
(with per-class recall + confusion matrix), replacing the n=6 sanity check. **Caveats:** English-only (so
it tests cross-lingual generalisation of an FR-heuristic-trained model) and synthetic. Loads via pandas
(`hf_hub_download` the `.jsonl`; the HF auto-loader errors on its mixed schema). Eval harness to be built next.

## LLM-consensus triage labelling (2026-06-24) — a real eval-gold + train set

**Problem.** The n=6 sanity eval can't settle Base-vs-Instruct or measure triage skill, and no validated FR
triage gold exists (no clinician available). **Approach (settled with the user over several turns):** label
the **3,075 real MediQAl `clinical_case` vignettes** with a **3-model consensus** (OpenAI `gpt-5.4`,
Mistral `medium-3.5`, Anthropic `claude-sonnet-4-6`), each returning a **3-level urgency + ESI 1-5 in one
call** against a cited rubric (`docs/TRIAGE_CRITERIA.md`). Rationale: MCQA≠triage; LLM-as-annotator is a
**silver standard** (not gold); grounding via a rubric-in-prompt (full-RAG overkill for 3 levels); label
real cases rather than generate them; teacher(grounded LLMs)→student(no-RAG Qwen3) distillation.

**Method choices that mattered.** (a) **Rubric → cached system prefix.** Expanded to ~2 pages (2,817
Anthropic tokens: worked example/level, MTS presentation discriminators, atypical-presentation pitfalls,
the ESI 4-decision-point algorithm, vital danger zones, over-triage + non-clinical rules) and moved into
the `system` message so it's a byte-stable prefix → **prompt-cached** (OpenAI auto ≥1024; Anthropic
`cache_control` ≥2048; Mistral REST uncached). Counter-intuitive: a bigger *cached* rubric is barely more
expensive than the lean uncached one, since the two priciest providers read it at ~0.1×. (b) **Vanilla
SDKs/REST, not LangChain** — no orchestration; the `mistralai` 2.5.0 SDK imports empty, so Mistral goes via
plain `urllib` REST. (c) Pipeline is **concurrent** (thread pool), **resumable** (skips labelled case_ids),
**cache-aware** in cost accounting. (d) **Sample-first gate:** a paid 200-case sample measured real cost +
caching + κ before committing the full run.

**Results (all 3,075 × 3 models).** Fleiss **κ ≈ 0.67 — *substantial* inter-model agreement** (0.678 on the
200 sample, 0.667 on the full run); the n=3 live-test κ≈0 was small-sample noise. **1,603 unanimous +
ESI-consistent gold (52%)**, urgency mix maximale 995 / modérée 452 / différée 156 (skews *maximale* — the
over-triage default + MediQAl's sick teaching vignettes). ~838 (27%) excluded as non-clinical by consensus
(279 unanimous). **Caching confirmed:** OpenAI 73% / Anthropic 91% of input tokens served from cache.
**Total cost $36.67** (sample $2.53 + full $34.14) — *under* the $38.90 extrapolation, as the hit-rate
improved at scale. gpt-5.4 emitted ~81 output tok/call (no reasoning-token blow-up).

**Deliverable.** `build` → **`data/processed/triage_eval_gold.jsonl`** (300 held-out gold: case +
gold_urgency + gold_esi) and **`triage_sft_train.jsonl`** (2,496 rows = leftover gold + majority cases,
rendered in the triage response structure). A statistically meaningful, κ-backed eval set — what the n=6
check could not be.

**Honest limitations for the report.** Silver standard, not clinical validation. LLM↔clinician triage
agreement is only *moderate* in the literature; mitigated via consensus + clear-case gold + over-triage.
Source cases are real French *exam* vignettes (good provenance, exam-style). Gold is class-imbalanced toward
*maximale* (47% of the triaged pool / 62% of gold, vs ~25–30% in a real ED) — **but an over-triage ablation
(n=100, relabelled with the tie-breaker removed) shifted the *maximale* rate only ~2 pp (50%→48%; 5/100
cases changed level), so the skew is corpus-driven (MediQAl exam vignettes over-represent serious pathology),
not a rubric-caution artifact** — consistent with the genuinely-critical gold cases, 3-model agreement, and
the near-identical skew in the independent `medical-triage-500` (46% immediate). A production system would
need a representative, prospectively-collected ED triage dataset; ours is a defensible PoC proxy. **Mitigation
in place:** eval-gold is **stratified 100/100/100** with **macro-F1** as the headline metric (raw accuracy is
gameable by over-predicting *maximale*); the *training* set is kept natural for the first SFT retrain
(under-triage is the more dangerous error), to be measured on the eval and rebalanced only if a class collapses.

**Next.** Retrain SFT (Base) on the LLM-labelled train set; eval on the 300 gold (per-class recall +
confusion matrix). Then a fair **second DPO attempt** with **triage preference pairs** (chosen = safer
consensus, rejected = an under-triaged answer — flagged-disagreement cases are a natural source). This
addresses the earlier DPO failure, whose root cause was **off-task data composition** (~99% UltraMedical
verbosity vs 11 triage/safety pairs), **not** insufficient pair count.

## SFT retrain on LLM-consensus labels (2026-06-25) — the first defensible eval

**Train.** Combined set (`scripts/build_retrain_sft.py`): dropped the 1,415 old heuristic triage rows
(`mediqal_triage` = same vignettes, now contradictory), kept the medical-QA + EN breadth, added the 2,346
LLM-consensus triage rows → **5,931 train / 572 val**. Kaggle T4, full 2 epochs, **train_loss 0.869**
(old heuristic-trained SFT was 0.845 — same convergence; the question is whether *better labels* help on a
*real* eval).

**Eval on the stratified 300-case gold (100/100/100), macro-F1 headline:**

| metric | value |
|---|---|
| **macro-F1** | **0.813** |
| macro-precision / macro-recall | 0.816 / 0.813 |
| accuracy | 0.813 |
| **recall urgence maximale (safety)** | **0.93** |
| Cohen's κ vs gold | 0.72 (substantial) |
| per-class F1 | maximale **0.885** · modérée 0.723 · différée 0.83 |
| behavioural | disclaimer 1.00 · format 1.00 · no-`<think>` 1.00 |

Confusion (gold→pred): maximale 93 ok / **7→modérée / 0→différée**; modérée 73 ok / 17→maximale / 10→différée;
différée 78 ok / 22→modérée / 0→maximale.

**Reading.** (1) **Strong, balanced** — macro-F1 0.81 means it's good across *all three* classes, not gaming
the maximale prior (raw accuracy = macro here because the eval is balanced). (2) **Safety holds** — 93%
emergency recall, and the 7% missed emergencies drop only **one level (to modérée), never to différée** — the
*safe* failure mode. (3) **Over-triage lean** — modérée→maximale (17) and différée→modérée (22) outnumber the
reverse, consistent with the corpus + over-triage default; clinically the preferred bias. (4) **No
degeneration** — perfect format/disclaimer/no-`<think>`, confirming the inference config (stop on `<|im_end|>`,
trained system prompt, `enable_thinking=False`). (5) Weakest class is **modérée** (F1 0.72) — the middle class
is inherently ambiguous (confused both up and down), as expected.

**Caveat / open comparison.** This is **not** an apples-to-apples "LLM labels beat heuristic" claim yet — the
old heuristic SFT was only eval'd on n=6. A clean head-to-head requires re-running the **old adapter**
(`models/sft-base-lora/`, local) on this same 300-gold. Pending (optional). Regardless, this n=300 macro-F1
0.81 is the **first statistically meaningful eval** of the project — the deliverable the n=6 sanity check
could not be.

**Next.** Optional old-vs-new head-to-head on the 300-gold; then the DPO preference-pair step (triage
chosen/rejected from disagreement cases) → eval → compare SFT vs SFT+DPO.

## Corrected re-run (2026-06-25, post-audit) — the honest headline supersedes 0.81

After the adversarial audit (`docs/KNOWN_ISSUES.md`) the data + eval were fixed: training is now leak-free
(E2: 66 eval-gold-leaked QA rows dropped) and consensus-clean (E1: 1,193 flagged/3-way/ESI-inconsistent
cases excluded — train 4,687 = 1,119 clean LLM-triage + 11 vignettes + 3,557 QA), and the eval is now
**deterministic (greedy), batched (84 min → 11 min), verdict-line-anchored, with Wilson CIs**.

**Corrected eval (greedy, leak-free, stratified 300):**

| metric | value | 95% CI |
|---|---|---|
| **macro-F1** | **0.653** | — |
| macro-precision / recall | 0.79 / 0.68 | — |
| recall *urgence maximale* (safety) | **0.91** | **[0.84, 0.95]** |
| recall *urgence modérée* | 0.85 | [0.77, 0.91] |
| recall *urgence différée* | **0.28** | [0.20, 0.38] |
| behavioural (disclaimer/format/no-think) | 1.00 | — |

Confusion: maximale 91 ✓ / 9→modérée / **0→différée**; modérée 85 ✓ / 15→maximale / **0→différée**;
différée 28 ✓ / **72→modérée** / 0→maximale.

**Reading — why 0.81 → 0.65, and why this is the better number.** The 0.81 was inflated by (a) the
eval-gold→train leak, (b) sampled (non-deterministic) decoding that occasionally landed on the right rare
class, and (c) noisy non-consensus rows that spuriously propped up *différée*. Greedy + leak-free reveals
the truth: the model **never under-triages** (0 maximale/modérée → différée — maximally safe) but **systematically
over-triages low-acuity** (*différée* recall collapses to 0.28; 72/100 pushed up to *modérée*). This is the
exact over-triage failure mode predicted from the class imbalance — and the audit's "rebalance only if a
class collapses" trigger is now met. **Root cause of the collapse:** E1 (correctly) removed the noisy
non-consensus rows, but *différée* cases are disproportionately the ambiguous/flagged ones, so the clean
train is now *différée*-starved (gold itself is only ~10% différée). Cleaner labels, but a class starved of
signal.

**Safety framing (unchanged, now with a CI):** maximale recall 0.91, **conservative floor 0.84** — still
**unacceptable for autonomous ICU/ED triage** (≥1-in-10 emergencies missed at the lower bound). Decision-support
/ human-in-the-loop only; the value is the *method* + the *progress signal*, not a deployable autonomous triager.

**Next.** (1) **naive-Base baseline** on the same 300-gold (greedy) — the honest progress floor. (2) Address the
*différée* collapse: rebalance/augment the *différée* training signal (oversample, or relax E1 to keep
majority-différée even when one rater dissents) and/or use **DPO** to penalise over-triage of low-acuity.
(3) Re-eval + compare.

## différée-restored retrain (2026-06-25) — honest macro-F1 **0.82**

Applied the data fix: **relaxed E1 to `n_agree>=2`** (keep legitimate 2-of-3 majorities — restores the
ambiguous low-acuity signal; excluded only 347 true no-majority splits vs 1,193 before) and
**oversampled the 11 hand-written vignettes ×8** (the main EN-triage + balanced-*différée* exemplars).
Train triage now 2,041 rows (1,098 max / 689 mod / 254 diff, vs near-zero *différée* before). Same
greedy/batched/leak-free eval (now also **None-safe** after the baseline exposed a confusion-sort crash).

| v9 eval (greedy, leak-free, stratified 300) | recall (95% CI) | precision |
|---|---|---|
| urgence maximale | 0.90 [0.83, 0.95] | 0.88 |
| urgence modérée | 0.85 [0.77, 0.91] | 0.69 |
| urgence différée | **0.71 [0.62, 0.79]** | 0.95 |
| **macro-F1 0.822** · macro-P 0.84 · macro-R 0.82 · acc 0.82 · Cohen κ 0.73 · behavioural 1.00 | | |

Confusion: maximale 90 ✓ / 9→modérée / **1→différée**; modérée 85 ✓ / 12→maximale / 3→différée;
différée **71 ✓** / 29→modérée / 0→maximale.

**Reading.** The data fix worked: *différée* recall **0.28 → 0.71**, macro-F1 **0.65 → 0.82** — and this
0.82 is *honest* (leak-free, deterministic), unlike the retracted, inflated 0.81. **Cost (the
precision↔safety trade):** restoring *différée* re-introduced a little under-triage — **1 maximale→différée**
(one emergency to the lowest level) + 3 modérée→différée, where the over-cautious 0.65 model had zero. That
single critical under-triage is the precise target for the next step.

**Baseline (progress floor).** Naive **Qwen3-1.7B-Base, no fine-tuning**, same 300-gold/harness:
**macro-F1 0.19** (acc 0.25; recall maximale 0.70 / modérée 0.05 / différée 0.00). It defaults to
"maximale" or **fails to produce a usable level on ~32%** (96 `(none)`), never differentiates the lower
classes, and never emits the disclaimer (format 0.68 / disclaimer 0.00). So **fine-tuning bought
0.19 → 0.82 macro-F1** and taught the format + safety disclaimer from scratch (0.68/0.00 → 1.00/1.00) —
the honest, dramatic progress signal.

**Next.** **cost-weighted DPO** (chosen=correct, rejected=wrong *either direction*, extra weight on
under-triage; sources = the 11 clear-cut hand-written safety pairs + unambiguous under-triaged red-flag
cases — NOT the ambiguous 2-1 splits) to drive the lone maximale→différée back to 0 while keeping
*différée* recall.

## DPO attempt #2 (2026-06-26) — instructive negative; ship SFT v9

Built a **direction-balanced** triage-preference set (`scripts/build_dpo_pairs.py`): chosen = correct
gold level, rejected = a wrong adjacent level, same generic justification on both sides; ~2:1
under:over (under 112 + 11 safety / over 56 / modérée 56 mixed), 211 train / 24 val, eval-gold excluded.
DPO on the v9 SFT adapter (1 epoch), merged, scored on the same 300-gold.

| n=300, greedy | macro-F1 | maximale R | modérée R | différée R | κ |
|---|--:|--:|--:|--:|--:|
| **SFT v9** | **0.822** | 0.90 | **0.85** | 0.71 | 0.73 |
| SFT + DPO | 0.799 | 0.92 | **0.55** | 0.96 | 0.72 |

DPO confusion: maximale 92 ✓ / 7→mod / **1→diff**; modérée 55 ✓ / **20→max / 25→diff**; différée 96 ✓ / 4→mod / 0→max.

**Reading.** DPO sharpened the *extremes* (*différée* recall 0.71→**0.96**, maximale 0.90→0.92) but
**collapsed the middle** (*modérée* 0.85→**0.55**), net macro-F1 **0.82→0.80**, and did **not** fix the lone
maximale→différée. **Mechanism (design lesson):** *modérée* is the `rejected` level for BOTH the
maximale-under and différée-over pairs, so it appeared as "the wrong answer" ~168× vs "right" 56× → DPO
learned to **avoid the middle class**. Adjacent-level preference pairs structurally hammer the middle; a
future fix would reject the *far* level or balance modérée's chosen/rejected count. **Decision:** the
brief's SFT→DPO arc is demonstrated with an honest result; **SFT v9 (macro-F1 0.82) is the served
deliverable** (best balanced; DPO traded the middle for the extremes with no net gain).

**Next.** Serve SFT v9: merge to 16-bit → FastAPI `/triage` wrapper (smoke-test on P710 CPU) →
RunPod serverless vLLM (model via private HF repo) → CI deploy → Presidio/GDPR pass → report.
