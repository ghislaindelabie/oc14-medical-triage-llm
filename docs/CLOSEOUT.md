# Project close-out — OC14 Medical Triage LLM

**Status: CLOSED — 2026-07-07** (portfolio state, git tag `v2`). This file is the single entry point to
**reopen, reuse, or write about** the project. Deep detail lives in `RAPPORT_FR.md` (+ PDF), the dev journal,
`KNOWN_ISSUES.md`, and `data/cards/`.

## 1. One-liner
Fine-tuned **Qwen3-1.7B-Base** into a **medical-triage assistant** inside a full agentic pipeline
(intake → GDPR anonymization → triage → explanation → traceability → FHIR output). **macro-F1 0.19 → 0.82**
(n=300 held-out gold); critical-urgency recall 0.90 [0.83–0.95]. Positioned as **human-in-the-loop decision
support**, not autonomous. Public repo: `github.com/ghislaindelabie/oc14-medical-triage-llm` (tag `v2`).

## 2. Services — state as left (CLOSED)
| Service | State | Cost |
|---|---|---|
| RunPod serverless endpoint `hb7hk0khg7ekaw` (`oc14-triage`) | **scale-to-zero** (`workersMin=0`, 0 active workers) — still exists, reopenable instantly | $0 idle (balance ~$4.37) |
| HF Space `ghislaindelabie/oc14-triage-demo` | **PAUSED** | $0 (cpu-basic is free anyway) |
| RunPod GPU pods | none | $0 |

## 3. Reopen guide
### Live demo (HF Space)
- **Resume:** `HfApi(token=$HF_TOKEN).restart_space("ghislaindelabie/oc14-triage-demo")` — or Space UI → *Settings → Restart*.
- Space variables are already correct (`OC14_MODEL_ID` = the **full HF id**, `VLLM_BASE_URL`, `OC14_MODEL_VERSION=sft-v9`, `VLLM_TIMEOUT=10`). App runs FastAPI on `127.0.0.1:8091` (internal) + Gradio `:7860` (public) → **Swagger `/docs` is local-only**.

### Serving endpoint (RunPod) — warm it for a smooth demo
Scale-to-zero means the first request cold-starts (>10 s → the app shows an honest "model starting" fallback,
**not** a fabricated verdict). To warm one always-on worker (~$0.25/hr), then revert after:
```bash
# WARM (workersMin 0->1)
curl -s -X POST "https://api.runpod.io/graphql?api_key=$RUNPOD_API_KEY" -H "Content-Type: application/json" -H "User-Agent: curl/8.5.0" \
  -d '{"query":"mutation { saveEndpoint(input: {id:\"hb7hk0khg7ekaw\", name:\"oc14-triage\", templateId:\"b0ha4wn5aa\", gpuIds:\"AMPERE_16,AMPERE_24,ADA_24\", workersMin:1, workersMax:1, idleTimeout:120, scalerType:\"QUEUE_DELAY\", scalerValue:4}) { id workersMin } }"}'
# REVERT (workersMin -> 0) — change workersMin:1 to workersMin:0 above.
# HEALTH: curl -s https://api.runpod.ai/v2/hb7hk0khg7ekaw/health -H "Authorization: Bearer $RUNPOD_API_KEY"
```
**Gotcha:** vLLM serves under the **full HF id** `ghislaindelabie/oc14-qwen3-1.7b-triage-sft`; sending
`oc14-triage` (the backend default / local `.env`) → HTTP **500**. RunPod GraphQL needs the `curl/8.5.0` UA (WAF).

### Local run (no GPU, stub backend)
```bash
OC14_TRIAGE_STUB=1 uv run uvicorn oc14_triage.agent.service:app --port 8080   # API + Swagger at /docs
AGENT_SERVICE_URL=http://localhost:8080 uv run python -m oc14_triage.agent.ui  # Gradio :7860
```

## 4. Artifacts & locations
- **SFT v9 (served)** — HF **public** `ghislaindelabie/oc14-qwen3-1.7b-triage-sft`.
- **DPO adapter (`rpo_alpha`, +0.018)** — HF **private** `ghislaindelabie/oc14-qwen3-1.7b-triage-dpo-rpo`.
- **v10** — LoRA adapter on Kaggle `oc14-sft-lora-qwen3-1-7b/sft_adapter` (W&B run `oc14-sft-v10`).
  ⚠️ **The Kaggle `oc14-sft-merge` output is a STALE v9 merge** (byte-identical to served v9). **If deploying v10, re-run the merge on the v10 adapter first.**
- **Datasets** — `data/kaggle_upload/*.jsonl` (+ Kaggle dataset `oc14-triage-data`); cards in `data/cards/`.
- **W&B** — `wandb.ai/ghislaindelabie/oc14-triage-eval` (arms + `sft-v9+dpo-rpo`) · `.../oc14-sft-sweep`.
- **Kaggle kernels** — `oc14-sft-lora-qwen3-1-7b`, `-sft-merge`, `-sft-eval`, `-dpo`, `-dpo-eval`, `-base-baseline-eval`, instruct arms.
- **Report** — `docs/RAPPORT_FR.pdf` (FR, 15 p); deliverables index `LIVRABLES.md`.

## 5. Lessons learned — raw material for a delabie.tech blog article
*(Method / approach / pitfalls — ready to turn into a "what fine-tuning a small model for a real task actually taught me" post.)*
- **Honest evaluation is a feature.** I *retracted my own* inflated 0.81 after an adversarial audit found an
  eval→train leak + sampled (non-deterministic) decoding. The defensible 0.82 (greedy, leak-free, Wilson CIs) replaced it.
- **The middle-class collapse (twice).** (a) Cleaning non-consensus rows *starved* the low-acuity class →
  systematic over-triage (recall 0.28); fixed by relaxing consensus to n≥2 + oversampling vignettes (→ 0.71).
  (b) DPO with adjacent-level pairs made the middle level the "rejected" side of *both* error directions →
  **likelihood displacement**, collapsing modérée recall 0.85→0.55.
- **DPO done right.** Balanced rejected distribution + **`rpo_alpha`** (an NLL anchor on the chosen answer) fixed
  the collapse → **+0.018 macro-F1 like-for-like, no collapse**. Lesson: the failure was *pair design*, not the method.
- **Base vs Instruct.** Base = cleaner SFT (no competing priors) but a chat-template confound nearly buried the
  comparison — n=6 sanity checks lie; n=300 settled it.
- **Silver labels.** A 3-LLM consensus (Fleiss κ 0.67) is a pragmatic stand-in for clinician gold — with an
  explicit circularity caveat (the student imitates the teachers; the gold is the *easy* unanimous subset).
- **Deterministic guardrails > weight-alignment for safety.** A red-flag override + an OOD input guardrail
  (gibberish → the model *confabulates* an urgent verdict) are the deterministic half of a hybrid system.
- **Serving pitfalls.** Serverless cold-start vs a short client timeout → "model unavailable"; model-name
  mismatch → 500; RunPod's WAF blocks the default urllib UA.
- **MLOps discipline.** v10 reached parity → don't redeploy on faith; I A/B-tested the "better justifications"
  hypothesis with a blind LLM judge (n=48) → **statistical wash** → kept v9 on *evidence*.

## 6. Open items (if reopening)
- 🔐 **SECURITY — rotate** the `KGAT_` Kaggle token + the `OC14/.env` keys (OpenAI / Mistral / Anthropic / RunPod), pasted in chat during the project.
- If shipping v10: re-merge on the v10 adapter first (see §4 gotcha); its justifications showed **no** measured advantage over v9.
- Toward a credible pilot: clinician-validated labels, real prospective ED data, larger eval (n ≥ 500), calibrated confidence (logprobs).
