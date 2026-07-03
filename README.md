# OC14 — Medical Triage Assistant (POC)

Proof-of-concept of a **bilingual (FR/EN) medical-triage agent** for the emergency department of a
fictional hospital (CHSA), built by specializing **Qwen3-1.7B-Base** with **SFT + LoRA**, then **DPO**,
served on **vLLM** and exposed through a **FastAPI** agent with a **Gradio** demo UI.

> OpenClassrooms AI-Engineer training project #14 ("Fine-tune your own LLM"). The **core task is triage** —
> prioritize urgency (*maximale / modérée / différée*), justify it clinically, recommend an action — not
> general medical Q&A.

**Headline result** (honest eval: stratified gold, n=300, greedy, leak-free): macro-F1
**0.19 (untrained Base) → 0.82 (SFT v9, served)**. A **DPO** pass with rebalanced preference pairs +
`rpo_alpha` further lifts it **0.827 → 0.845** like-for-like, with no class collapse (documented and
reproducible; v9 is the model served in V1). Positioned as **human-in-the-loop decision support**,
**not** an autonomous triage system.

## Links

| | |
|---|---|
| 📄 **Technical report** (PDF, 15 p, FR) | [`docs/RAPPORT_FR.pdf`](docs/RAPPORT_FR.pdf) · deliverables index: [`LIVRABLES.md`](LIVRABLES.md) |
| 🩺 **Live demo** (full agent) | [HF Space](https://ghislaindelabie-oc14-triage-demo.hf.space/) — real v9 model via RunPod; first request ~1–2 min if the endpoint is asleep (scale-to-zero), then ~2 s |
| 🧠 **Served model** (SFT v9) | [ghislaindelabie/oc14-qwen3-1.7b-triage-sft](https://huggingface.co/ghislaindelabie/oc14-qwen3-1.7b-triage-sft) |
| 📊 **Experiment tracking** (W&B) | [oc14-triage-eval](https://wandb.ai/ghislaindelabie/oc14-triage-eval) — eval-arm comparison + the LoRA hyperparameter sweep |

## Code map

The code is a Python package under `src/oc14_triage/`, organized by pipeline stage.

| Area | Path | What it does |
|---|---|---|
| **Data** | `data/{sources,download,build_sft,build_dpo,vignettes,templates,card}.py` | Source download, SFT/DPO dataset build, hand-written triage vignettes, GDPR data card |
| **Labeling** | `labeling/{clients,rubric,cases,aggregate,run}.py` | 3-LLM consensus labeling (ESI/MTS rubric, Fleiss κ) |
| **Anonymization** | `anonymization.py` | GDPR boundary — Presidio + bilingual spaCy + custom FR NIR recognizer + hashing |
| **Agent** | `agent/` | The LangGraph chain (detailed below) |
| **Eval** | `eval/metrics.py` | Triage-first metrics: `extract_urgency`, macro-F1, Wilson CI, confusion matrix |
| **Serving** | `serving/app.py` (+ `Dockerfile`) | FastAPI `/triage` wrapper in front of vLLM |
| **Experiments** | `runpod/{launcher,sweep,results}.py` | RunPod GPU-pod launcher, W&B sweep config, DPO-pair balance checks |
| **Config** | `config.py` | System prompts, 3-level urgency taxonomy, model IDs |

The **agent** (`src/oc14_triage/agent/`):

| File | Role |
|---|---|
| `graph.py` | The LangGraph `StateGraph` — 6 nodes (anonymize → pretreat → triage → explain → persist → SIH) + the single red-flag safety override. **Start here.** |
| `questionnaire.py` | Adaptive rule-based intake (motif → onset → severity + one red-flag follow-up) + `detect_red_flags` |
| `validation.py` | Deterministic input-sanity guardrail (rejects gibberish before triage) |
| `backend.py` | LLM call — `stub` (CI/demo, no GPU) \| real vLLM; response parsing |
| `state.py` · `sih.py` · `store.py` | `TriageCase` state + `RED_FLAGS` · FHIR R4 mock · SQLite traceability dossier |
| `service.py` · `ui.py` | FastAPI service (`/session/*`, `/trace`) · Gradio demo UI |

Other top-level dirs: `tests/` (pytest, TDD — the LLM is mocked), `notebooks/` (Kaggle SFT/DPO/eval — where
the GPU runs), `scripts/` (dataset build, W&B logging, RunPod experiments), `deploy/hf-space/` (Space source
+ auto-deploy), `docs/` (report + research notes), `.github/workflows/` (CI + Space deploy).

## Quickstart

Requires [`uv`](https://github.com/astral-sh/uv). No GPU is needed for data prep, tests, or the stubbed agent.

```bash
uv sync            # create the isolated venv
uv run pytest      # run the test suite (LLM mocked)
```

### Reproduce the dataset (CPU)

The shipped dataset lives in `data/kaggle_upload/`. To rebuild it from sources:

```bash
uv run python -m oc14_triage.data.download   # download sources → data/raw/
uv run python scripts/build_retrain_sft.py   # → data/kaggle_upload/sft_{train,val}.jsonl (3-LLM labelled)
uv run python scripts/build_dpo_pairs.py     # → data/kaggle_upload/dpo_{train,val}.jsonl (balanced triage pairs)
uv run python -m oc14_triage.data.card       # → data/cards/DATA_CARD.md
```

Training + evaluation run on **Kaggle** (free T4) from `notebooks/` (SFT LoRA → merge → DPO → evals);
seed **3407** throughout for reproducibility.

### Run the agent locally

The agent runs with a deterministic **stub** backend (no GPU/model required) or against a real vLLM endpoint.

```bash
# 1) API — stub backend, no model needed
OC14_TRIAGE_STUB=1 uv run uvicorn oc14_triage.agent.service:app --port 8080

# 2) Gradio demo UI (in another shell), pointed at the API
AGENT_SERVICE_URL=http://localhost:8080 uv run python -m oc14_triage.agent.ui
# → open http://localhost:7860
```

To use the **real model**, drop `OC14_TRIAGE_STUB` and set the vLLM endpoint variables
(`VLLM_BASE_URL`, `VLLM_API_KEY`, `OC14_MODEL_ID`) — see `serving/` and report §6.

## Serving & deployment

- **Model** — vLLM (OpenAI-compatible API) on **RunPod serverless** (scale-to-zero), model pulled from the
  public HF repo. A FastAPI `/triage` wrapper (`serving/app.py`, `serving/Dockerfile`) injects the system
  prompt, forces non-thinking, and stops on `<|im_end|>`.
- **Demo** — the full agent is deployed as a **Hugging Face Space** (`deploy/hf-space/`).
- **CI/CD** (`.github/workflows/`) — every push runs `ruff` + `pytest`; pushes to `main` touching the runtime
  **auto-deploy the Space** (`deploy-space.yml`, serialized with cancel-in-progress). Reproducible builds:
  seed 3407, `uv.lock`, versioned Kaggle notebooks.

## Status

All OpenClassrooms deliverables are complete — bilingual dataset, fine-tuned model (SFT v9 served; DPO
corrected with `rpo_alpha`), cloud vLLM endpoint + live demo, CI/CD, and the technical report. Positioned as
**decision support (human-in-the-loop)**, not autonomous triage. Full deliverables index and packaging:
[`LIVRABLES.md`](LIVRABLES.md). Public repo: <https://github.com/ghislaindelabie/oc14-medical-triage-llm>.
