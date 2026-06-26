# OC14 serving

Two pieces: **(A)** the model on **vLLM** (OpenAI-compatible API), and **(B)** a thin **FastAPI wrapper**
(`src/oc14_triage/serving/app.py`) that injects the trained triage system prompt, forces non-thinking
output (stop on `<|im_end|>`), gates on an API key, and writes a privacy-safe audit log (metadata only).

## Model
Served model = **SFT v9** (macro-F1 0.82), merged to 16-bit by the `oc14-sft-merge` kernel
(`/kaggle/working/sft_merged_16bit`). The merged weights load in vanilla vLLM with no special flags;
the ChatML template + `<|im_end|>` stop are baked into the saved tokenizer.

## A — vLLM backend
**RunPod serverless** (recommended — scale-to-zero) with the official `worker-vllm` template, model
pulled from a **private HF repo**. Launch flag for non-thinking: `--default-chat-template-kwargs
'{"enable_thinking": false}'` (harmless here — our template ignores it — but kept per Decision-H).
Local equivalent (GPU box): `vllm serve <model> --served-model-name oc14-triage`.

## B — FastAPI wrapper
Local dev against any vLLM backend:
```bash
uv sync --extra serving
VLLM_BASE_URL=http://localhost:8000/v1 OC14_MODEL_ID=oc14-triage \
  uv run uvicorn oc14_triage.serving.app:app --port 8080
curl -s localhost:8080/triage -H 'content-type: application/json' \
  -d '{"text":"Homme 60 ans, douleur thoracique constrictive et sueurs depuis 20 min.","lang":"fr"}'
```
Container: `docker build -f serving/Dockerfile -t oc14-triage-wrapper .` (see the Dockerfile header).

Env: `VLLM_BASE_URL`, `OC14_MODEL_ID`, `OC14_API_KEY` (if set → required via `X-API-Key`),
`VLLM_API_KEY` (backend key, default `EMPTY`).

## CI/CD
`.github/workflows/` builds + pushes the wrapper image to GHCR; RunPod redeploy is a documented manual
step (+ optional `workflow_dispatch` live smoke). *(deploy step pending the RunPod credential)*
