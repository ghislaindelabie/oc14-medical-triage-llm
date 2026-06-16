> **TL;DR — key takeaways**
>
> - vLLM's built-in server gives you an OpenAI-compatible HTTP API out of the box — for a POC you likely do NOT need a separate FastAPI layer unless you want to inject the triage system prompt, add a simple API key check, or attach structured logging.
> - For a single fine-tuned model, merging LoRA weights into the base model (merge_and_unload) then serving as ordinary 16-bit weights is simpler and more foolproof than runtime LoRA for a POC — runtime LoRA is worth it only when you need to hot-swap many adapters on one GPU.
> - The official Docker image vllm/vllm-openai (pinned to a version tag) is the right base; bake in your wrapper code but download model weights at container startup via a RunPod Network Volume or the built-in model-caching feature — do NOT bake 3+ GB of weights into the image.
> - RunPod Serverless with model caching reduces cold start for a 1.7B model to roughly 5–15 seconds from scratch, or a few seconds once the model is cached on the host; estimated cost per 1000 short requests is well under $0.10 on an L4 GPU.
> - A realistic GitHub Actions CI/CD pipeline for this POC has three stages: lint/test on every push, build and push Docker image on merges to main, and update the RunPod endpoint image via a GraphQL API call or the halbgut/runpod-serverless-deploy community action.
> - vLLM exposes Prometheus metrics (vllm:time_to_first_token_seconds, vllm:e2e_request_latency_seconds, vllm:generation_tokens_total) out of the box; the vllm bench serve CLI gives you TTFT, TPOT, and throughput numbers in one command for latency testing.
> - RunPod Serverless is the best fit for this POC: lowest raw GPU cost, good vLLM integration with the official worker-vllm template, and scale-to-zero avoids idle charges; Modal is developer-friendlier (pure Python, no Dockerfile required) but slightly more expensive per GPU-second; HF Inference Endpoints requires the least infrastructure knowledge but costs more and gives less control.


# Serving & Deployment Stack: vLLM + FastAPI + Docker + RunPod Serverless + GitHub Actions

> Audience: an engineer learning this domain for an OpenClassrooms AI-engineer project (medical triage POC, Qwen3-1.7B fine-tuned with LoRA, free training on Kaggle/Colab, serving on RunPod serverless, ~2-week timeline).

---

## Table of Contents

1. [vLLM's Built-In OpenAI-Compatible Server](#1-vllms-built-in-openai-compatible-server)
2. [Serving a Fine-Tuned Model: Merged Weights vs Runtime LoRA](#2-serving-a-fine-tuned-model-merged-weights-vs-runtime-lora)
3. [Do You Need a FastAPI Wrapper?](#3-do-you-need-a-fastapi-wrapper)
4. [Dockerfile Essentials](#4-dockerfile-essentials)
5. [Deploying on RunPod Serverless](#5-deploying-on-runpod-serverless)
6. [GitHub Actions CI/CD Pipeline](#6-github-actions-cicd-pipeline)
7. [Measuring Latency](#7-measuring-latency)
8. [Provider Comparison: RunPod vs Modal vs HF Inference Endpoints](#8-provider-comparison-runpod-vs-modal-vs-hf-inference-endpoints)

---

## 1. vLLM's Built-In OpenAI-Compatible Server

### What is vLLM?

**vLLM** is an open-source Python library designed specifically for fast LLM inference. Its key internal technique is **PagedAttention** — a memory management scheme that avoids wasting GPU memory by dynamically paging the KV cache (the working memory the model builds as it reads your prompt). The result is significantly higher throughput than naive serving, especially with concurrent users.

vLLM ships with a built-in HTTP server that implements the **OpenAI API spec**. "OpenAI-compatible" means any client that already knows how to talk to OpenAI (the official `openai` Python SDK, `curl`, LangChain, etc.) can point to your vLLM server by just changing one URL — no code changes needed.

### Starting the Server

```bash
vllm serve ./merged-model \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --api-key "your-secret-key"
```

Key flags explained:

| Flag | What it does | POC recommendation |
|---|---|---|
| `--host 0.0.0.0` | Listens on all interfaces (needed inside Docker) | Required |
| `--port 8000` | Default port | Keep default |
| `--dtype bfloat16` | Weight precision. `auto` also works but explicit is safer | Use `bfloat16` for Qwen3 |
| `--max-model-len 4096` | Maximum prompt + output tokens. Limits KV cache RAM use | Set lower (e.g. 2048) to save VRAM |
| `--api-key` | Bearer token required on every request — basic auth | Set even for a POC |
| `--tensor-parallel-size` | Split across multiple GPUs. Leave at default 1 for single-GPU | Leave at 1 |
| `--gpu-memory-utilization` | Fraction of VRAM reserved for the model (default 0.9) | Lower to 0.85 if OOM |

### What You Get Out of the Box

Once the server starts, these HTTP endpoints are immediately available with no extra code:

- `POST /v1/chat/completions` — chat interface (messages array)
- `POST /v1/completions` — raw text completion
- `GET /v1/models` — list loaded models
- `GET /health` — liveness probe
- `GET /metrics` — Prometheus metrics (latency, throughput, KV cache usage)
- `POST /v1/load_lora_adapter` / `POST /v1/unload_lora_adapter` — dynamic LoRA management (if `--enable-lora`)

Streaming is supported (Server-Sent Events) via `"stream": true` in the request body.

**Using it from Python:**

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="your-secret-key",
)

response = client.chat.completions.create(
    model="merged-model",   # must match --served-model-name or the path
    messages=[{"role": "user", "content": "J'ai une douleur thoracique"}],
    max_tokens=256,
)
print(response.choices[0].message.content)
```

Source: [vLLM CLI serve reference](https://docs.vllm.ai/en/stable/cli/serve/) | [OpenAI-Compatible Server docs](https://docs.vllm.ai/en/stable/serving/openai_compatible_server/)

---

## 2. Serving a Fine-Tuned Model: Merged Weights vs Runtime LoRA

### Background: What is LoRA?

**LoRA** (Low-Rank Adaptation) is the fine-tuning technique used in this project. Instead of updating all 1.7B parameters, it trains a small set of auxiliary matrices (the "adapter"). After training, you have two things: the original base model weights, and a small adapter file (~10–100 MB).

You have two options at serving time:

### Option A: Merge the Adapter into the Base Model (Recommended for POC)

After training, run `merge_and_unload()` (from the `peft` library) to permanently bake the adapter into the base weights, then save as a standard HuggingFace model directory. Serve it like any ordinary model:

```bash
vllm serve ./merged-qwen3-chsa \
  --dtype bfloat16 \
  --max-model-len 2048
```

No special flags needed. The weights are a single self-contained directory compatible with any HuggingFace tool.

**Pros:** Simple, portable, no vLLM-specific configuration, maximum compatibility, slightly faster inference (no adapter overhead at runtime).

**Cons:** You lose the ability to hot-swap adapters — but for a POC with one fine-tune, that does not matter.

### Option B: Runtime LoRA (Keep Adapter Separate)

vLLM supports loading LoRA adapters at serve time or even dynamically at runtime:

```bash
vllm serve Qwen/Qwen3-1.7B-Base \
  --enable-lora \
  --lora-modules chsa-triage=./lora-adapter-dir
```

Clients then select which adapter to use by passing `"model": "chsa-triage"` in the request. You can also load/unload adapters at runtime without restarting by setting `VLLM_ALLOW_RUNTIME_LORA_UPDATING=True` and hitting `POST /v1/load_lora_adapter`.

**When this is worth it:** if you need to run multiple task-specific fine-tunes on one GPU simultaneously (e.g., one adapter per medical specialty), runtime LoRA avoids duplicating base model memory. For a single fine-tune POC, this is unnecessary complexity.

### Recommendation

**Merge and serve as 16-bit weights.** It is the simpler, more foolproof path. The merged Qwen3-1.7B in `bfloat16` will be approximately 3.4 GB on disk (1.7B × 2 bytes), well within the 24 GB VRAM of an L4 GPU.

Source: [vLLM LoRA Adapters documentation](https://docs.vllm.ai/en/latest/features/lora/)

---

## 3. Do You Need a FastAPI Wrapper?

### The Short Answer: Probably Not, But There Are Good Reasons to Add One

vLLM's built-in server already handles authentication (`--api-key`), streaming, concurrent requests, and metrics. For a POC demo that is just being graded, you can go directly to the vLLM server.

**However**, a thin FastAPI wrapper is justified in two scenarios for this project:

1. **Injecting the triage system prompt automatically** — so callers just send the patient text and do not need to know the internal prompt format.
2. **Adding structured request logging** — for the GDPR traceability requirement (log who asked what without storing PHI in raw form).

### Minimal FastAPI Wrapper Pattern

If you add a wrapper, keep it lightweight. Run vLLM as a subprocess or in the same container on port 8001, and expose your FastAPI app on port 8000:

```python
# app/main.py
import httpx
import uuid
import logging
from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

VLLM_URL = "http://localhost:8001/v1/chat/completions"
API_KEY = "your-triage-api-key"
SYSTEM_PROMPT = (
    "Tu es un assistant de triage médical du CHSA. "
    "Réponds en français. Évalue la gravité (1-5) et recommande une orientation. "
    "Ne pose pas de diagnostic définitif."
)

logger = logging.getLogger("triage")
security = HTTPBearer()
app = FastAPI(title="CHSA Triage API")

class TriageRequest(BaseModel):
    patient_message: str
    max_tokens: int = 512

def verify_key(creds: HTTPAuthorizationCredentials = Security(security)):
    if creds.credentials != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

@app.post("/triage")
async def triage(req: TriageRequest, _=Depends(verify_key)):
    request_id = str(uuid.uuid4())[:8]
    logger.info({"request_id": request_id, "tokens": len(req.patient_message.split())})
    
    payload = {
        "model": "chsa-triage",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": req.patient_message},
        ],
        "max_tokens": req.max_tokens,
    }
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(VLLM_URL, json=payload)
        r.raise_for_status()
    
    result = r.json()
    logger.info({"request_id": request_id, "finish_reason": result["choices"][0]["finish_reason"]})
    return {"request_id": request_id, "response": result["choices"][0]["message"]["content"]}
```

### When to Skip the Wrapper

If the graders will call `/v1/chat/completions` directly with the OpenAI SDK (common in technical evaluations), the wrapper adds latency for no benefit. You can inject the system prompt in your demo notebook instead. Skip the wrapper and use `--chat-template` or a custom chat template file to bake in the system message at the vLLM level.

**GOTCHA:** vLLM also natively supports adding custom ASGI middleware via `--middleware your.module.MiddlewareClass`, so you can add logging without a separate process.

---

## 4. Dockerfile Essentials

### Use the Official Pre-Built Image as Base

The vLLM project publishes a ready-to-use image: `vllm/vllm-openai`. Do NOT build vLLM from source — it is complex and slow. Use a pinned version tag for reproducibility:

```dockerfile
FROM vllm/vllm-openai:v0.9.1
# ^^^ pin the version; "latest" changes without warning and can break things

# If you have a FastAPI wrapper:
COPY requirements-wrapper.txt /app/
RUN pip install --no-cache-dir -r /app/requirements-wrapper.txt
COPY app/ /app/

# Model is NOT baked in — downloaded at runtime
# This keeps the image small (~7 GB base image, vs 10+ GB with weights)

ENV HF_HOME=/runpod-volume/huggingface
# ^^^ point the HF cache to the RunPod Network Volume mount point

EXPOSE 8000

# If serving vLLM directly (no wrapper):
CMD ["vllm", "serve", "/runpod-volume/model", \
     "--host", "0.0.0.0", "--port", "8000", \
     "--dtype", "bfloat16", "--max-model-len", "2048"]
```

### Model Download: Build Time vs Runtime

**Build time (bake into image):** Simple, predictable. But: a 3.4 GB model makes your image ~10+ GB, slowing every push/pull by several minutes. Also, updating the model means rebuilding the entire image.

**Runtime (download on start):** The model is pulled from HuggingFace or loaded from a mounted Network Volume when the container starts. This is the standard pattern for RunPod:

- On first cold start: downloads ~3.4 GB (~2–4 minutes on fast connection, billed on RunPod)
- With RunPod Network Volume or model caching: loads from disk in ~5–20 seconds, **not billed during download**

**Recommendation for this POC:** use RunPod's built-in model caching feature (described in Section 5). No custom download script needed.

### Key Environment Variables

| Variable | Purpose |
|---|---|
| `HF_TOKEN` | Access gated HuggingFace repos (set in RunPod endpoint settings, never in Dockerfile) |
| `HF_HOME` | Override HF cache directory (point to `/runpod-volume/huggingface`) |
| `VLLM_WORKER_MULTIPROC_METHOD=spawn` | Prevents silent failures with multi-GPU tensor parallelism (safe to set even for single-GPU) |
| `CUDA_VISIBLE_DEVICES` | Restrict to specific GPUs (usually not needed on RunPod) |

### Docker Build and Push (for CI)

```bash
# Build
docker build -t ghcr.io/your-org/chsa-triage:$(git rev-parse --short HEAD) .

# Push to GitHub Container Registry (free for public repos)
docker push ghcr.io/your-org/chsa-triage:$(git rev-parse --short HEAD)
```

Sources: [vLLM Docker deployment docs](https://docs.vllm.ai/en/stable/deployment/docker/) | [RunPod vLLM Docker guide](https://www.runpod.io/articles/guides/deploy-vllm-runpod-docker)

---

## 5. Deploying on RunPod Serverless

### How RunPod Serverless Works

**Serverless GPU** means: no GPU is running when there are no requests. A worker is provisioned on demand, handles your request, and shuts down after an idle period. You pay only for the seconds the GPU is actually running.

Contrast this with a **dedicated pod** (always-on), which costs $0.19–$0.50/hour even when idle.

The RunPod serverless flow for each request:

```
Client request → RunPod queue
  → Is a warm worker available?
      YES → route immediately (~0 extra latency)
      NO  → Cold start: provision GPU, start container, load model
             → Process request
             → Idle timeout → worker shuts down
```

### Quick Deploy: Use the Official worker-vllm Template

RunPod maintains [`runpod-workers/worker-vllm`](https://github.com/runpod-workers/worker-vllm) — a pre-built image that wraps vLLM in the RunPod serverless handler format. For a POC, using this directly (or as your Dockerfile base) is the fastest path:

1. Go to RunPod console → Serverless → New Endpoint → Quick Deploy
2. Select model: `your-hf-username/chsa-triage-merged` (or a local path via Network Volume)
3. Set environment variables:
   - `MODEL_NAME=your-hf-username/chsa-triage-merged`
   - `MAX_MODEL_LEN=2048`
   - `DTYPE=bfloat16`
   - `HF_TOKEN=<your token>` (from RunPod secrets, not hardcoded)
4. GPU: L4 (24 GB, $0.00019/sec) is ideal for a 1.7B model — gives ample headroom

The worker auto-exposes an OpenAI-compatible API at `https://api.runpod.ai/v2/{endpoint_id}/openai/v1/`.

### Custom Handler Pattern (If You Need More Control)

If using a custom Docker image, the RunPod SDK expects a `handler` function:

```python
# handler.py
import runpod
from vllm import LLM, SamplingParams

llm = LLM(model="/runpod-volume/model", dtype="bfloat16", max_model_len=2048)

def handler(event):
    """Called by RunPod for each request."""
    inp = event["input"]
    prompt = inp.get("prompt", "")
    params = SamplingParams(
        temperature=inp.get("temperature", 0.3),
        max_tokens=inp.get("max_tokens", 512),
    )
    outputs = llm.generate([prompt], params)
    return {"response": outputs[0].outputs[0].text}

runpod.serverless.start({"handler": handler})
```

However, for an OpenAI-compatible endpoint, prefer the `worker-vllm` template which handles this for you.

### Model Caching: Reducing Cold Starts

RunPod's **model caching** feature (separate from Network Volumes) pre-stages model weights on host machines. When you specify a model ID in the endpoint configuration:

- RunPod downloads the model to `/runpod-volume/huggingface-cache/hub/` on select hosts
- Workers start preferentially on those cached hosts
- **You are not charged for the download time**
- Cold start shrinks from ~3 minutes (fresh download) to **5–20 seconds** (cached load)

For a 1.7B model at ~3.4 GB, cached cold start should be at the low end (~5–10 seconds).

Setup: in your endpoint config, add your model ID (e.g., `Qwen/Qwen3-1.7B-Base`) in the "Model" field. [RunPod model caching docs](https://docs.runpod.io/serverless/endpoints/model-caching).

### Cost Estimate

**GPU choice:** L4 (24 GB VRAM) at $0.00019/second = $0.684/hour.

For a 1.7B model at ~150 tokens/sec output:

- Typical triage response: 200 tokens → ~1.3 seconds of generation
- Add cold start amortized over N requests
- Add input prefill time: ~0.5 seconds for a 100-token prompt

**Warm request cost:** ~2 seconds × $0.00019 = **~$0.00038 per request**

**1000 warm requests:** ~$0.38

**With cold starts (worst case, all cold):** a 10-second cold start adds $0.0019 each → **~$2.28 per 1000 requests**

**With model caching and 60s idle timeout:** most requests after the first few hit a warm worker; realistic average including cold starts ~**$0.50–$1.00 per 1000 requests**.

**Idle cost:** zero — scale-to-zero means no charge when there are no requests.

### Cost Pitfall

If you set a long idle timeout (e.g., 600 seconds), your worker stays alive for 10 minutes after the last request. At $0.00019/sec, that is $0.114 per idle period. For a low-traffic POC demo, set a **shorter idle timeout (30–60 seconds)** to accept a cold start but avoid burning money while the grader reads your report.

Sources: [RunPod serverless pricing](https://docs.runpod.io/serverless/pricing) | [RunPod vLLM get started](https://docs.runpod.io/serverless/vllm/get-started) | [worker-vllm GitHub](https://github.com/runpod-workers/worker-vllm)

---

## 6. GitHub Actions CI/CD Pipeline

### What Is Realistic for a POC vs Overkill

**Worth doing (low effort, high value for the grade):**
- Lint (ruff/flake8) + basic unit tests on every push
- Docker image build + push on merge to main
- Automated deployment update to RunPod

**Overkill for a 2-week solo POC:**
- Integration tests against a live GPU (expensive, slow)
- Multi-environment staging/prod promotion
- Semantic versioning automation
- Full e2e load testing in CI

### Minimal Realistic Pipeline

```yaml
# .github/workflows/ci-cd.yml
name: CI/CD

on:
  push:
    branches: [main, "feature/**"]
  pull_request:
    branches: [main]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}/chsa-triage

jobs:
  lint-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: "pip"
      
      - name: Install dev dependencies
        run: pip install ruff pytest httpx
      
      - name: Lint
        run: ruff check .
      
      - name: Unit tests (no GPU needed)
        run: pytest tests/unit/ -v --tb=short
        # Unit tests mock the vLLM call; they test the FastAPI wrapper logic only

  build-push:
    needs: lint-test
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Log in to GHCR
        uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      
      - name: Build and push
        uses: docker/build-push-action@v5
        with:
          context: .
          push: true
          tags: |
            ghcr.io/${{ env.IMAGE_NAME }}:latest
            ghcr.io/${{ env.IMAGE_NAME }}:${{ github.sha }}
          cache-from: type=gha
          cache-to: type=gha,mode=max

  deploy-runpod:
    needs: build-push
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Update RunPod endpoint image
        uses: halbgut/runpod-serverless-deploy@v1   # community action
        with:
          api-key: ${{ secrets.RUNPOD_API_KEY }}
          endpoint-id: ${{ secrets.RUNPOD_ENDPOINT_ID }}
          image-name: ghcr.io/${{ env.IMAGE_NAME }}:${{ github.sha }}
```

### Required GitHub Secrets

Add in your repo Settings → Secrets → Actions:
- `RUNPOD_API_KEY` — from RunPod dashboard → Settings → API Keys
- `RUNPOD_ENDPOINT_ID` — from your serverless endpoint URL

### How the RunPod Deploy Step Works

RunPod's API does not have a simple REST endpoint to update image names. The `halbgut/runpod-serverless-deploy` community action ([lucaschmid.net/blog/runpod-deploy](https://lucaschmid.net/blog/runpod-deploy/)) uses a GraphQL mutation to update the template bound to your endpoint:

```graphql
mutation saveTemplate($input: SaveTemplateInput) {
  saveTemplate(input: $input) { id }
}
```

**Alternatively**, RunPod has a native GitHub integration (Settings → Serverless → GitHub) that triggers a rebuild from your repo on every new GitHub Release. This avoids writing the GraphQL call but requires creating a GitHub Release tag for each deployment rather than pushing to main. Either approach works for a POC.

### What the Lint/Test Job Actually Tests

Since the inference code itself cannot be unit-tested without a GPU, your tests should cover:
- The FastAPI wrapper: mock the `httpx.AsyncClient` call to vLLM and verify the system prompt is injected correctly
- Input validation: empty strings, excessively long inputs
- Auth: requests without a Bearer token return 401

This runs in ~30 seconds on a standard `ubuntu-latest` runner (free tier: 2000 min/month).

Source: [RunPod CI/CD guide](https://www.runpod.io/articles/guides/integrating-runpod-with-ci-cd-pipelines) | [RunPod GitHub integration](https://docs.runpod.io/serverless/workers/github-integration)

---

## 7. Measuring Latency

### Key Metrics Defined

| Metric | What it measures | Why it matters |
|---|---|---|
| **TTFT** (Time to First Token) | Time from sending the request to receiving the first token of the response | Perceived responsiveness — a user feels "the model is thinking" until this |
| **TPOT** (Time Per Output Token) | Average time between successive tokens after the first | Generation speed — lower is smoother streaming |
| **E2E latency** | Total time from request send to complete response | What matters for synchronous (non-streaming) calls |
| **Throughput** | Output tokens/second across all concurrent requests | Capacity metric — how many users you can serve simultaneously |

### Method 1: vLLM Built-in Prometheus Metrics

vLLM exposes these at `GET /metrics` (scraped by Prometheus, or just read raw):

```
vllm:time_to_first_token_seconds_sum / _count  → mean TTFT
vllm:e2e_request_latency_seconds_bucket        → latency histogram (p50, p95, p99)
vllm:generation_tokens_total                   → total tokens generated (rate = throughput)
vllm:num_requests_running                      → concurrency
```

For a quick POC check, just `curl http://localhost:8000/metrics | grep ttft`.

### Method 2: vLLM Benchmark CLI (Recommended for the Report)

```bash
# Install vLLM and run against a live server:
pip install vllm

vllm bench serve \
  --backend openai-chat \
  --model chsa-triage \
  --endpoint /v1/chat/completions \
  --dataset-name random \
  --num-prompts 50 \
  --request-rate 2 \
  --output-json bench_results.json
```

This outputs:
```
Request throughput (req/s): 1.73
Output token throughput (tok/s): 382.89
TTFT (mean/median/p99): 0.45s / 0.42s / 0.78s
TPOT (mean): 0.008s
```

For the report, run with `--num-prompts 100 --request-rate 1` (single user, representative of POC demo) and report mean TTFT and mean E2E latency.

### Method 3: Simple Manual Measurement

For a quick sanity check before running the full benchmark:

```python
import time, openai

client = openai.OpenAI(base_url="https://api.runpod.ai/v2/{endpoint_id}/openai/v1", api_key="...")

# End-to-end latency (non-streaming)
start = time.perf_counter()
resp = client.chat.completions.create(
    model="chsa-triage",
    messages=[{"role": "user", "content": "Patient: douleur thoracique depuis 2h, essoufflement"}],
    max_tokens=256,
)
elapsed = time.perf_counter() - start

tokens_out = resp.usage.completion_tokens
print(f"E2E: {elapsed:.2f}s | Tokens: {tokens_out} | Tok/s: {tokens_out/elapsed:.1f}")
```

### Expected Numbers for Qwen3-1.7B on L4 (Warm)

Based on published benchmarks for small models on L4-class GPUs:
- TTFT: ~0.2–0.5 seconds (100-token prompt, bfloat16)
- Generation speed: ~100–200 tokens/second
- E2E for 256-token response: ~1.5–3 seconds

These are estimates; run the actual benchmark and report measured values.

Sources: [vLLM Benchmark CLI](https://docs.vllm.ai/en/latest/benchmarking/cli/) | [vLLM Metrics](https://docs.vllm.ai/en/stable/design/metrics/)

---

## 8. Provider Comparison: RunPod vs Modal vs HF Inference Endpoints

| Dimension | RunPod Serverless | Modal | HF Inference Endpoints |
|---|---|---|---|
| **Pricing model** | Per-second GPU billing | Per-second GPU billing | Per-minute billing |
| **L4 / A10G cost** | $0.00019/sec ($0.68/hr) | ~$0.000306/sec ($1.10/hr) | $0.80/hr (AWS L4) |
| **H100 cost** | ~$0.00116/sec ($4.17/hr) | ~$0.001097/sec ($3.95/hr) | $10/hr (GCP) |
| **Scale-to-zero** | Yes (Flex workers) | Yes (built-in) | Yes (after 15 min idle) |
| **Idle cost** | $0 when scaled to zero | $0 when scaled to zero | $0 when scaled to zero |
| **Cold start (small model, cached)** | 5–15 seconds | 10–60 seconds (memory snapshot alpha: ~5s) | 15–30 seconds |
| **Minimum billing unit** | 1 second | 1 second | 1 minute |
| **Setup complexity** | Medium (Docker + handler) | Low (pure Python decorators, no Dockerfile) | Low (point at HF model ID, click deploy) |
| **Custom Dockerfile** | Required (or use worker-vllm template) | Optional | Not supported |
| **vLLM support** | Official template (worker-vllm) | Manual setup but well-documented | Built-in (text-generation-inference, not vLLM) |
| **GPU selection** | Wide range, spot-price availability | Good selection (A10G, A100, H100) | Limited by cloud region |
| **CI/CD integration** | Via GraphQL API or GitHub integration | Programmatic via `modal deploy` CLI | Via HF Hub API |
| **Best for** | Cost-sensitive POCs, custom Docker stacks | Developer speed, Python-native workflows | Maximum simplicity, HF ecosystem |

### Recommendation for This POC

**RunPod Serverless** is the right choice because:
1. Cheapest GPU cost — important for a student project where the endpoint may be idle 95% of the time
2. Official `worker-vllm` template removes most of the Docker complexity
3. Wide GPU selection increases availability (spot market)
4. Direct vLLM integration gives the OpenAI-compatible API out of the box
5. The graded deliverable requires demonstrating a Docker + vLLM serving stack — RunPod makes this visible and auditable

**Modal** would be the better choice if you did not want to write a Dockerfile at all. Its Python-decorator approach (`@app.function(gpu="A10G")`) is cleaner code. The tradeoff is slightly higher per-second cost and less direct vLLM template support.

**HF Inference Endpoints** is the easiest to set up but uses TGI (Text Generation Inference) not vLLM, which means the graded deliverable requirement for a "vLLM + Docker" pipeline is not satisfied. It also costs more per hour on comparable hardware.

Sources: [RunPod pricing page](https://docs.runpod.io/serverless/pricing) | [Modal pricing](https://modal.com/pricing) | [HF Inference Endpoints pricing](https://huggingface.co/docs/inference-endpoints/en/pricing) | [Serverless LLM comparison 2026](https://blog.premai.io/serverless-llm-deployment-runpod-vs-modal-vs-lambda-2026/)

---

## Summary Checklist

For the OC14 CHSA POC, the minimal viable serving stack is:

- [ ] **Train and merge:** `model.merge_and_unload()` after SFT+DPO, save to `./merged-model/`
- [ ] **Dockerfile:** `FROM vllm/vllm-openai:v0.9.1`, add any wrapper code, no model weights baked in
- [ ] **RunPod endpoint:** Create via Quick Deploy with worker-vllm, set `MODEL_NAME`, `DTYPE=bfloat16`, `MAX_MODEL_LEN=2048`, `HF_TOKEN` as secret
- [ ] **Enable model caching** in endpoint config to eliminate fresh-download cold starts
- [ ] **GitHub Actions:** lint/test on push → build+push image → update RunPod endpoint on merge to main
- [ ] **Latency measurement:** run `vllm bench serve` against the live RunPod endpoint and record TTFT + throughput for the report
- [ ] **API key:** set `--api-key` or use the RunPod endpoint's built-in auth; document in README


---

## Open questions to confirm during implementation

- What is the exact size of the merged Qwen3-1.7B weights after SFT? If they exceed 4 GB as fp16, you need to verify the L4 (24 GB VRAM) can load them with headroom for the KV cache.
- Does RunPod's built-in model caching support private HuggingFace repos with HF_TOKEN, or is a Network Volume required for that?
- The halbgut/runpod-serverless-deploy GitHub Action uses an undocumented GraphQL API — confirm it still works with the current RunPod API version before depending on it in CI.
- If you add a FastAPI wrapper, should it run in the same container as vLLM (proxy on a different port) or as a separate sidecar? For RunPod serverless, same-container is simpler but combining processes complicates health checks.
- What idle-timeout setting on RunPod balances demo availability vs cost? A 60-second idle timeout means a cold start on every grader visit; 300 seconds keeps it warm longer at a modest cost.
