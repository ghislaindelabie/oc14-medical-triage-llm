"""Thin FastAPI wrapper over the vLLM endpoint serving the OC14 triage model.

The model runs on vLLM (RunPod serverless / Modal / a local process) behind its OpenAI-compatible API.
This wrapper exists for the three reasons the research doc flags as justifying it:
  - **inject the trained triage system prompt** (callers send only the patient text);
  - **force non-thinking output + stop on `<|im_end|>`** (Decision-H — the small model otherwise runs on);
  - an **API-key gate** + a **privacy-safe audit log** (metadata only — never the patient text).

Config via env:  VLLM_BASE_URL (default http://localhost:8000/v1), OC14_MODEL_ID, OC14_API_KEY
(if set, required via the X-API-Key header), VLLM_API_KEY (for the backend, default "EMPTY").
"""
from __future__ import annotations

import logging
import os
import time

from fastapi import FastAPI, Header, HTTPException
from openai import OpenAI
from pydantic import BaseModel

from ..config import SYSTEM_PROMPT
from ..eval.metrics import extract_urgency

VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
MODEL_ID = os.environ.get("OC14_MODEL_ID", "oc14-triage")
API_KEY = os.environ.get("OC14_API_KEY")  # if set, required via X-API-Key

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_audit = logging.getLogger("oc14.audit")
_client = OpenAI(base_url=VLLM_BASE_URL, api_key=os.environ.get("VLLM_API_KEY", "EMPTY"))

app = FastAPI(title="OC14 medical-triage assistant", version="0.1.0")


class TriageRequest(BaseModel):
    text: str
    lang: str = "fr"


class TriageResponse(BaseModel):
    urgency: str | None
    answer: str
    model: str


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": MODEL_ID, "backend": VLLM_BASE_URL}


@app.post("/triage", response_model=TriageResponse)
def triage(req: TriageRequest, x_api_key: str | None = Header(default=None)) -> TriageResponse:
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="invalid or missing API key")
    if not req.text.strip():
        raise HTTPException(status_code=422, detail="empty patient text")
    lang = req.lang if req.lang in SYSTEM_PROMPT else "fr"
    t0 = time.time()
    try:
        r = _client.chat.completions.create(
            model=MODEL_ID, temperature=0, max_tokens=256,
            messages=[{"role": "system", "content": SYSTEM_PROMPT[lang]},
                      {"role": "user", "content": req.text}],
            stop=["<|im_end|>"],
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
    except Exception as e:  # noqa: BLE001 — surface backend errors as 502
        raise HTTPException(status_code=502, detail=f"triage backend error: {e}") from e
    answer = (r.choices[0].message.content or "").strip()
    urgency = extract_urgency(answer)
    # Privacy: log METADATA only (lang / predicted urgency / latency / input length) — never the text.
    _audit.info("triage lang=%s urgency=%s ms=%d chars=%d", lang, urgency,
                int(1000 * (time.time() - t0)), len(req.text))
    return TriageResponse(urgency=urgency, answer=answer, model=r.model)
