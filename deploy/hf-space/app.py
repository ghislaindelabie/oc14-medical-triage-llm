"""Hugging Face Space entrypoint for the CHSA medical-triage agent.

Runs BOTH tiers of the deliverable in one process (approach (a)):
  1. the FastAPI triage service (`oc14_triage.agent.service:app`) on 127.0.0.1:8091,
     started in a daemon thread via uvicorn;
  2. the Gradio patient-facing UI (`oc14_triage.agent.ui:build_ui`) on 0.0.0.0:7860,
     which HF Spaces exposes publicly.

The UI talks to the service over HTTP at AGENT_SERVICE_URL, exactly as in the local
demo — so the Space exercises the real API deliverable, not a shortcut. No agent logic
is reimplemented here; this file only wires the two existing modules together.

Backend selection is entirely by environment (set as Space secrets/variables):
  VLLM_BASE_URL, VLLM_API_KEY, OC14_MODEL_ID, OC14_MODEL_VERSION, VLLM_TIMEOUT.
No OC14_TRIAGE_STUB → the real fine-tuned model on the RunPod vLLM endpoint is used.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path

import httpx

# The `src/` layout package lives next to this file — put it on the path so
# `import oc14_triage...` resolves without an editable install.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

SERVICE_HOST = "127.0.0.1"
SERVICE_PORT = int(os.environ.get("AGENT_SERVICE_PORT", "8091"))
SERVICE_URL = f"http://{SERVICE_HOST}:{SERVICE_PORT}"

# ui.py reads AGENT_SERVICE_URL at import time → it MUST be set before importing ui.
os.environ["AGENT_SERVICE_URL"] = SERVICE_URL
# The traceability SQLite dossier must live on a writable path. On a Space only /tmp
# (and /data with paid storage) is writable, so default the DB there.
os.environ.setdefault("OC14_AGENT_DB", "/tmp/agent_sessions.db")


def _serve_api() -> None:
    """Run the FastAPI triage service under uvicorn (blocking) — called in a daemon thread."""
    import uvicorn

    # Import the ASGI app lazily inside the thread so any heavy first-import cost
    # (Presidio/spaCy load happens on the service's own startup lifespan) is off the
    # main import path.
    uvicorn.run(
        "oc14_triage.agent.service:app",
        host=SERVICE_HOST,
        port=SERVICE_PORT,
        log_level="info",
    )


def _wait_for_health(timeout_s: float = 90.0) -> bool:
    """Poll GET /health until the service answers ok, or the timeout elapses.

    The service warms Presidio/spaCy (~7.5 s) in its startup lifespan, so first health
    can lag a few seconds; we give it generous headroom without failing the boot.
    """
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            r = httpx.get(f"{SERVICE_URL}/health", timeout=5)
            if r.status_code == 200 and r.json().get("status") == "ok":
                return True
        except httpx.HTTPError:
            pass
        time.sleep(1.0)
    return False


def main() -> None:
    threading.Thread(target=_serve_api, name="triage-api", daemon=True).start()

    if _wait_for_health():
        print(f"[app] triage API healthy at {SERVICE_URL}", flush=True)
    else:
        # Non-fatal: the UI degrades to a friendly "service unavailable" message and a
        # later request may still succeed once the API finishes warming. Launch anyway
        # so the Space reaches RUNNING rather than crash-looping.
        print("[app] WARNING: triage API not healthy yet; launching UI anyway", flush=True)

    from oc14_triage.agent.ui import build_ui

    build_ui().launch(server_name="0.0.0.0", server_port=int(os.environ.get("UI_PORT", "7860")))


if __name__ == "__main__":
    main()
