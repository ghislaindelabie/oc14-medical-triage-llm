"""Backend-agnostic triage call + an offline STUB + a response parser.

Two ways to get a triage answer:
  - **stub mode** (`OC14_TRIAGE_STUB=1` or `stub=True`) returns a canned, correctly-structured
    3-part answer so the LangGraph chain runs end-to-end with NO model — used in CI/dev and as
    the safe fallback when no vLLM endpoint is configured.
  - **real mode** calls the OpenAI-compatible vLLM endpoint (same knobs as the serving wrapper:
    temperature 0, stop on ``<|im_end|>``, thinking disabled).

`parse_triage` splits a structured answer back into fields for the graph state, reusing the
canonical `extract_urgency` so parsing matches the eval metric exactly.
"""
from __future__ import annotations

import os
import re

import openai
from openai import OpenAI

from ..config import SYSTEM_PROMPT
from ..eval.metrics import extract_urgency

VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
MODEL_ID = os.environ.get("OC14_MODEL_ID", "oc14-triage")

# timeout: never hang the request thread on an unresponsive vLLM endpoint.
_client = OpenAI(base_url=VLLM_BASE_URL, api_key=os.environ.get("VLLM_API_KEY", "EMPTY"),
                 timeout=30)

_DISCLAIMER = ("ne remplace pas", "does not replace")
_DISCLAIMER_FR = "Cet avis ne remplace pas une consultation médicale."

# Safe, structured triage returned when the backend is unreachable/errors — degrades to a
# moderate level with a clear "service unavailable" justification rather than raising.
_SAFE_FALLBACK = (
    "1. Niveau d'urgence : urgence modérée.\n"
    "2. Justification : service de triage momentanément indisponible.\n"
    "3. Recommandation : orienter vers une évaluation médicale.\n"
    f"{_DISCLAIMER_FR}"
)

# Benign-side keyword cues → downgrade below the default modérée to différée.
_DEFERRED_CUES = ("rhume", "petit", "léger", "bénin", "depuis longtemps", "chronique stable")


def _stub_answer(anon_text: str, red_flags: list | None) -> str:
    if red_flags:
        level = "urgence maximale"
        justification = ("signe(s) d'alerte détecté(s) : "
                         + ", ".join(str(f) for f in red_flags) + " — prise en charge immédiate.")
        recommandation = "orienter sans délai vers une évaluation médicale d'urgence."
    else:
        low = (anon_text or "").lower()
        if any(cue in low for cue in _DEFERRED_CUES):
            level = "urgence différée"
            recommandation = "surveiller et consulter si les symptômes persistent ou s'aggravent."
        else:
            level = "urgence modérée"
            recommandation = "consulter un médecin dans un délai raisonnable."
        justification = "aucun signe d'alerte détecté à ce stade."
    return (
        f"1. Niveau d'urgence : {level}.\n"
        f"2. Justification clinique : {justification}\n"
        f"3. Recommandation : {recommandation}\n"
        f"{_DISCLAIMER_FR}"
    )


def triage_once(anon_text: str, lang: str = "fr", *, red_flags: list | None = None,
                stub: bool | None = None) -> str:
    """Return one structured triage answer for `anon_text`.

    `stub` defaults to the ``OC14_TRIAGE_STUB`` env flag; in stub mode no model is called.
    """
    if stub is None:
        stub = os.environ.get("OC14_TRIAGE_STUB") == "1"
    if stub:
        return _stub_answer(anon_text, red_flags)

    lang = lang if lang in SYSTEM_PROMPT else "fr"
    try:
        r = _client.chat.completions.create(
            model=MODEL_ID, temperature=0, max_tokens=160,
            messages=[{"role": "system", "content": SYSTEM_PROMPT[lang]},
                      {"role": "user", "content": anon_text}],
            stop=["<|im_end|>"],
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
    except openai.OpenAIError:
        # Any backend/network/timeout failure → safe structured fallback, never propagate.
        return _SAFE_FALLBACK
    return (r.choices[0].message.content or "").strip()


_JUSTIF_RE = re.compile(r"(?:justification[^:]*|clinical justification)\s*:?\s*(.+)", re.IGNORECASE)
_RECO_RE = re.compile(r"(?:recommandation|recommendation)\s*:?\s*(.+)", re.IGNORECASE)


def _section(pattern: re.Pattern, text: str) -> str:
    """First line matching `pattern`, returning its captured group stripped of trailing markers."""
    for line in (text or "").splitlines():
        m = pattern.search(line)
        if m:
            return m.group(1).strip()
    return ""


def parse_triage(text: str) -> dict:
    """Split a structured triage answer into {urgency, justification, recommendation,
    disclaimer_present}. Reuses the canonical `extract_urgency` for the level."""
    low = (text or "").lower()
    return {
        "urgency": extract_urgency(text),
        "justification": _section(_JUSTIF_RE, text),
        "recommendation": _section(_RECO_RE, text),
        "disclaimer_present": any(m in low for m in _DISCLAIMER),
    }
