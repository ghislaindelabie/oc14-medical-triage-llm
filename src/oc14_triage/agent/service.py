"""FastAPI service exposing the triage agent — this IS the "API" deliverable.

Drives the adaptive questionnaire (collecte) turn by turn, then runs the LangGraph chain
once the core fields are gathered, and exposes the traceability history. Session answers
are held in memory only (transient); everything PERSISTED goes through the chain's
anonymisation node first, so the SQLite dossier holds no raw PII.
"""

from __future__ import annotations

import os
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .. import anonymization
from ..config import DATA
from .graph import process_case
from .questionnaire import assemble_case_text, is_optional, next_field, next_question
from .store import Store


@asynccontextmanager
async def _lifespan(app: FastAPI):
    # Warm the spaCy/Presidio models at startup (~7.5s cold load) so the FIRST real answer
    # isn't stalled loading NER models mid-demo. Failures here are non-fatal (regex fallback).
    try:
        anonymization.anonymize("warmup", lang="fr")
    except Exception:  # noqa: BLE001
        pass
    yield


app = FastAPI(title="CHSA — Agent de triage médical (POC)", lifespan=_lifespan)

_SESSIONS: dict[str, dict] = {}
_store: Store | None = None


def get_store() -> Store:
    global _store
    if _store is None:
        _store = Store(os.environ.get("OC14_AGENT_DB", str(DATA / "agent_sessions.db")))
    return _store


class StartReq(BaseModel):
    lang: str = "fr"


class AnswerReq(BaseModel):
    session_id: str
    answer: str


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/session/start")
def start(req: StartReq) -> dict:
    lang = req.lang if req.lang in ("fr", "en") else "fr"  # coerce unsupported langs
    sid = str(uuid.uuid4())
    _SESSIONS[sid] = {"lang": lang, "answers": {}}
    return {"session_id": sid, "field": next_field({}, lang),
            "question": next_question({}, lang)}


@app.post("/session/answer")
def answer(req: AnswerReq) -> dict:
    sess = _SESSIONS.get(req.session_id)
    if sess is None:
        raise HTTPException(status_code=404, detail="unknown session")
    lang, answers = sess["lang"], sess["answers"]
    lang = lang if lang in ("fr", "en") else "fr"  # coerce unsupported langs

    field = next_field(answers, lang)
    # A blank answer must not fill a REQUIRED field — re-ask the SAME question so collecte can't
    # complete on empty input (which would yield a confident verdict on no information). An
    # OPTIONAL field (vitals) is instead SKIPPED on blank so the collecte can complete.
    if field is not None and not req.answer.strip() and not is_optional(field):
        return {"done": False, "field": field, "question": next_question(answers, lang)}
    if field is not None:
        answers[field] = req.answer

    remaining = next_field(answers, lang)
    if remaining is not None:
        return {"done": False, "field": remaining,
                "question": next_question(answers, lang)}

    # collecte complete → run the chain (assembled text is anonymised inside the graph)
    final = process_case(assemble_case_text(answers, lang), session_id=req.session_id,
                         lang=lang, store=get_store(), answers=answers)
    return {
        "done": True, "session_id": req.session_id, "urgency": final.get("urgency"),
        "justification": final.get("justification", ""),
        "recommendation": final.get("recommendation", ""),
        "confidence": final.get("confidence"),
        "needs_review": final.get("needs_review", False),
        "disclaimer_present": final.get("disclaimer_present", False),
        "interaction_id": final.get("interaction_id"), "anon_text": final.get("anon_text", ""),
        "red_flags": final.get("red_flags", []), "sih_record": final.get("sih_record", {}),
        "trace": final.get("trace", []),
    }


@app.get("/session/{session_id}")
def session_history(session_id: str) -> dict:
    return {"session_id": session_id, "interactions": get_store().history(session_id)}


@app.get("/sessions")
def sessions() -> dict:
    return {"sessions": get_store().all_sessions()}
