"""API tests for the agent service — the questionnaire loop + running the chain.
Backend stubbed, SQLite in a temp dir. No model, no network."""
import json
import os
import pathlib
import tempfile

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("langgraph")
os.environ["OC14_TRIAGE_STUB"] = "1"
os.environ["OC14_AGENT_DB"] = str(pathlib.Path(tempfile.mkdtemp()) / "svc.db")

from fastapi.testclient import TestClient  # noqa: E402

from oc14_triage.agent.service import app  # noqa: E402

client = TestClient(app)


def _run(answers, lang="fr"):
    sid = client.post("/session/start", json={"lang": lang}).json()["session_id"]
    r = None
    for a in answers:
        r = client.post("/session/answer", json={"session_id": sid, "answer": a}).json()
        if r.get("done"):
            break
    return sid, r


def test_health():
    assert client.get("/health").json()["status"] == "ok"


def test_start_asks_motif_first():
    r = client.post("/session/start", json={"lang": "fr"}).json()
    assert r["session_id"] and r["field"] == "motif" and r["question"]


def test_full_flow_redflag_reaches_maximale_and_anonymises():
    _, r = _run(["Jean Dupont, douleur thoracique aiguë", "irradie bras gauche, sueurs",
                 "depuis 30 min", "9"])
    assert r["done"] is True
    assert r["urgency"] == "urgence maximale"       # red-flag → escalated
    assert "Jean Dupont" not in r["anon_text"]        # PII gone
    assert r["interaction_id"] and r["disclaimer_present"] is True
    assert r["sih_record"]["resourceType"] == "Bundle"


def test_history_endpoint_has_no_pii():
    sid, _ = _run(["Marie Curie, mal de gorge léger", "depuis hier", "3"])
    hist = client.get(f"/session/{sid}").json()
    assert len(hist["interactions"]) == 1
    assert "Marie Curie" not in json.dumps(hist, ensure_ascii=False)


def test_answer_unknown_session_is_404():
    assert client.post("/session/answer",
                       json={"session_id": "nope", "answer": "x"}).status_code == 404
