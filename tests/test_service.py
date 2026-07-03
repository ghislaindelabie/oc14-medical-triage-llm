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
                 "depuis 30 min", "9", "SpO2 90%, FC 130"])
    assert r["done"] is True
    assert r["urgency"] == "urgence maximale"       # red-flag → escalated
    assert "Jean Dupont" not in r["anon_text"]        # PII gone
    assert r["interaction_id"] and r["disclaimer_present"] is True
    assert r["sih_record"]["resourceType"] == "Bundle"
    assert r["confidence"] == "high"                  # red-flag + model agree


def test_history_endpoint_has_no_pii():
    sid, _ = _run(["Marie Curie, mal de gorge léger", "depuis hier", "3", ""])  # blank vitals skip
    hist = client.get(f"/session/{sid}").json()
    assert len(hist["interactions"]) == 1
    assert "Marie Curie" not in json.dumps(hist, ensure_ascii=False)


def test_optional_vitals_can_be_skipped_blank():
    """The vitals step is OPTIONAL: a blank answer skips it (no re-ask) and completes."""
    sid = client.post("/session/start", json={"lang": "fr"}).json()["session_id"]
    r = None
    for a in ["toux légère", "hier", "2"]:
        r = client.post("/session/answer", json={"session_id": sid, "answer": a}).json()
    assert r["done"] is False and r["field"] == "vitals"   # optional vitals step offered
    r = client.post("/session/answer", json={"session_id": sid, "answer": "   "}).json()
    assert r["done"] is True
    assert r.get("confidence") in ("high", "medium", "low")


def test_answer_unknown_session_is_404():
    assert client.post("/session/answer",
                       json={"session_id": "nope", "answer": "x"}).status_code == 404


def test_trace_endpoint_archives_cases_across_sessions():
    """GET /trace returns the GLOBAL dossier archive — every case AND every re-evaluation turn
    across every session — so the demo panel shows all cases an evaluator submitted."""
    _run(["mal de dos", "hier", "4", ""])                       # session 1, one case
    _run(["entorse cheville", "ce matin", "5", ""])             # session 2, a DIFFERENT case
    trace = client.get("/trace").json()
    joined = " || ".join(it["symptoms_anon"] for it in trace["interactions"])
    assert "mal de dos" in joined
    assert "entorse cheville" in joined                          # both distinct cases present
    assert len(trace["interactions"]) >= 2


def test_followup_after_verdict_accumulates_and_appends_new_interaction():
    """After the verdict, a free-text follow-up is an 'information complémentaire': it is
    accumulated, the WHOLE case is re-assembled, and a NEW interaction is appended to the SAME
    session (advice evolving) — NOT a stale re-triage of the old answers."""
    sid, r0 = _run(["mal de tête", "hier", "4", ""])
    assert r0["done"] is True
    before = len(client.get(f"/session/{sid}").json()["interactions"])
    r1 = client.post("/session/answer",
                     json={"session_id": sid, "answer": "j'ai aussi de la fièvre à 39"}).json()
    assert r1["done"] is True
    assert "fièvre" in r1["anon_text"] or "39" in r1["anon_text"]   # complement folded into case
    hist = client.get(f"/session/{sid}").json()["interactions"]
    assert len(hist) == before + 1                                  # new turn appended, same session


def test_followup_with_redflag_reescalates_updated_verdict():
    """A follow-up that introduces a red-flag must re-escalate the UPDATED verdict to maximale —
    red-flag detection + confidence re-run on the AUGMENTED text, not the original answers."""
    sid, r0 = _run(["mal de dos léger", "hier", "3", ""])
    assert r0["urgency"] != "urgence maximale"                      # benign to start
    r1 = client.post("/session/answer", json={
        "session_id": sid, "answer": "maintenant j'ai une douleur thoracique et j'étouffe"}).json()
    assert r1["done"] is True
    assert r1["urgency"] == "urgence maximale"                      # red-flag in complement escalates
    assert "douleur thoracique" in r1["red_flags"]


def test_blank_answer_reasks_same_field_and_does_not_complete():
    """A blank answer must re-ask the SAME field, never advance / complete the collecte."""
    start = client.post("/session/start", json={"lang": "fr"}).json()
    sid = start["session_id"]
    first_field = start["field"]  # "motif"
    r = client.post("/session/answer", json={"session_id": sid, "answer": "   "}).json()
    assert r["done"] is False
    assert r["field"] == first_field   # still on motif, not advanced
    # a real answer then advances past motif
    r2 = client.post("/session/answer",
                     json={"session_id": sid, "answer": "mal de tête"}).json()
    assert r2["field"] != first_field
