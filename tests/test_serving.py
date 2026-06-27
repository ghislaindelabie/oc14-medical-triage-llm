"""Wrapper plumbing tests with a MOCKED vLLM backend — no GPU/model needed. Skip if the serving
extra isn't installed (CI installs only the dev group)."""
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("openai")
from fastapi.testclient import TestClient  # noqa: E402

from oc14_triage.serving import app as appmod  # noqa: E402


class _Resp:
    def __init__(self, content):
        self.model = "oc14-triage"
        msg = type("M", (), {"content": content})()
        self.choices = [type("C", (), {"message": msg})()]


def test_health():
    assert TestClient(appmod.app).get("/health").json()["status"] == "ok"


def test_triage_extracts_urgency(monkeypatch):
    monkeypatch.setattr(appmod._client.chat.completions, "create",
                        lambda **k: _Resp("1. Niveau d'urgence : urgence maximale. "
                                          "3. Recommandation : agir. … ne remplace pas une consultation."))
    r = TestClient(appmod.app).post("/triage", json={"text": "Douleur thoracique aiguë", "lang": "fr"})
    assert r.status_code == 200
    body = r.json()
    assert body["urgency"] == "urgence maximale"
    assert "ne remplace pas" in body["answer"]


def test_triage_rejects_empty(monkeypatch):
    monkeypatch.setattr(appmod._client.chat.completions, "create", lambda **k: _Resp("x"))
    assert TestClient(appmod.app).post("/triage", json={"text": "   ", "lang": "fr"}).status_code == 422
