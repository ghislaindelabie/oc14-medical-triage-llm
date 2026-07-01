"""Backend-agnostic triage call: STUB path (no model) + parser + a MOCKED real path.

NEVER hits the network — the real path is exercised by monkeypatching the module-level OpenAI
client, mirroring the mock style in tests/test_serving.py. Run with:
    uv run --no-sync pytest tests/test_backend.py -q
"""
import pytest

pytest.importorskip("openai")

from oc14_triage.agent import backend as bk  # noqa: E402


class _Resp:
    """Shape-compatible with openai chat.completions.create(...) return."""

    def __init__(self, content: str):
        self.model = "oc14-triage"
        msg = type("M", (), {"content": content})()
        self.choices = [type("C", (), {"message": msg})()]


# --- STUB path ---------------------------------------------------------------

def test_stub_red_flag_forces_maximale_with_disclaimer():
    out = bk.triage_once("...", stub=True, red_flags=["douleur thoracique"])
    assert "urgence maximale" in out.lower()
    assert "ne remplace pas" in out.lower()


def test_stub_benign_input_is_structured_three_part():
    out = bk.triage_once("j'ai un petit rhume depuis hier", stub=True)
    low = out.lower()
    assert "niveau d'urgence" in low
    assert "justification" in low
    assert "recommandation" in low


# --- parser ------------------------------------------------------------------

_SAMPLE = (
    "1. Niveau d'urgence : urgence modérée.\n"
    "2. Justification clinique : fièvre persistante depuis 3 jours sans signe de gravité.\n"
    "3. Recommandation : consulter un médecin dans les 24 heures.\n"
    "Cet avis ne remplace pas une consultation médicale."
)


def test_parse_extracts_all_fields():
    parsed = bk.parse_triage(_SAMPLE)
    assert parsed["urgency"] == "urgence modérée"
    assert parsed["justification"].strip()
    assert parsed["recommendation"].strip()
    assert parsed["disclaimer_present"] is True


def test_parse_without_disclaimer_flags_false():
    text = "1. Niveau d'urgence : urgence différée.\n3. Recommandation : repos."
    assert bk.parse_triage(text)["disclaimer_present"] is False


# --- REAL path (mocked — no network) -----------------------------------------

def test_real_path_passes_temperature_stop_and_system_prompt(monkeypatch):
    from oc14_triage.config import SYSTEM_PROMPT

    captured = {}

    def _fake_create(**kwargs):
        captured.update(kwargs)
        return _Resp("1. Niveau d'urgence : urgence différée. 3. Recommandation : repos.")

    monkeypatch.setattr(bk._client.chat.completions, "create", _fake_create)

    out = bk.triage_once("douleur au genou depuis une semaine", lang="fr", stub=False)

    assert out  # message content returned
    assert captured["temperature"] == 0
    assert captured["stop"] == ["<|im_end|>"]
    system_msg = next(m for m in captured["messages"] if m["role"] == "system")
    assert system_msg["content"] == SYSTEM_PROMPT["fr"]


def test_real_path_backend_error_returns_safe_fallback(monkeypatch):
    """A backend/network failure must degrade to a SAFE structured triage, never raise."""
    import openai

    def _boom(**kwargs):
        raise openai.APIConnectionError(request=None)

    monkeypatch.setattr(bk._client.chat.completions, "create", _boom)

    out = bk.triage_once("douleur au genou depuis une semaine", lang="fr", stub=False)
    low = out.lower()
    assert "urgence modérée" in low            # safe default level
    assert "ne remplace pas" in low            # FR disclaimer preserved
    # and it is still parseable into the three-part structure
    parsed = bk.parse_triage(out)
    assert parsed["urgency"] == "urgence modérée"
    assert parsed["recommendation"].strip()
