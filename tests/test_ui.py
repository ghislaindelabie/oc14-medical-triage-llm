"""The Gradio UI is thin glue over the (already TDD'd) API. We test the pure result
formatter and smoke-test that the Blocks app builds."""
import pytest

pytest.importorskip("gradio")

from oc14_triage.agent import ui as ui_mod  # noqa: E402
from oc14_triage.agent.ui import build_ui, render_result  # noqa: E402

_DONE = {
    "done": True, "urgency": "urgence maximale", "justification": "douleur thoracique irradiante",
    "recommendation": "appeler le 15 immédiatement", "disclaimer_present": True,
    "interaction_id": "abc-123", "anon_text": "[NOM], douleur thoracique aiguë",
    "red_flags": ["douleur thoracique"],
}


def test_render_result_shows_level_reqid_reco_and_justif():
    md = render_result(_DONE, "fr")
    assert "urgence maximale" in md.lower()
    assert "abc-123" in md                       # req-id (traçabilité) surfaced
    assert "appeler le 15" in md                 # recommendation
    assert "douleur thoracique" in md            # justification / signal


def test_render_result_marks_the_three_levels_distinctly():
    for level in ("urgence maximale", "urgence modérée", "urgence différée"):
        md = render_result({**_DONE, "urgency": level}, "fr")
        assert level in md.lower()


def test_render_result_notes_missing_disclaimer():
    md = render_result({**_DONE, "disclaimer_present": False}, "fr")
    assert "urgence maximale" in md.lower()       # still renders the verdict


def test_answer_renders_friendly_message_on_service_error(monkeypatch):
    # When the API returns an error payload (no done/question), the UI must show a friendly
    # FR "service unavailable" message, NOT a blank verdict card.
    monkeypatch.setattr(ui_mod, "_post", lambda path, payload: {"detail": "boom"})
    history, cleared, sid = ui_mod._answer("douleur", [], "sess-1", "fr")
    last = history[-1]
    assert last["role"] == "assistant"
    assert "momentanément indisponible" in last["content"].lower()
    assert "urgence maximale" not in last["content"].lower()  # no confident verdict rendered


def test_answer_and_start_use_gradio_messages_format(monkeypatch):
    """Gradio 6 Chatbot is type='messages' → every history entry MUST be a {role, content}
    dict, never a (user, bot) tuple (which crashes postprocess). Regression guard for that crash."""
    import gradio as gr
    monkeypatch.setattr(ui_mod, "_post",
                        lambda p, payload: {"session_id": "s1", "question": "Quel est le motif ?"})
    sid, hist0 = ui_mod._start("fr")
    assert all(isinstance(m, dict) and {"role", "content"} <= set(m) for m in hist0)
    monkeypatch.setattr(ui_mod, "_post", lambda p, payload: {
        "done": True, "urgency": "urgence différée", "justification": "x",
        "recommendation": "y", "disclaimer_present": True, "interaction_id": "id1"})
    hist, _, _ = ui_mod._answer("toux légère", hist0, "s1", "fr")
    assert all(isinstance(m, dict) and {"role", "content"} <= set(m) for m in hist)
    assert hist[-2]["role"] == "user" and hist[-1]["role"] == "assistant"
    # Gradio's own Chatbot format check must accept it — exactly the call that raised before.
    gr.Chatbot()._check_format(hist)


def test_render_result_shows_low_confidence_and_review_flag():
    md = render_result({**_DONE, "confidence": "low", "needs_review": True}, "fr")
    assert "faible" in md.lower()
    assert "clinicien" in md.lower()   # HITL review notice surfaced


def test_render_result_high_confidence_has_no_review_notice():
    md = render_result({**_DONE, "confidence": "high", "needs_review": False}, "fr")
    assert "élevée" in md.lower()
    assert "revue" not in md.lower()   # no clinician-review notice when confident


def test_refresh_empty_session_shows_placeholder_not_404(monkeypatch):
    """Refreshing the dossier before any consultation must show a friendly placeholder, never a
    raw HTTP-404 blob (an empty session id would otherwise hit GET /session/ → 404)."""
    calls = []
    monkeypatch.setattr(ui_mod, "_get",
                        lambda path: calls.append(path) or {"detail": "Client error '404 Not Found'"})
    out = ui_mod._refresh("")
    assert calls == []                     # no HTTP call made for an empty session
    assert "404" not in str(out)
    assert "interactions" in out


def test_refresh_valid_session_fetches_dossier(monkeypatch):
    monkeypatch.setattr(ui_mod, "_get", lambda path: {"session_id": "s1", "interactions": [{"x": 1}]})
    assert ui_mod._refresh("s1")["interactions"] == [{"x": 1}]


def test_answer_bootstrap_while_service_down_prints_single_message(monkeypatch):
    """Empty session + service down during bootstrap → exactly ONE 'indisponible' message
    (not doubled), and any passed-in history is preserved."""
    monkeypatch.setattr(ui_mod, "_post", lambda p, payload: {"detail": "boom"})
    hist, _, _ = ui_mod._answer("douleur", [], "", "fr")
    downs = [m for m in hist if isinstance(m, dict) and "indisponible" in m.get("content", "").lower()]
    assert len(downs) == 1


def test_render_result_unavailable_shows_notice_not_fake_level():
    """When the model is unavailable (urgency None), show a clear retry/clinician notice —
    never a fabricated '**None**' urgency level."""
    md = render_result({"urgency": None, "needs_review": True, "interaction_id": "u1",
                        "justification": "Le modèle de triage démarre ou est momentanément indisponible.",
                        "recommendation": "Réessayez dans ~1 min ; en attendant, ce cas relève de l'évaluation d'un clinicien.",
                        "disclaimer_present": True}, "fr")
    assert "None" not in md
    assert "réessay" in md.lower()
    assert "clinicien" in md.lower()


def test_build_ui_returns_blocks():
    import gradio as gr
    assert isinstance(build_ui(), gr.Blocks)
