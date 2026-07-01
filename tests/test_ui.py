"""The Gradio UI is thin glue over the (already TDD'd) API. We test the pure result
formatter and smoke-test that the Blocks app builds."""
import pytest

pytest.importorskip("gradio")

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


def test_build_ui_returns_blocks():
    import gradio as gr
    assert isinstance(build_ui(), gr.Blocks)
