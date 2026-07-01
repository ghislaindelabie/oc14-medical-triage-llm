"""Tests for the rule-guided adaptive questionnaire (built strict TDD)."""

from __future__ import annotations

from oc14_triage.agent.questionnaire import (
    assemble_case_text,
    detect_red_flags,
    is_complete,
    next_question,
)


def test_empty_answers_asks_for_motif():
    q = next_question({})
    assert q is not None
    low = q.lower()
    assert "motif" in low or "consult" in low or "plaint" in low or "amène" in low


def test_chest_pain_motif_triggers_red_flag_and_targeted_followup():
    answers = {"motif": "douleur thoracique depuis ce matin"}
    flags = detect_red_flags(answers["motif"])
    assert "douleur thoracique" in flags
    # Adaptive: with a chest-pain cue and no follow-up yet, ask a chest-pain-specific
    # question (irradiation / sueurs / essoufflement), NOT the generic onset question.
    q = next_question(answers)
    assert q is not None
    low = q.lower()
    assert any(cue in low for cue in ("irradi", "sueur", "essouffl", "respir"))
    assert "début" not in low and "onset" not in low


def test_benign_motif_asks_onset_then_severity():
    # No red flag -> straight to the remaining core fields, in order.
    q1 = next_question({"motif": "toux légère"})
    assert q1 is not None and ("début" in q1.lower() or "quand" in q1.lower())
    q2 = next_question({"motif": "toux légère", "debut": "hier"})
    assert q2 is not None and ("intensit" in q2.lower() or "1" in q2 or "10" in q2)


def test_core_fields_present_is_complete_and_no_more_questions():
    answers = {"motif": "toux légère", "debut": "hier", "intensite": "3"}
    assert is_complete(answers) is True
    assert next_question(answers) is None


def test_incomplete_is_not_complete():
    assert is_complete({"motif": "toux légère"}) is False


def test_assemble_case_text_includes_motif():
    answers = {"motif": "douleur thoracique", "debut": "ce matin", "intensite": "8"}
    text = assemble_case_text(answers)
    assert "douleur thoracique" in text
    assert isinstance(text, str) and text.strip()
