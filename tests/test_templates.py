from oc14_triage.config import SYSTEM_PROMPT, URGENCY_LEVELS
from oc14_triage.data.templates import chat_example, heuristic_urgency, triage_response


def test_heuristic_urgency_red_flag_fr():
    assert heuristic_urgency("Patient avec douleur thoracique et sueurs", "fr") == "urgence maximale"


def test_heuristic_urgency_red_flag_en():
    assert heuristic_urgency("sudden chest pain and shortness of breath", "en") == "urgence maximale"


def test_heuristic_urgency_deferred():
    assert heuristic_urgency("demande de conseil de prévention", "fr") == "urgence différée"


def test_heuristic_urgency_default_moderate():
    assert heuristic_urgency("douleur au genou depuis une semaine", "fr") == "urgence modérée"


def test_triage_response_structure():
    out = triage_response("urgence maximale", "raison", "agir", "fr")
    assert "urgence maximale" in out
    assert "1." in out and "2." in out and "3." in out
    assert "ne remplace pas" in out.lower()


def test_chat_example_shape():
    row = chat_example("scénario", "réponse", "fr", "unit", "triage")
    msgs = row["messages"]
    assert [m["role"] for m in msgs] == ["system", "user", "assistant"]
    assert msgs[0]["content"] == SYSTEM_PROMPT["fr"]
    assert row["lang"] == "fr" and row["kind"] == "triage"


def test_all_levels_are_french():
    assert URGENCY_LEVELS == ("urgence maximale", "urgence modérée", "urgence différée")
