"""next_field() exposes WHICH field the next question fills — the service needs the key
to store each answer under. It must stay consistent with next_question()."""
from oc14_triage.agent.questionnaire import next_field, next_question


def test_starts_with_motif():
    assert next_field({}) == "motif"


def test_followup_after_redflag_motif():
    assert next_field({"motif": "douleur thoracique aiguë"}) == "followup"


def test_core_order_without_redflag():
    assert next_field({"motif": "mal de gorge"}) == "debut"
    assert next_field({"motif": "mal de gorge", "debut": "hier"}) == "intensite"


def test_optional_vitals_offered_after_core_then_none():
    core = {"motif": "x", "debut": "y", "intensite": "5"}
    assert next_field(core) == "vitals"                    # optional vitals offered after core
    assert next_field({**core, "vitals": ""}) is None       # answered (even blank) → done


def test_vitals_is_optional_core_is_not():
    from oc14_triage.agent.questionnaire import is_optional
    assert is_optional("vitals") is True
    assert is_optional("motif") is False
    assert is_optional("intensite") is False


def test_next_question_consistent_with_field():
    # red-flag motif → next_field is followup and next_question is the chest-pain probe
    a = {"motif": "douleur thoracique"}
    assert next_field(a) == "followup"
    assert "irradie" in next_question(a).lower()
