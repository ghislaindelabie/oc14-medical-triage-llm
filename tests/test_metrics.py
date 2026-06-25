from oc14_triage.eval.metrics import (
    behavioural_report,
    extract_urgency,
    format_ok,
    has_disclaimer,
    has_think_block,
    response_language,
    triage_report,
)


def test_extract_urgency():
    assert extract_urgency("1. Niveau d'urgence : urgence modérée.") == "urgence modérée"
    assert extract_urgency("no level here") is None


def test_has_disclaimer():
    assert has_disclaimer("… ne remplace pas une consultation médicale.")
    assert has_disclaimer("… does not replace a medical consultation.")
    assert not has_disclaimer("plain text")


def test_has_think_block():
    assert has_think_block("<think>reasoning</think> answer")
    assert not has_think_block("clean answer")


def test_format_ok():
    assert format_ok("Niveau d'urgence : urgence maximale. Recommandation : agir.")
    assert not format_ok("just some medical prose without structure")


def test_response_language():
    assert response_language("Le patient présente une douleur thoracique aiguë.") == "fr"
    assert response_language("The patient presents with acute chest pain.") == "en"


def test_triage_report_accuracy_and_recall():
    pairs = [
        ("urgence maximale", "urgence maximale"),
        ("urgence modérée", "urgence maximale"),  # missed escalation
        ("urgence modérée", "urgence modérée"),
        ("urgence différée", "urgence différée"),
    ]
    rep = triage_report(pairs)
    assert rep["n"] == 4
    assert rep["accuracy"] == 0.75
    assert rep["recall_urgence_maximale"] == 0.5


def test_triage_report_handles_none_predictions():
    # the untrained Base produces unparseable output -> pred None; must not crash (regression)
    pairs = [("urgence maximale", "urgence maximale"), (None, "urgence différée"),
             (None, "urgence modérée"), ("urgence modérée", "urgence modérée")]
    rep = triage_report(pairs)
    assert rep["n"] == 4
    assert "urgence différée->(none)" in rep["confusion_gold_pred"]
    assert rep["recall_ci_per_level"]["urgence maximale"] is not None


def test_extract_urgency_prefers_verdict_line():
    # E3: a level name-dropped in prose before the verdict line must NOT win
    t = "Ce n'est pas une urgence maximale. Niveau d'urgence : urgence modérée. Recommandation : ..."
    assert extract_urgency(t) == "urgence modérée"


def test_behavioural_report_rates():
    responses = [
        {"text": "Niveau d'urgence : urgence maximale. Recommandation : agir. ne remplace pas.",
         "lang": "fr"},
        {"text": "<think>x</think> Niveau d'urgence : urgence modérée. Recommandation : voir.",
         "lang": "fr"},
    ]
    rep = behavioural_report(responses)
    assert rep["n"] == 2
    assert rep["no_think_rate"] == 0.5
    assert rep["disclaimer_rate"] == 0.5
