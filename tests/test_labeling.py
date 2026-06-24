from oc14_triage.labeling.aggregate import Label, consensus, fleiss_kappa, parse_label
from oc14_triage.labeling.rubric import build_user_prompt, esi_to_urgency

MAX, MOD, DIF = "urgence maximale", "urgence modérée", "urgence différée"


# --- mapping ---
def test_esi_to_urgency():
    assert [esi_to_urgency(i) for i in (1, 2, 3, 4, 5)] == [MAX, MAX, MOD, DIF, DIF]


# --- prompt ---
def test_prompt_has_rubric_and_case():
    p = build_user_prompt("Patient de 60 ans, douleur thoracique.")
    assert "urgence maximale" in p and "ESI" in p and "douleur thoracique" in p


# --- parsing ---
def test_parse_valid_consistent():
    raw = '{"is_triage_case": true, "urgency": "urgence maximale", "esi": 2, "justification": "x"}'
    lab = parse_label("m", raw)
    assert lab.urgency == MAX and lab.esi == 2 and lab.consistent and lab.is_triage_case


def test_parse_inconsistent_esi():
    lab = parse_label("m", '{"urgency": "urgence maximale", "esi": 4}')  # 4 -> différée, not maximale
    assert lab.urgency == MAX and lab.esi == 4 and lab.consistent is False


def test_parse_code_fence_and_prose():
    raw = "Voici ma réponse:\n```json\n{\"urgency\":\"urgence différée\",\"esi\":5}\n```\nmerci"
    lab = parse_label("m", raw)
    assert lab.urgency == DIF and lab.esi == 5 and lab.consistent


def test_parse_non_triage_and_garbage():
    assert parse_label("m", '{"is_triage_case": false, "urgency": "urgence modérée", "esi": 3}').is_triage_case is False
    bad = parse_label("m", "pas de json ici")
    assert bad.urgency is None and bad.error


# --- consensus ---
def _lab(model, urg, esi, triage=True):
    return Label(model, triage, urg, esi, esi_to_urgency(esi) == urg if urg else False)


def test_consensus_unanimous_is_gold():
    c = consensus("c1", [_lab("a", MAX, 1), _lab("b", MAX, 2), _lab("c", MAX, 2)])
    assert c.urgency == MAX and c.unanimous and c.is_gold and not c.flagged and c.n_agree == 3


def test_consensus_majority_not_gold():
    c = consensus("c2", [_lab("a", MAX, 2), _lab("b", MAX, 2), _lab("c", MOD, 3)])
    assert c.urgency == MAX and c.n_agree == 2 and not c.unanimous and not c.is_gold and c.flagged


def test_consensus_non_triage_blocks_gold():
    c = consensus("c3", [_lab("a", MAX, 1), _lab("b", MAX, 1), _lab("c", MAX, 1, triage=False)])
    assert not c.all_triage and not c.is_gold


# --- Fleiss' kappa ---
def test_fleiss_perfect_agreement():
    assert fleiss_kappa([[MAX, MAX, MAX], [DIF, DIF, DIF], [MOD, MOD, MOD]]) == 1.0


def test_fleiss_more_agreement_scores_higher():
    high = fleiss_kappa([[MAX, MAX, MAX], [DIF, DIF, DIF]] * 5)
    low = fleiss_kappa([[MAX, MOD, DIF], [MAX, DIF, MOD]] * 5)
    assert high > low
