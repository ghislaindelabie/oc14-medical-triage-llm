import json

from oc14_triage.agent.sih import URGENCY_CODING, to_fhir


def _entries_of_type(bundle: dict, resource_type: str) -> list[dict]:
    return [e["resource"] for e in bundle["entry"] if e["resource"]["resourceType"] == resource_type]


def test_to_fhir_is_a_collection_bundle_with_one_encounter_and_one_observation():
    case = {
        "interaction_id": "int-001",
        "session_id": "sess-abc",
        "urgency": "urgence modérée",
        "timestamp_utc": "2026-07-01T10:00:00Z",
    }
    bundle = to_fhir(case)
    assert bundle["resourceType"] == "Bundle"
    assert bundle["type"] == "collection"
    assert len(_entries_of_type(bundle, "Encounter")) == 1
    assert len(_entries_of_type(bundle, "Observation")) == 1


def test_observation_encodes_urgency_via_urgency_coding():
    case = {
        "interaction_id": "int-002",
        "session_id": "sess-def",
        "urgency": "urgence maximale",
        "timestamp_utc": "2026-07-01T11:00:00Z",
    }
    obs = _entries_of_type(to_fhir(case), "Observation")[0]
    coding = obs["valueCodeableConcept"]["coding"][0]
    assert coding == URGENCY_CODING["urgence maximale"]
    assert coding["code"] == "max"
    assert obs["effectiveDateTime"] == "2026-07-01T11:00:00Z"


def test_subject_is_pseudonymous_and_bundle_has_no_pii_name_key():
    # a name in the case must never leak into the FHIR output — subject is session_id only
    case = {
        "interaction_id": "int-003",
        "session_id": "sess-ghi",
        "urgency": "urgence différée",
        "timestamp_utc": "2026-07-01T12:00:00Z",
        "name": "Jean Dupont",  # PII that MUST NOT propagate
    }
    bundle = to_fhir(case)
    for resource_type in ("Encounter", "Observation"):
        subj = _entries_of_type(bundle, resource_type)[0]["subject"]
        assert subj == {"reference": "Patient/sess-ghi"}
    raw = json.dumps(bundle).lower()
    assert '"name"' not in raw
    assert "dupont" not in raw


def test_unknown_urgency_does_not_crash_and_omits_coding():
    # urgency=None (unparseable model output) must not raise KeyError(None) — the bundle
    # still builds, just without a valueCodeableConcept coding.
    case = {
        "interaction_id": "int-none",
        "session_id": "sess-none",
        "urgency": None,
        "timestamp_utc": "2026-07-01T14:00:00Z",
    }
    bundle = to_fhir(case)
    assert bundle["resourceType"] == "Bundle"
    obs = _entries_of_type(bundle, "Observation")[0]
    assert "coding" not in obs.get("valueCodeableConcept", {})


def test_encounter_id_is_interaction_id():
    case = {
        "interaction_id": "int-xyz-42",
        "session_id": "sess-jkl",
        "urgency": "urgence modérée",
        "timestamp_utc": "2026-07-01T13:00:00Z",
    }
    enc = _entries_of_type(to_fhir(case), "Encounter")[0]
    assert enc["id"] == "int-xyz-42"
