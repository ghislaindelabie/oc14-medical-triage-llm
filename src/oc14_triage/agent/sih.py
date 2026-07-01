"""FHIR-shaped mock of a hospital-information-system (SIH) record — POC integration only.

This is NOT a live SIH connection: `to_fhir` just *shapes* a triage case into a minimal
FHIR R4 Bundle (Encounter + Observation) so the deliverable can demonstrate what a real
SIH push would look like. The subject reference is deliberately PSEUDONYMOUS — the synthetic
`session_id` only, never a patient name — to keep the "no PII stored" invariant intact.
"""

from __future__ import annotations

# Local, invented coding system for the 3-level triage scale (not a real terminology server).
URGENCY_SYSTEM = "https://chsa.local/fhir/urgence"
URGENCY_CODING = {
    "urgence maximale": {"system": URGENCY_SYSTEM, "code": "max", "display": "urgence maximale"},
    "urgence modérée": {"system": URGENCY_SYSTEM, "code": "mod", "display": "urgence modérée"},
    "urgence différée": {"system": URGENCY_SYSTEM, "code": "diff", "display": "urgence différée"},
}


def to_fhir(case: dict) -> dict:
    """Shape a triage `case` into a mock FHIR collection Bundle (Encounter + Observation)."""
    subject = {"reference": f"Patient/{case['session_id']}"}
    encounter = {
        "resourceType": "Encounter",
        "id": case["interaction_id"],
        "status": "finished",
        "class": {
            "system": "http://terminology.hl7.org/CodeSystem/v3-ActCode",
            "code": "EMER",
            "display": "emergency",
        },
        "subject": subject,
    }
    # An unparseable model output can leave urgency=None; don't index URGENCY_CODING with it
    # (KeyError). Encode the coding only when known, else fall back to a plain "indéterminé".
    coding = URGENCY_CODING.get(case.get("urgency"))
    value = {"coding": [coding]} if coding else {"text": "indéterminé"}
    observation = {
        "resourceType": "Observation",
        "status": "final",
        "subject": subject,
        "effectiveDateTime": case.get("timestamp_utc"),
        "valueCodeableConcept": value,
        "note": [{"text": "POC mock — not a live SIH record."}],
    }
    return {
        "resourceType": "Bundle",
        "type": "collection",
        "entry": [{"resource": encounter}, {"resource": observation}],
    }
