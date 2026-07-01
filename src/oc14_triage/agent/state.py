"""Shared contract for the agentic triage chain — the single source of truth the graph
nodes and the leaf modules (anonymisation, questionnaire, backend, store, sih) agree on.

Pure data/types: no behaviour here, so nothing to unit-test. Every module that DOES have
behaviour is built test-first against these shapes.
"""

from __future__ import annotations

from typing import Literal, TypedDict

from ..config import URGENCY_LEVELS  # ("urgence maximale", "urgence modérée", "urgence différée")

Urgency = Literal["urgence maximale", "urgence modérée", "urgence différée"]

# Red-flag cues → force / justify *urgence maximale* (safety override). Kept minimal and
# clinically obvious; the questionnaire module owns detection, this is the shared vocabulary.
RED_FLAGS = {
    "fr": [
        "douleur thoracique", "douleur à la poitrine", "détresse respiratoire",
        "difficulté à respirer", "perte de connaissance", "inconscient", "convulsion",
        "hémorragie", "saigne abondamment", "paralysie", "trouble de la parole",
        "raideur de la nuque", "douleur abdominale intense", "idées suicidaires",
    ],
    "en": [
        "chest pain", "shortness of breath", "respiratory distress", "unconscious",
        "loss of consciousness", "seizure", "hemorrhage", "heavy bleeding", "paralysis",
        "slurred speech", "stiff neck", "severe abdominal pain", "suicidal",
    ],
}


class TriageCase(TypedDict, total=False):
    """LangGraph state. After the `anonymisation` node, `raw_text` is cleared and every
    downstream node reads `anon_text` ONLY — the invariant that proves "no PII stored"."""

    # identity / tracing (synthetic UUIDs — not derived from patient attributes)
    session_id: str
    interaction_id: str
    lang: str  # "fr" | "en"

    # collecte
    answers: dict  # questionnaire field -> value
    pending_question: str | None
    complete: bool

    # anonymisation (the invariant boundary)
    raw_text: str  # TRANSIENT — present only until the anonymisation node runs, then removed
    anon_text: str  # everything downstream uses this
    input_sha256: str  # one-way hash of the raw input, for traceability WITHOUT retention
    pii_entities: list  # [{"type": "PERSON", "count": 1}, ...] found by Presidio

    # prétraitement / validation
    red_flags: list  # matched red-flag cues in anon_text
    valid: bool
    validation_error: str | None

    # triage (LLM) + explication
    model_output: str  # raw model text
    model_version: str
    urgency: Urgency | None
    justification: str
    recommendation: str
    disclaimer_present: bool

    # persistance / SIH
    timestamp_utc: str
    latency_ms: float
    trace: list  # [{"node": str, "ms": float}, ...] — per-node timing
    sih_record: dict  # FHIR-shaped Encounter+Observation


# The persisted "dossier patient" schema (SQLite columns == FHIR-shaped fields).
# Every free-text field here is POST-anonymisation by construction. Referenced by the
# store + sih modules so their tests assert the same field set.
DOSSIER_FIELDS = (
    "session_id", "interaction_id", "timestamp_utc", "model_version",
    "symptoms_anon", "antecedents_anon", "constantes",
    "urgency", "justification", "recommandation_anon",
    "source", "confidence_level", "input_sha256",
    "disclaimer_present", "latency_ms", "deleted",
)

__all__ = ["Urgency", "TriageCase", "RED_FLAGS", "DOSSIER_FIELDS", "URGENCY_LEVELS"]
