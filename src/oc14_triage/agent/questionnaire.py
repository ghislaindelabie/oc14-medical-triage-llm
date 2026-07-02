"""Rule-guided ADAPTIVE questionnaire for the triage collecte phase.

Deliberately a small rule table + one conditional follow-up, NOT an open dialogue engine:
we gather three core fields (motif / debut / intensite) in order, and when the chief
complaint carries a red-flag cue we insert ONE targeted follow-up before finishing. This
keeps the collecte deterministic and testable — the LLM does the reasoning downstream.
"""

from __future__ import annotations

from .state import RED_FLAGS

# Core fields, asked in this order. "motif" = chief complaint, "debut" = onset,
# "intensite" = severity on a 1-10 scale.
CORE_FIELDS = ("motif", "debut", "intensite")
# Optional trailing fields: offered once, after the core fields. A BLANK answer skips them
# (the service treats a blank on an optional field as "skip", not "re-ask"), so they never
# block completion. "vitals" = free-text vital signs (T°, TA, pouls, SpO₂) — a strong triage
# signal the model can reason over; injected into the case text and stored as `constantes`.
OPTIONAL_FIELDS = ("vitals",)

_QUESTIONS = {
    "fr": {
        "motif": "Quel est le motif de consultation ? Décrivez ce qui vous amène.",
        "debut": "Depuis quand ces symptômes ont-ils débuté ?",
        "intensite": "Sur une échelle de 1 à 10, quelle est l'intensité de vos symptômes ?",
        "vitals": "Si vous les connaissez, indiquez vos constantes — température, tension, "
                  "pouls, saturation (SpO₂). Sinon, laissez vide pour continuer.",
    },
    "en": {
        "motif": "What is the reason for your visit? Describe what brings you in.",
        "debut": "When did these symptoms begin (onset)?",
        "intensite": "On a scale of 1 to 10, how severe are your symptoms?",
        "vitals": "If known, enter your vital signs — temperature, blood pressure, pulse, "
                  "oxygen saturation (SpO₂). Otherwise leave blank to continue.",
    },
}

# One targeted follow-up per red-flag cue that has clinically-useful modifiers to probe.
# Cues not listed here still escalate downstream but don't add a collecte question.
_FOLLOWUP = {
    "fr": {
        "douleur thoracique": "La douleur irradie-t-elle (bras, mâchoire) ? Avez-vous des "
                              "sueurs ou un essoufflement ?",
        "douleur à la poitrine": "La douleur irradie-t-elle (bras, mâchoire) ? Avez-vous des "
                                 "sueurs ou un essoufflement ?",
    },
    "en": {
        "chest pain": "Does the pain radiate (arm, jaw)? Do you have sweating or shortness "
                      "of breath?",
    },
}


def detect_red_flags(text: str, lang: str = "fr") -> list[str]:
    """Substring-match the chief complaint against the shared RED_FLAGS vocabulary."""
    low = (text or "").lower()
    return [cue for cue in RED_FLAGS[lang] if cue in low]


def _followup_cue(motif: str, lang: str) -> str | None:
    """First detected red-flag cue that has a targeted follow-up question, or None."""
    followups = _FOLLOWUP[lang]
    return next((cue for cue in detect_red_flags(motif, lang) if cue in followups), None)


_TEMPLATE = {
    "fr": {
        "motif": "Motif : {}.",
        "debut": "Début : {}.",
        "intensite": "Intensité (1-10) : {}.",
        "followup": "Précision : {}.",
        "vitals": "Constantes : {}.",
    },
    "en": {
        "motif": "Chief complaint: {}.",
        "debut": "Onset: {}.",
        "intensite": "Severity (1-10): {}.",
        "followup": "Detail: {}.",
        "vitals": "Vitals: {}.",
    },
}


def is_optional(field: str) -> bool:
    """True for fields the collecte offers but does not require (a blank answer skips them)."""
    return field in OPTIONAL_FIELDS


def is_complete(answers: dict) -> bool:
    """True once the collecte has nothing left to ask — the core fields are gathered AND the
    optional steps (vitals) have been offered (answered or explicitly skipped)."""
    return next_field(answers) is None


def assemble_case_text(answers: dict, lang: str = "fr") -> str:
    """Compose a short clinical free-text from the gathered answers — the model input.

    Ordered motif -> followup -> debut -> intensite; only non-empty fields are emitted so a
    partially-filled dossier still yields usable text."""
    order = ("motif", "followup", "debut", "intensite", "vitals")
    tmpl = _TEMPLATE[lang]
    parts = [tmpl[f].format(str(answers[f]).strip()) for f in order
             if f in answers and str(answers[f]).strip()]
    return " ".join(parts)


def next_field(answers: dict, lang: str = "fr") -> str | None:
    """Return the KEY of the next field to fill (or None when complete).

    Order: motif first; then — if the chief complaint carries a red-flag cue with a known
    follow-up not yet asked (tracked via the "followup" key) — the "followup" field; then
    the remaining core fields (debut, intensite). The service stores each answer under this
    key, so it stays the single source of truth for both the key and (via next_question) the
    text."""
    if "motif" not in answers:
        return "motif"
    if "followup" not in answers and _followup_cue(answers["motif"], lang) is not None:
        return "followup"
    for field in CORE_FIELDS:
        if field not in answers:
            return field
    for field in OPTIONAL_FIELDS:
        if field not in answers:
            return field
    return None


def _question_for(field: str, answers: dict, lang: str) -> str:
    if field == "followup":
        cue = _followup_cue(answers.get("motif", ""), lang)
        if cue is not None:
            return _FOLLOWUP[lang][cue]
    return _QUESTIONS[lang][field]


def next_question(answers: dict, lang: str = "fr") -> str | None:
    """Return the text of the next question, or None when the core fields are gathered."""
    field = next_field(answers, lang)
    return _question_for(field, answers, lang) if field else None
