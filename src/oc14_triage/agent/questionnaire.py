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

_QUESTIONS = {
    "fr": {
        "motif": "Quel est le motif de consultation ? Décrivez ce qui vous amène.",
        "debut": "Depuis quand ces symptômes ont-ils débuté ?",
        "intensite": "Sur une échelle de 1 à 10, quelle est l'intensité de vos symptômes ?",
    },
    "en": {
        "motif": "What is the reason for your visit? Describe what brings you in.",
        "debut": "When did these symptoms begin (onset)?",
        "intensite": "On a scale of 1 to 10, how severe are your symptoms?",
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


_TEMPLATE = {
    "fr": {
        "motif": "Motif : {}.",
        "debut": "Début : {}.",
        "intensite": "Intensité (1-10) : {}.",
        "followup": "Précision : {}.",
    },
    "en": {
        "motif": "Chief complaint: {}.",
        "debut": "Onset: {}.",
        "intensite": "Severity (1-10): {}.",
        "followup": "Detail: {}.",
    },
}


def is_complete(answers: dict) -> bool:
    """True once the three core fields (motif, debut, intensite) are gathered."""
    return all(f in answers for f in CORE_FIELDS)


def assemble_case_text(answers: dict, lang: str = "fr") -> str:
    """Compose a short clinical free-text from the gathered answers — the model input.

    Ordered motif -> followup -> debut -> intensite; only non-empty fields are emitted so a
    partially-filled dossier still yields usable text."""
    order = ("motif", "followup", "debut", "intensite")
    tmpl = _TEMPLATE[lang]
    parts = [tmpl[f].format(str(answers[f]).strip()) for f in order
             if f in answers and str(answers[f]).strip()]
    return " ".join(parts)


def next_question(answers: dict, lang: str = "fr") -> str | None:
    """Return the next question to ask, or None when the core fields are gathered.

    Order: motif first; then — if the chief complaint carries a red-flag cue with a known
    follow-up and it hasn't been asked yet (tracked via the "followup" key) — ONE targeted
    follow-up; then the remaining core fields (debut, intensite) in order."""
    if "motif" not in answers:
        return _QUESTIONS[lang]["motif"]

    if "followup" not in answers:
        for cue in detect_red_flags(answers["motif"], lang):
            if cue in _FOLLOWUP[lang]:
                return _FOLLOWUP[lang][cue]

    for field in CORE_FIELDS:
        if field not in answers:
            return _QUESTIONS[lang][field]

    return None
