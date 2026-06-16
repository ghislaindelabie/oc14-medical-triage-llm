"""Turn raw rows into chat examples, and build the triage response structure.

We store examples in the **messages** format ([{role, content}, ...]) rather than
pre-rendering ChatML text: the training notebook applies the model's own chat
template with `tokenizer.apply_chat_template`, which keeps us correct on the
Qwen3 stop-token gotcha (the real eos_token is <|endoftext|>; <|im_end|> is the
turn terminator) — see docs/research/01 and §0b. Pure functions, fully testable.
"""

from __future__ import annotations

from ..config import SYSTEM_PROMPT, URGENCY_LEVELS

# Red-flag symptoms → "urgence maximale". Lowercased substring match. Honest heuristic,
# documented as such; eval vignettes are hand-labelled independently (no circularity).
RED_FLAGS = {
    "fr": (
        "douleur thoracique", "douleur à la poitrine", "détresse respiratoire",
        "difficulté à respirer", "perte de connaissance", "inconscioence", "convulsion",
        "hémorragie", "saigne abondamment", "avc", "paralysie", "aphasie", "déficit neurologique",
        "anaphylax", "choc", "arrêt cardiaque", "douleur abdominale intense", "septic",
        "suicidaire", "overdose", "intoxication grave",
    ),
    "en": (
        "chest pain", "respiratory distress", "shortness of breath", "trouble breathing",
        "loss of consciousness", "unconscious", "seizure", "hemorrhage", "heavy bleeding",
        "stroke", "paralysis", "aphasia", "neurological deficit", "anaphylax", "shock",
        "cardiac arrest", "severe abdominal pain", "sepsis", "suicidal", "overdose",
    ),
}

# Subjects/keywords that lean non-urgent → "urgence différée".
DEFERRED_HINTS = {
    "fr": ("prévention", "dépistage", "suivi", "chronique", "information", "vaccination",
           "renouvellement", "conseil"),
    "en": ("prevention", "screening", "follow-up", "chronic", "information", "vaccination",
           "education", "support group"),
}

DISCLAIMER = {
    "fr": "⚠️ Cet avis d'orientation ne remplace pas une consultation médicale.",
    "en": "⚠️ This triage guidance does not replace a medical consultation.",
}

# Canned, per-level recommendations (used when reshaping QA rows into triage form).
RECO = {
    "fr": {
        "urgence maximale": "Prise en charge immédiate : alerter sans délai un médecin urgentiste.",
        "urgence modérée": "À évaluer rapidement par un soignant ; surveiller l'évolution.",
        "urgence différée": "Peut être orienté vers une consultation non urgente.",
    },
    "en": {
        "urgence maximale": "Immediate management: alert an emergency physician without delay.",
        "urgence modérée": "To be assessed promptly by a clinician; monitor for changes.",
        "urgence différée": "Can be directed to a non-urgent consultation.",
    },
}


def heuristic_urgency(text: str, lang: str = "fr") -> str:
    """Best-effort urgency from symptom text. HEURISTIC — flagged as such in the report."""
    low = (text or "").lower()
    flags = RED_FLAGS.get(lang, ()) + RED_FLAGS["en"]  # check both: bilingual robustness
    if any(f in low for f in flags):
        return "urgence maximale"
    hints = DEFERRED_HINTS.get(lang, ()) + DEFERRED_HINTS["en"]
    if any(h in low for h in hints):
        return "urgence différée"
    return "urgence modérée"


def triage_response(level: str, justification: str, recommendation: str, lang: str = "fr") -> str:
    """The canonical 3-part triage answer + disclaimer."""
    assert level in URGENCY_LEVELS, f"unknown urgency level: {level!r}"
    if lang == "fr":
        return (
            f"1. Niveau d'urgence : {level}.\n"
            f"2. Justification clinique : {justification}\n"
            f"3. Recommandation : {recommendation}\n"
            f"{DISCLAIMER['fr']}"
        )
    return (
        f"1. Urgency level: {level}.\n"
        f"2. Clinical justification: {justification}\n"
        f"3. Recommendation: {recommendation}\n"
        f"{DISCLAIMER['en']}"
    )


def chat_example(user: str, assistant: str, lang: str, source: str, kind: str) -> dict:
    """One SFT row in messages format, with metadata for traceability + split stats."""
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT[lang]},
            {"role": "user", "content": user.strip()},
            {"role": "assistant", "content": assistant.strip()},
        ],
        "lang": lang,
        "source": source,
        "kind": kind,  # "qa" (medical knowledge) | "triage" (urgency + justification + reco)
    }
