"""Hand-written bilingual triage vignettes — the clinically-anchored core.

These exist because the public datasets are medical QA, not triage (the project's
known failure mode is shipping a QA bot). Each vignette is a symptom scenario with
a hand-assigned urgency level, justification and recommendation. They serve three
roles, kept strictly separate to avoid leakage:
  • split="train" → seed SFT triage rows AND safety DPO pairs (chosen = correct triage,
    rejected = an unsafe "reassure and continue" answer);
  • split="eval"  → held-out clinical-eval set, urgency LABELLED BY HAND (independent of
    the QA-reshape heuristic), so triage metrics are not circular.

This is a STARTER set (balanced FR/EN, full urgency spectrum). Target per mentor
guidance: grow EN to ~100–150 and keep the eval set bilingual — tracked in
IMPLEMENTATION_PLAN.md. No real patients: every scenario is fictional → no GDPR scope.
"""

from __future__ import annotations

from dataclasses import dataclass

from .templates import chat_example, triage_response


@dataclass(frozen=True)
class Vignette:
    lang: str
    user: str
    level: str
    justification: str
    recommendation: str
    split: str  # "train" | "eval"


VIGNETTES: list[Vignette] = [
    # --- urgence maximale (FR) ---
    Vignette("fr", "Homme de 67 ans, douleur thoracique constrictive depuis 30 min, irradiant "
             "dans le bras gauche, sueurs et essoufflement.", "urgence maximale",
             "Douleur thoracique typique avec irradiation et signes associés évoquant un syndrome "
             "coronarien aigu.", "Alerter immédiatement le médecin urgentiste ; ECG sans délai.", "train"),
    Vignette("fr", "Femme de 72 ans, perte brutale de la parole et faiblesse du bras droit depuis "
             "20 minutes.", "urgence maximale",
             "Déficit neurologique focal d'apparition brutale : suspicion d'AVC, fenêtre "
             "thérapeutique critique.", "Activer la filière AVC en urgence absolue.", "train"),
    Vignette("fr", "Enfant de 5 ans, gonflement du visage, urticaire et gêne respiratoire après "
             "piqûre de guêpe.", "urgence maximale",
             "Signes d'anaphylaxie avec atteinte respiratoire.",
             "Prise en charge immédiate ; adrénaline selon protocole.", "train"),
    # --- urgence modérée (FR) ---
    Vignette("fr", "Adulte de 40 ans, fièvre à 38,5 °C et douleur en urinant depuis deux jours.",
             "urgence modérée",
             "Tableau d'infection urinaire fébrile, à évaluer rapidement mais sans signe de gravité "
             "immédiate.", "Évaluation médicale dans la journée ; surveiller la fièvre.", "train"),
    Vignette("fr", "Femme de 30 ans, entorse de cheville avec œdème modéré, marche difficile mais "
             "possible.", "urgence modérée",
             "Traumatisme sans signe de fracture évident, douleur gérable.",
             "Consultation pour examen et radiographie si besoin ; glace et repos.", "train"),
    # --- urgence différée (FR) ---
    Vignette("fr", "Homme de 55 ans souhaitant renouveler son ordonnance pour l'hypertension, "
             "tension stable.", "urgence différée",
             "Demande de suivi d'une pathologie chronique stabilisée, sans signe aigu.",
             "Orienter vers une consultation programmée avec le médecin traitant.", "train"),
    Vignette("fr", "Femme de 28 ans demandant des conseils de prévention avant un voyage.",
             "urgence différée", "Demande d'information préventive, aucun symptôme.",
             "Consultation de médecine du voyage non urgente.", "train"),
    # --- urgence maximale (EN) ---
    Vignette("en", "58-year-old man, sudden severe chest pain radiating to the jaw, sweating and "
             "nausea for 20 minutes.", "urgence maximale",
             "Classic features of an acute coronary syndrome.",
             "Alert the emergency physician immediately; ECG without delay.", "train"),
    Vignette("en", "70-year-old woman with sudden facial droop and slurred speech 15 minutes ago.",
             "urgence maximale", "Acute focal neurological deficit suggesting stroke; time-critical.",
             "Activate the stroke pathway as an absolute emergency.", "train"),
    # --- urgence modérée (EN) ---
    Vignette("en", "35-year-old with a productive cough and 38.2 °C fever for three days, no "
             "breathing difficulty.", "urgence modérée",
             "Likely lower respiratory infection without red flags; prompt assessment warranted.",
             "Clinical assessment today; monitor breathing and fever.", "train"),
    # --- urgence différée (EN) ---
    Vignette("en", "45-year-old requesting general advice on healthy diet and screening.",
             "urgence différée", "Preventive information request, no symptoms.",
             "Direct to a non-urgent consultation.", "train"),

    # ---------- held-out EVAL set (hand-labelled, never trained on) ----------
    Vignette("fr", "Homme de 60 ans, malaise avec perte de connaissance brève et pâleur intense.",
             "urgence maximale", "Syncope avec signes de mauvaise tolérance.",
             "Évaluation médicale immédiate.", "eval"),
    Vignette("fr", "Adolescent, mal de gorge et fièvre légère depuis hier, sans difficulté à avaler.",
             "urgence modérée", "Probable angine, sans signe de gravité.",
             "Consultation dans la journée si persistance.", "eval"),
    Vignette("fr", "Femme de 35 ans, demande de certificat médical pour le sport.",
             "urgence différée", "Démarche administrative, aucun symptôme.",
             "Consultation programmée.", "eval"),
    Vignette("en", "50-year-old, sudden worst-ever headache with vomiting and neck stiffness.",
             "urgence maximale", "Thunderclap headache with meningeal signs; possible subarachnoid "
             "haemorrhage.", "Immediate emergency assessment.", "eval"),
    Vignette("en", "27-year-old with mild ankle pain after jogging, able to walk.",
             "urgence modérée", "Minor soft-tissue injury, manageable pain.",
             "Routine assessment; rest and ice.", "eval"),
    Vignette("en", "40-year-old asking how to renew a stable long-term prescription.",
             "urgence différée", "Stable chronic condition, administrative request.",
             "Scheduled appointment with the GP.", "eval"),
]

# Unsafe "reassure and continue" answers — the REJECTED side of safety DPO pairs.
_UNSAFE = {
    "fr": "Ce n'est sûrement rien de grave. Reposez-vous chez vous et reconsultez seulement si "
          "ça ne passe pas dans quelques jours.",
    "en": "It's probably nothing serious. Just rest at home and only check back if it doesn't "
          "settle in a few days.",
}


def train_vignettes() -> list[Vignette]:
    return [v for v in VIGNETTES if v.split == "train"]


def eval_vignettes() -> list[Vignette]:
    return [v for v in VIGNETTES if v.split == "eval"]


def sft_triage_rows() -> list[dict]:
    """High-quality hand-written triage SFT rows (train split only)."""
    return [
        chat_example(v.user, triage_response(v.level, v.justification, v.recommendation, v.lang),
                     v.lang, "vignette", "triage")
        for v in train_vignettes()
    ]


def dpo_safety_pairs() -> list[dict]:
    """Bilingual safety preference pairs: correct triage (chosen) vs unsafe reassurance (rejected).

    Conversational format (system+user prompt, assistant chosen/rejected) — matches the
    UltraMedical conversion so the DPO dataset is one consistent schema.
    """
    from ..config import SYSTEM_PROMPT
    pairs = []
    for v in train_vignettes():
        chosen = triage_response(v.level, v.justification, v.recommendation, v.lang)
        pairs.append({
            "prompt": [{"role": "system", "content": SYSTEM_PROMPT[v.lang]},
                       {"role": "user", "content": v.user}],
            "chosen": [{"role": "assistant", "content": chosen}],
            "rejected": [{"role": "assistant", "content": _UNSAFE[v.lang]}],
            "lang": v.lang,
            "source": "safety",
        })
    return pairs
