"""Central configuration: paths, taxonomy, prompts, model ids, dataset targets.

Single source of truth so the data-prep, eval and serving layers agree on the
triage taxonomy, the response structure, and the system prompts. Plain data +
small helpers only — no I/O, no heavy imports.
"""

from __future__ import annotations

from pathlib import Path

# --- Paths -------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"  # untouched downloads, one folder per source
INTERIM = DATA / "interim"  # normalised but not yet split
PROCESSED = DATA / "processed"  # final train/val/test JSONL ready for training
CARDS = DATA / "cards"  # provenance + GDPR data cards (versioned)
EVAL_DIR = DATA / "eval"  # hand-written clinical-eval vignettes + probes

# --- Languages ---------------------------------------------------------------
LANGS = ("fr", "en")
# Business context is a French hospital: French must be well covered (mentor guidance).
LANG_TARGET = {"fr": 0.80, "en": 0.20}

# --- Triage taxonomy (3-level French scale, exactly the brief's wording) -----
# The model's PRIMARY task is triage: assign one of these, justify it, recommend an action.
URGENCY_LEVELS = ("urgence maximale", "urgence modérée", "urgence différée")
URGENCY_EN_GLOSS = {
    "urgence maximale": "maximal urgency (immediate / life-threatening)",
    "urgence modérée": "moderate urgency (should be seen soon)",
    "urgence différée": "deferred urgency (non-urgent / can wait)",
}

# --- Models ------------------------------------------------------------------
# We fine-tune BOTH and compare; Base is the primary served deliverable (brief names it),
# Instruct is the comparison arm (see docs/research/00-OVERALL-APPROACH.md §0).
MODEL_BASE = "Qwen/Qwen3-1.7B-Base"
MODEL_INSTRUCT = "Qwen/Qwen3-1.7B"
PRIMARY_MODEL = MODEL_BASE

# --- System prompts ----------------------------------------------------------
# Triage-first framing + hard safety rules. Injected at training time AND enforced
# by the serving wrapper so they are never optional.
SYSTEM_PROMPT_FR = (
    "Tu es un assistant de triage médical pour le service des urgences du Centre "
    "Hospitalier Saint-Aurélien (CHSA). Tu assistes le personnel soignant ; tu ne "
    "remplaces jamais un professionnel de santé.\n"
    "Pour chaque situation décrite, réponds dans la langue de la question et structure "
    "ta réponse ainsi :\n"
    "1. Niveau d'urgence : urgence maximale / urgence modérée / urgence différée.\n"
    "2. Justification clinique : explique brièvement les éléments qui motivent ce niveau.\n"
    "3. Recommandation : l'action à entreprendre.\n"
    "Règles de sécurité : signale immédiatement tout signe d'alerte (douleur thoracique, "
    "détresse respiratoire, signes neurologiques aigus, etc.) comme urgence maximale ; "
    "ne pose jamais de diagnostic définitif et ne prescris aucun médicament ; en cas de "
    "doute, oriente vers une évaluation médicale. Termine par un bref avertissement "
    "rappelant que cet avis ne remplace pas une consultation médicale."
)
SYSTEM_PROMPT_EN = (
    "You are a medical triage assistant for the emergency department of the Centre "
    "Hospitalier Saint-Aurélien (CHSA). You support clinical staff; you never replace a "
    "healthcare professional.\n"
    "For each described situation, answer in the language of the question and structure "
    "your answer as:\n"
    "1. Urgency level: urgence maximale / urgence modérée / urgence différée.\n"
    "2. Clinical justification: briefly explain what drives this level.\n"
    "3. Recommendation: the action to take.\n"
    "Safety rules: immediately flag any red-flag sign (chest pain, respiratory distress, "
    "acute neurological signs, etc.) as urgence maximale; never give a definitive "
    "diagnosis and never prescribe medication; when in doubt, refer to a medical "
    "evaluation. End with a short disclaimer that this advice does not replace a medical "
    "consultation."
)
SYSTEM_PROMPT = {"fr": SYSTEM_PROMPT_FR, "en": SYSTEM_PROMPT_EN}

# --- Dataset targets (see docs/research/04 + §0b mentor refinements) ----------
SFT_TARGET = 5000  # instruction-response pairs (train), ~80% FR / 20% EN
TRIAGE_MIN = 1200  # of which: triage-structured rows (mentor: triage must be central)
DPO_TARGET = 1500  # preference pairs: ~half hand-written bilingual safety + UltraMedical
VAL_FRACTION = 0.10
SEED = 3407  # Unsloth's canonical seed; reused everywhere for reproducibility
