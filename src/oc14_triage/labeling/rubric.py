"""The triage rubric (medical core) + the labelling prompt + the ESI<->3 mapping.

3-level French scale (urgence maximale / modérée / différée) grounded in the published
5-level emergency-triage scales, with the level boundaries pre-collapsed here (once,
transparently) so an LLM applies a single 3-level rule set and *also* returns the ESI 1-5
equivalent (free cross-check + comparability to ESI datasets).

Sources the criteria are distilled from (verify exact editions before publishing):
- ESI — Emergency Severity Index, AHRQ handbook (Gilboy et al.) — 5-level, the explicit
  4-decision-point algorithm + vital-sign danger zones.
- MTS — Manchester Triage System (Mackway-Jones et al.) — presentational red-flag discriminators.
- FRENCH — French Emergency Nurses Classification in Hospital (Taboulet et al., SFMU) — FR 5-level scale.
- CIMU — Classification Infirmière des Malades aux Urgences (Fourestié et al.) — FR 5-level scale.
Red-flag discriminators are universal across these scales; the French scales fix the taxonomy.
"""

from __future__ import annotations

URGENCY_LEVELS = ("urgence maximale", "urgence modérée", "urgence différée")

# ESI 1-5 -> our 3 levels (standard monotonic collapse: 1-2 emergent, 3 urgent, 4-5 non-urgent).
_ESI_TO_URGENCY = {1: "urgence maximale", 2: "urgence maximale",
                   3: "urgence modérée",
                   4: "urgence différée", 5: "urgence différée"}


def esi_to_urgency(esi: int) -> str:
    """Map an ESI 1-5 level to our 3-level scale."""
    return _ESI_TO_URGENCY[int(esi)]


# The criteria block injected into the labelling prompt (French — matches the cases + product).
TRIAGE_RUBRIC = """\
Tu es un classificateur expert de triage aux urgences. Applique STRICTEMENT le référentiel
ci-dessous (échelle française à 3 niveaux, alignée sur ESI 1-5 / FRENCH / CIMU / Manchester).

NIVEAUX (avec équivalent ESI) :
1) urgence maximale  (ESI 1-2) — menace vitale/fonctionnelle, ne peut pas attendre.
   Signes d'alerte (UN SEUL suffit) : arrêt cardio-respiratoire ou inconscience ; détresse
   respiratoire / SpO2 < 90% ; douleur thoracique évoquant un syndrome coronarien ; signes
   d'AVC (asymétrie faciale, déficit moteur, trouble du langage) ; hémorragie active sévère ;
   anaphylaxie ; sepsis/choc (PAS < 90, marbrures) ; trouble de conscience (Glasgow < 13) ;
   convulsion en cours ; traumatisme grave ; idées suicidaires avec plan ; constantes
   critiques (FC > 130 ou < 40, FR > 30, T° > 40 avec sepsis).
2) urgence modérée  (ESI 3) — symptomatique, à évaluer rapidement, stable, sans signe d'alerte.
   Ex. : fièvre avec foyer infectieux, douleur modérée (abdominale sans défense), déshydratation
   légère/modérée, suspicion de fracture avec appui possible, décompensation stable d'une
   pathologie chronique. Constantes proches de la normale.
3) urgence différée  (ESI 4-5) — non urgent, ambulatoire, mineur ou administratif.
   Ex. : plainte mineure (mal de gorge léger, entorse marchant), renouvellement d'ordonnance,
   certificat médical, conseil de prévention, suivi chronique stable, demande d'information.

PROCÉDURE :
(a) Cherche un signe d'alerte -> si présent : urgence maximale (ESI 1-2).
(b) Sinon, tableau aigu nécessitant une évaluation rapide -> urgence modérée (ESI 3).
(c) Sinon, mineur/administratif -> urgence différée (ESI 4-5).
En cas de doute, sur-trie (choisis le niveau plus urgent).

Si le texte n'est PAS une présentation clinique de patient (p. ex. question d'examen théorique,
mécanisme, pharmacologie), mets "is_triage_case": false.

Réponds UNIQUEMENT par un objet JSON, sans texte autour :
{"is_triage_case": true|false, "urgency": "urgence maximale|urgence modérée|urgence différée",
 "esi": 1-5, "red_flags": ["..."], "justification": "<= 25 mots, en français"}
L'ESI doit être cohérent avec le niveau (1-2=maximale, 3=modérée, 4-5=différée).
"""

SYSTEM_PROMPT = (
    "Tu es un classificateur de triage médical rigoureux. Tu appliques un référentiel donné "
    "et tu réponds exclusivement par du JSON valide, sans commentaire."
)


def build_user_prompt(case_text: str) -> str:
    """Rubric + the patient presentation to classify."""
    return f"{TRIAGE_RUBRIC}\n\nPRÉSENTATION DU PATIENT :\n{case_text.strip()}\n\nClasse cette présentation."


# Keys expected in each model's JSON answer.
REQUIRED_KEYS = ("is_triage_case", "urgency", "esi")
