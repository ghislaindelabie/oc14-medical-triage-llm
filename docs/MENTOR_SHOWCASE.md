# OC14 — Manuel de présentation au mentor

Guide de démonstration (~20 min) : quoi montrer, dans quel ordre, et comment présenter le code
efficacement. Tout est **privé / tailnet** → la séance se fait en **partage d'écran** (tu pilotes, le
mentor regarde). Garde ce plan + la *Note de synthèse* ouverts à côté.

## Liens à ouvrir (onglets, avant la séance)
- **GitHub (codebase)** : https://github.com/ghislaindelabie/oc14-medical-triage-llm — branche `feat/data-prep-and-scaffold`
- **W&B (suivi d'expériences)** : https://wandb.ai/ghislaindelabie/oc14-triage-eval
- **Kaggle — entraînement SFT** : https://www.kaggle.com/code/ghislaindelabie/oc14-sft-lora-qwen3-1-7b
- **Kaggle — éval** : https://www.kaggle.com/code/ghislaindelabie/oc14-sft-eval · **dataset** : https://www.kaggle.com/datasets/ghislaindelabie/oc14-triage-data
- **Note de synthèse (toi, mobile)** : `…:8445/doc/oc14-mentor-brief` — ta antisèche

## Déroulé (≈20 min)

**1. Cadrage (2 min).** « POC d'assistant de **triage** médical bilingue (FR/EN) pour un hôpital fictif,
par spécialisation de **Qwen3-1.7B** (SFT+LoRA → DPO), servi en vLLM, CI/CD, rapport. » Annonce les
**5 livrables** (dataset RGPD, poids fine-tunés, endpoint cloud, CI/CD, rapport) et la décision centrale :
**triage, pas Q&A** ; **FR-primaire** ; modèle **Base**. → ouvrir la *Note de synthèse* (§1).

**2. Données & méthode (4 min) — le point fort.** Pas de clinicien → **labellisation par consensus de
3 LLM** (GPT-5.4 + Mistral + Sonnet) avec une **rubrique citée** (ESI/MTS/FRENCH/CIMU), 3 niveaux + ESI,
**Fleiss κ ≈ 0,67**. Montre `docs/TRIAGE_CRITERIA.md` (le cœur clinique) et `src/oc14_triage/labeling/`.
Mentionne RGPD (Presidio prévu, vignettes synthétiques, *data card*). Insiste : **standard argent assumé**.

**3. Entraînement (3 min) — le modèle.** Ouvre le notebook **oc14-sft-lora** (Kaggle) : SFT+LoRA
(Unsloth) sur **T4 gratuit**, gabarit ChatML + stop sur `<|im_end|>`, *train-on-responses-only*. C'est le
**livrable #2**. Dis un mot du **DPO** : tenté, **résultat honnête négatif** (améliore les extrêmes,
affaisse le milieu) → je conserve le SFT. *(Montrer qu'on maîtrise la technique ET qu'on sait diagnostiquer
un échec = un atout.)*

**4. Éval & rigueur (5 min) — le différenciateur.** Ouvre le **dashboard W&B** : la comparaison
**Base 0,19 → SFT 0,82** (et DPO 0,80) = le **vrai signal de progrès**. Explique l'éval : gold
**stratifié**, décodage **greedy** (déterministe), **macro-F1** + rappel/précision par classe + **IC de
Wilson** + matrice de confusion. Puis le moment fort : **l'audit adverse** (`docs/KNOWN_ISSUES.md`) qui a
détecté un **0,81 gonflé** (fuite + échantillonnage) → corrigé en **0,82 honnête**. Termine par les
**limites assumées** (`docs/REPORT_LIMITATIONS.md`) : circularité, sur-triage/barre de sécurité, FR-primaire.

**5. Serving & CI/CD (3 min).** Montre `serving/` : le **wrapper FastAPI `/triage`** (injection du prompt
système, stop `<|im_end|>`, clé API, log d'audit *sans* données patient) + le plan **vLLM/RunPod
serverless**. Montre **CI verte** sur GitHub (ruff + pytest à chaque push). *(Endpoint live = dernière
étape, en attente d'une clé RunPod — dis-le franchement.)*

**6. Clôture (3 min).** Le **rapport** (tenu comme journal de bord, `DEVELOPMENT_JOURNAL.md`), les
**prochaines étapes** (endpoint live, passe Presidio, rapport final), puis questions.

## Comment introduire le code efficacement
Ne pas ouvrir 30 fichiers. Donne la **carte**, puis 2-3 fichiers clés :
1. **Structure** (`README` / arbo) : `src/oc14_triage/` → `data/` · `labeling/` · `eval/` · `serving/` ;
   `notebooks/` (générés par `build_kaggle_notebooks.py`) ; `docs/` ; `scripts/` ; `tests/`.
2. **Les docs de pilotage comme fil conducteur** : `IMPLEMENTATION_PLAN` (quoi/où on en est),
   `DEVELOPMENT_JOURNAL` (chronologie + résultats), `ARCHITECTURE_AND_DECISIONS` (le *pourquoi*),
   `KNOWN_ISSUES` (l'audit), `REPORT_LIMITATIONS`, `TRIAGE_CRITERIA`.
3. **3 fichiers à montrer** : `labeling/rubric.py` (la rubrique), une cellule SFT du notebook,
   `eval/metrics.py` (macro-F1 + IC). 
4. **L'historique git** (commits conventionnels) + **CI verte** = workflow pro.

## À souligner (ce qui distingue le projet)
- **Honnêteté méthodologique** : audit adverse, fuites éliminées, IC reportés, **0,81 retiré** au profit
  d'un **0,82 reproductible** ; résultat **non survendu**.
- **Compréhension** : le DPO (négatif analysé), le compromis **sur-triage ↔ précision**, la **barre de
  sécurité** (0,90 de rappel = aide à la décision, **pas** triage autonome).
- **Pipeline complet** : données → entraînement → éval → serving → CI/CD, sur **GPU gratuit**.

## Les 5 livrables — où chacun se voit
1. **Dataset RGPD** → `data/` + `TRIAGE_CRITERIA` + dataset Kaggle (+ Presidio à finaliser).
2. **Poids fine-tunés** → notebook SFT + W&B + merge 16-bit.
3. **Endpoint cloud** → `serving/` (wrapper + plan RunPod ; live = dernière étape).
4. **CI/CD** → `.github/workflows/` (verte) + déploiement à câbler.
5. **Rapport** → `DEVELOPMENT_JOURNAL` / la *Note de synthèse* / le rapport ≤20 p.
