# OC14 — Assistant de triage médical (POC)

Proof-of-concept d'un **assistant de triage médical bilingue (FR/EN)** pour le service des
urgences d'un hôpital fictif (CHSA), construit en spécialisant **Qwen3-1.7B** par
**SFT + LoRA** puis **DPO**, servi via **vLLM** et exposé par une API.

> Projet de formation OpenClassrooms (AI Engineer). La tâche **centrale est le triage**
> (prioriser l'urgence, justifier cliniquement, recommander une action), pas le Q&A médical
> général. Voir le rapport de recherche et le plan ci-dessous.

> **Résultat (éval honnête, gold stratifié n=300, greedy, sans fuite) :** macro-F1 **0,19 (Base non entraîné) → 0,82 (SFT v9, livré)**. DPO testé (0,80) puis écarté (résultat honnête). Positionné comme **aide à la décision / human-in-the-loop**, pas triage autonome.

## Documents (source de vérité = Markdown)

- **Approche & plan** : `docs/research/00-OVERALL-APPROACH.md` (+ `01`–`09` deep-dives, `09` red-team).
- **Plan d'implémentation** : `IMPLEMENTATION_PLAN.md`.
- **Suivi d'expériences (W&B)** : https://wandb.ai/ghislaindelabie/oc14-triage-eval — *tableau de comparaison des résultats d'évaluation* (base/SFT/DPO).
- **Carte de données / RGPD** : `data/cards/DATA_CARD.md` (auto-générée).

> Lecture mobile (tailnet privé) : `https://p710.tail3089b5.ts.net:8445/doc/oc14-finetune-llm-report`.

## Structure

```
src/oc14_triage/
  config.py            # taxonomie 3 niveaux, prompts système, ids modèles, cibles dataset
  data/                # sources.py · download.py · templates.py · build_sft.py · build_dpo.py
                       # vignettes.py (triage écrit à la main, bilingue) · card.py (RGPD)
  eval/metrics.py      # métriques triage-first (accord d'urgence, escalade, disclaimer, langue…)
  serving/             # wrapper FastAPI /triage devant vLLM (construit + testé ; endpoint live à déployer)
data/                  # raw/ interim/ processed/ — GITIGNORÉ ; data/cards/ versionné
tests/                 # pytest (templates, metrics, sanity dataset)
notebooks/             # entraînement + éval Kaggle (SFT LoRA, DPO, base) — exécutés ; générés par build_kaggle_notebooks.py
.github/workflows/     # CI : ruff + pytest
```

## Démarrage (préparation des données — CPU, aucune carte GPU requise)

Pipeline livré (jeu de données embarqué dans `data/kaggle_upload/`) :

```bash
uv sync                                          # venv isolé
uv run python -m oc14_triage.data.download       # télécharge les sources -> data/raw/ + inventaire
uv run python scripts/build_retrain_sft.py       # -> data/kaggle_upload/sft_{train,val}.jsonl (LLM-labelled)
uv run python scripts/build_dpo_pairs.py         # -> data/kaggle_upload/dpo_{train,val}.jsonl (paires triage équilibrées)
uv run python -m oc14_triage.data.card           # -> data/cards/DATA_CARD.md
uv run pytest                                     # tests
```

<details><summary>Ancien pipeline (jeu v1, remplacé — <code>data/processed/</code>, retiré)</summary>

```bash
uv run python -m oc14_triage.data.build_sft      # -> data/processed/sft_{train,val}.jsonl
uv run python -m oc14_triage.data.build_dpo      # -> data/processed/dpo_{train,val}.jsonl
```

</details>

## État

Voir `IMPLEMENTATION_PLAN.md` § "État actuel". En bref : **données + entraînement + éval +
serving (wrapper) + CI : faits** ; restent : endpoint live (clé RunPod/Modal), passe
Presidio/RGPD, courbes d'entraînement live W&B, rapport ≤20 p. Identifiants requis listés
dans le plan. Le dépôt est **public** : https://github.com/ghislaindelabie/oc14-medical-triage-llm.
