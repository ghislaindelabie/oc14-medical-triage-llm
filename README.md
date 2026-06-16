# OC14 — Assistant de triage médical (POC)

Proof-of-concept d'un **assistant de triage médical bilingue (FR/EN)** pour le service des
urgences d'un hôpital fictif (CHSA), construit en spécialisant **Qwen3-1.7B** par
**SFT + LoRA** puis **DPO**, servi via **vLLM** et exposé par une API.

> Projet de formation OpenClassrooms (AI Engineer). La tâche **centrale est le triage**
> (prioriser l'urgence, justifier cliniquement, recommander une action), pas le Q&A médical
> général. Voir le rapport de recherche et le plan ci-dessous.

## Documents (source de vérité = Markdown)

- **Approche & plan** : `docs/research/00-OVERALL-APPROACH.md` (+ `01`–`09` deep-dives, `09` red-team).
- **Plan d'implémentation** : `IMPLEMENTATION_PLAN.md`.
- **Lecture mobile** (tailnet) : `https://p710.tail3089b5.ts.net:8445/doc/oc14-finetune-llm-report`.
- **Carte de données / RGPD** : `data/cards/DATA_CARD.md` (auto-générée).

## Structure

```
src/oc14_triage/
  config.py            # taxonomie 3 niveaux, prompts système, ids modèles, cibles dataset
  data/                # sources.py · download.py · templates.py · build_sft.py · build_dpo.py
                       # vignettes.py (triage écrit à la main, bilingue) · card.py (RGPD)
  eval/metrics.py      # métriques triage-first (accord d'urgence, escalade, disclaimer, langue…)
  serving/             # (à venir) wrapper FastAPI devant vLLM
data/                  # raw/ interim/ processed/ — GITIGNORÉ ; data/cards/ versionné
tests/                 # pytest (templates, metrics, sanity dataset)
notebooks/             # (à venir) entraînement Kaggle/Colab (SFT LoRA, DPO)
.github/workflows/     # CI : ruff + pytest
```

## Démarrage (préparation des données — CPU, aucune carte GPU requise)

```bash
uv sync                                          # venv isolé
uv run python -m oc14_triage.data.download       # télécharge les sources -> data/raw/ + inventaire
uv run python -m oc14_triage.data.build_sft      # -> data/processed/sft_{train,val}.jsonl
uv run python -m oc14_triage.data.build_dpo      # -> data/processed/dpo_{train,val}.jsonl
uv run python -m oc14_triage.data.card           # -> data/cards/DATA_CARD.md
uv run pytest                                     # tests
```

## État

Voir `IMPLEMENTATION_PLAN.md` § "État actuel". En bref : **données collectées + pipeline
SFT/DPO + évaluation + CI échafaudés et fonctionnels** ; entraînement (Kaggle), service
(RunPod vLLM) et rapport restent à faire. Identifiants requis listés dans le plan.
