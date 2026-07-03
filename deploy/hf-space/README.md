---
title: CHSA — Agent de triage médical (POC)
emoji: 🏥
colorFrom: red
colorTo: indigo
sdk: gradio
sdk_version: 6.19.0
python_version: "3.12"
app_file: app.py
pinned: false
short_description: Agent de triage médical (FR/EN) — Qwen3-1.7B fine-tuné, vLLM
---

# CHSA — Agent de triage médical (POC)

Démonstration publique de l'agent de triage médical du **Centre Hospitalier Saint-Aurélien (CHSA)**.

Cette Space fait tourner **deux briques dans un même processus** :

1. le **service FastAPI** de triage (`oc14_triage.agent.service`) sur `127.0.0.1:8091` ;
2. l'**UI Gradio** patient (`oc14_triage.agent.ui`) sur le port `7860`, qui appelle le
   service par HTTP — exactement comme la démo locale.

Le nœud `triage` de la chaîne LangGraph appelle le **modèle réel fine-tuné**
(`ghislaindelabie/oc14-qwen3-1.7b-triage-sft`, arm **sft-v9**) servi par un endpoint
**vLLM sur RunPod** (compatible OpenAI). Les données patient sont anonymisées (Presidio +
spaCy) avant tout traitement ; aucune donnée brute n'est journalisée ni persistée.

> Aide à la décision pour le personnel soignant — **ne remplace pas** un professionnel de santé.

## Note pour l'évaluateur

L'endpoint RunPod est **scale-to-zero** : la **toute première requête** peut prendre
**1 à 2 minutes** (démarrage à froid du GPU). Les requêtes suivantes répondent en quelques
secondes. Si le service met un instant à répondre, l'UI affiche un message d'attente ; relancez.

## Configuration (variables/secrets de la Space)

- `VLLM_BASE_URL` — endpoint OpenAI-compatible du modèle servi
- `VLLM_API_KEY` — **secret** (clé RunPod)
- `OC14_MODEL_ID`, `OC14_MODEL_VERSION`, `VLLM_TIMEOUT`
