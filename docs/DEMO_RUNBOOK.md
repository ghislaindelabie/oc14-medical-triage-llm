# OC14 — Runbook de démonstration (POC agent de triage CHSA)

_Comment lancer, démontrer et « aller/ne pas aller » (go/no-go) le POC. Complète le rapport et la soutenance._

## 1. Architecture (= la chaîne du parcours patient)

```
Patient ─▶ UI Gradio ─HTTP▶ API FastAPI (/session/*)
                              └─ Agent LangGraph :
                                 collecte (questionnaire adaptatif)
                                 → anonymisation (Presidio)   ← frontière RGPD : raw effacé, seul input_sha256 conservé
                                 → prétraitement / validation (signes d'alerte)
                                 → triage (LLM fine-tuné)      ← stub | RunPod vLLM (SFT v9)
                                 → explication (niveau + justification + reco + disclaimer ; override sécurité)
                                 → persistance (SQLite « dossier », req-id, latence)
                                 → intégration SIH (enregistrement FHIR-shaped)
```

## 2. Lancer la démo (mode stub — aucune GPU requise)

Deux terminaux, depuis la racine du dépôt :
```bash
# Terminal 1 — API + agent
OC14_TRIAGE_STUB=1 uv run uvicorn oc14_triage.agent.service:app --port 8080

# Terminal 2 — interface patient
AGENT_SERVICE_URL=http://localhost:8080 uv run python -m oc14_triage.agent.ui
# → Gradio sur http://localhost:7860
```
Vérification avant de passer en direct : `curl -s localhost:8080/health` → `{"status":"ok"}`.

**Basculer sur le vrai modèle (RunPod vLLM, SFT v9) :** ne pas définir `OC14_TRIAGE_STUB` ; définir `VLLM_BASE_URL` (`https://api.runpod.ai/v2/<endpoint>/openai/v1`), `OC14_MODEL_ID`, `VLLM_API_KEY=$RUNPOD_API_KEY`. Envoyer **une requête de préchauffe** avant la séance (cold start serverless).

## 3. Cas de démonstration (présenter le 🔴 en premier)

| # | Saisies (motif → suivi → début → intensité) | Niveau attendu |
|---|---|---|
| 1 🔴 **maximale** | `douleur thoracique` → `oui, irradie dans le bras gauche, sueurs` → `il y a 30 min` → `9` | urgence maximale (override signe d'alerte + branche de suivi adaptatif) |
| 2 🟠 **modérée** | `mal de ventre depuis hier avec nausées` → `hier` → `6` | urgence modérée |
| 3 🟢 **différée** | `petit rhume léger, nez qui coule` → `2 jours` → `2` | urgence différée |
| 4 🔒 **RGPD** | `Je m'appelle Jean Dupont, tél 06 12 34 56 78, forte fièvre` → `ce matin` → `7` | montre `[NOM]`/`[TEL]` dans le panneau de traçabilité tout en triant |

Après le cas 1, ouvrir l'accordéon **Traçabilité** : montrer le `req-id`, le texte **anonymisé** stocké, et l'absence de nom/téléphone dans le dossier (la correction de la remarque d'évaluation précédente, en direct).

## 4. Check-list go / no-go (avant mise en service pilote)

| Contrôle | Seuil / attendu | Statut |
|---|---|---|
| `/health` répond | `{"status":"ok"}` | ✅ |
| Chaîne complète end-to-end | les 3 cas donnent le bon niveau | ✅ (stub) |
| Anonymisation RGPD | aucun identifiant direct dans le dossier persistté (test de fuite) | ✅ (test `test_graph`/`test_service`) |
| Traçabilité | `req-id` + `input_sha256` par interaction ; pas de texte brut | ✅ |
| Rappel *urgence maximale* | ≥ 0,83 (plancher IC, SFT v9) | ✅ (0,90 [0,83–0,95]) |
| Taux de disclaimer | = 1,00 | ✅ |
| Secrets protégés | clés en `.env`/secrets GitHub, jamais commit | ✅ |
| Endpoint cloud (vLLM) | requête de préchauffe OK + latence mesurée | ⏳ (RunPod, après clé) |
| Latence | p50/p95 mesurées en conditions réelles | ⏳ (mesure sur endpoint réel) |

## 5. Limites d'usage (à communiquer aux utilisateurs)

- **Aide à la décision, pas triage autonome** : l'agent assiste le personnel soignant ; la décision finale reste humaine (« human-in-the-loop »).
- **Ne pose pas de diagnostic, ne prescrit pas** ; escalade systématiquement les signes d'alerte ; termine toujours par un avertissement.
- **Standard « argent »** : étiquettes issues d'un consensus de 3 LLM (κ≈0,67), non d'un jury clinicien — à valider cliniquement avant tout usage réel.
- **Évaluation 100 % française** ; couverture EN de triage minime à l'entraînement (limite documentée).
- **Constantes vitales non collectées** dans le POC (pas d'intégration dispositifs) — prévu au passage à l'échelle.
- **POC** : intégration SIH simulée (enregistrement FHIR-shaped, pas de connexion hospitalière réelle).

## 6. Surveillance post-déploiement (prévu)

- Journal d'audit par interaction (`req-id`, `input_sha256`, niveau, latence) — déjà écrit par le nœud de persistance.
- À câbler en production : alerte sur taux d'erreur / latence p95, dérive du taux d'*urgence maximale* (sur-triage), et revue périodique d'un échantillon par un clinicien.
