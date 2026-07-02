# Livrables — POC Agent de triage médical (CHSA)

**Projet OpenClassrooms #14 — « Finetunez votre propre LLM »**
Auteur : **Ghislain Delabie** · Démarrage projet : **06 / 2026** (`mmaaaa = 062026`)
Dépôt (public) : <https://github.com/ghislaindelabie/oc14-medical-triage-llm> — **tag `v1`** sur `main` (`git checkout v1`) = l'état livré.

> **Convention de nommage OpenClassrooms**
> Dossier zip : `Triage_Medical_LLM_Delabie_Ghislain`
> Fichiers : `Delabie_Ghislain_<N>_<NomLivrable>_062026`

---

## Accès rapides

| Ressource | Lien / emplacement |
|---|---|
| **Rapport technique (PDF, 16 p ≤ 20)** | `Delabie_Ghislain_4_Rapport_062026.pdf` (présent dans ce dossier) |
| **Démo live — agent complet (URL stable)** | **HF Space** : <https://ghislaindelabie-oc14-triage-demo.hf.space/> — modèle **réel** (v9 via RunPod). 1ʳᵉ requête ~1-2 min si l'endpoint dort (scale-to-zero), puis ~2 s. |
| **Modèle servi (SFT v9)** | HF **public** : <https://huggingface.co/ghislaindelabie/oc14-qwen3-1.7b-triage-sft> |
| **Endpoint cloud (vLLM)** | RunPod serverless (OpenAI-compatible, **privé/clé**) — **testable via la démo HF Space** ci-dessus |
| **Code + CI** | dépôt GitHub public ci-dessus (tag `v1`) |

---

## Les 6 livrables

| N° | Livrable | Nom de fichier (zip) | Où / lien | Statut |
|----|----------|----------------------|-----------|--------|
| 1 | **Dataset médical bilingue** (format JSONL / Hugging Face `messages`, versionné) | `Delabie_Ghislain_1_Dataset_062026` | **dépôt public** `data/kaggle_upload/*.jsonl` (tag `v1`) · carte `data/cards/DATA_CARD.md` · audit `data/cards/anonymization_audit.json` | ✅ |
| 2 | **Modèle spécialisé SFT + LoRA** | `Delabie_Ghislain_2_Modele_SFT_LoRA_062026` | HF **public** `ghislaindelabie/oc14-qwen3-1.7b-triage-sft` (v9 fusionné 16-bit) · adapter LoRA · notebook Kaggle `oc14-sft-lora` | ✅ |
| 3 | **Alignement par préférences (DPO)** | `Delabie_Ghislain_3_DPO_062026` | paires `data/kaggle_upload/dpo_*.jsonl` · notebook DPO · analyse **§3.4** du rapport | ✅ (négatif instructif, non retenu) |
| 4 | **Rapport technique** (PDF ≤ 20 p) | `Delabie_Ghislain_4_Rapport_062026.pdf` | **présent dans ce dossier** (**16 pages**) ; source `docs/RAPPORT_FR.pdf` | ✅ |
| 5 | **Endpoint cloud vLLM** (inférence rapide) | `Delabie_Ghislain_5_Endpoint_vLLM_062026` | RunPod serverless (image `worker-v1-vllm`, privé) — **testable via la démo HF Space** ci-dessus · code : wrapper FastAPI `serving/` + `serving/Dockerfile` | ✅ déployé (~2 s à chaud) |
| 6 | **CI/CD GitHub Actions** | `Delabie_Ghislain_6_CICD_062026` | `.github/workflows/ci.yml` (ruff + pytest verts + job `deploy`) · exécutions Actions | ✅ |

---

## Notes de synthèse

- **Modèle servi = SFT v9** — macro-F1 **0,82** sur le *gold* stratifié n=300 (décodage déterministe, sans fuite). Le **DPO** est analysé comme un **négatif instructif** (effondrement de la classe médiane par déplacement de vraisemblance) et **non retenu** — cf. §3.4.
- **RGPD** — dataset **non-personnel par construction** (questions d'examen ECN/pharmacie + vignettes synthétiques → Recital 26) ; **anonymisation Presidio** au *runtime* ; **aucun texte brut persisté** (hash SHA-256 pour la traçabilité). Détail : `data/cards/DATA_CARD.md` + `data/cards/anonymization_audit.json`.
- **Endpoint privé** (clé requise). La démo utilise le **modèle réel** (RunPod *serverless* / scale-to-zero). Une requête *à chaud* répond en ~2 s ; si l'endpoint **dort** (timeout ~10 s), la chaîne renvoie un **message honnête sans verdict fabriqué** — *« le modèle démarre → réessayez dans ~1 min ; en attendant, ce cas relève de l'évaluation d'un clinicien »* (`needs_review`, urgence *indéterminée*) — la 1ʳᵉ requête ayant réveillé le GPU entre-temps. **Exception de sécurité** : un *red-flag* détecté escalade quand même en *maximale*. Un *stub* déterministe (règles statiques) sert la CI/le dev.
- **Reproductibilité** — seed **3407**, notebooks Kaggle versionnés, dépendances figées (`pyproject.toml` / `uv.lock`), 117 tests verts.

## Empaquetage

```
Triage_Medical_LLM_Delabie_Ghislain.zip
├── Delabie_Ghislain_4_Rapport_062026.pdf     # = docs/RAPPORT_FR.pdf
└── LIVRABLES.md                              # ce fichier (index + liens vers 1,2,3,5,6)
```
Les livrables 1, 2, 3, 5, 6 sont des artefacts **cloud / dépôt** (GitHub, Hugging Face, Kaggle, RunPod) : ils sont **référencés par lien** ci-dessus plutôt que copiés dans le zip.
