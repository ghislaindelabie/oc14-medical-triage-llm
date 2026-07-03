# Rapport technique — POC d'agent de triage médical (CHSA)

> **Livrable 5 — OpenClassrooms « Finetunez votre propre LLM ».**
> Auteur : Ghislain Delabie · IA Engineer junior · mission Centre Hospitalier Saint-Aurélien (CHSA).
> Dépôt : `github.com/ghislaindelabie/oc14-medical-triage-llm` (public).
> Tous les chiffres cités sont mesurés (sources : `DEVELOPMENT_JOURNAL.md`, `data/cards/DATA_CARD.md`,
> les logs de runs Kaggle, les fichiers `data/processed/_*_stats.json`). Rien n'est estimé sauf mention
> explicite. Ce document tient en ≤ 20 pages.
>
> **Version 2 (2026-07-03).** Ajouts depuis la V1 : DPO **corrigé** (`rpo_alpha` → +0,018 sans effondrement),
> recherche d'hyperparamètres (sweep W&B), garde-fou d'entrée hors-distribution, et cadrage honnête de v10.

---

## 0. Résumé exécutif

Le CHSA fait face à une surcharge chronique de son service d'urgences : effectifs de triage tendus,
temps d'attente longs, et risque que des cas critiques ne soient pas identifiés assez vite. La Direction
Innovation Médicale (Dr. Marie Dubois) m'a confié la réalisation, en quatre semaines, d'un **POC d'agent
IA de triage médical** : un assistant qui collecte les symptômes du patient, évalue un **niveau de priorité**
(*urgence maximale / modérée / différée*), explique son évaluation, s'intègre au système d'information
hospitalier (SIH) et garantit la **traçabilité** de chaque interaction.

**Ce que je livre** (les cinq livrables du brief) :

1. un **jeu de données médical bilingue** anonymisé et versionné, prêt pour le SFT (~5 600 paires) et le DPO ;
2. un **modèle spécialisé** — Qwen3-1.7B-Base affiné par SFT+LoRA, puis évalué face à une tentative de DPO ;
3. un **endpoint de démonstration** : agent complet exposé par API FastAPI, avec un plan de serving cloud vLLM/RunPod ;
4. un **pipeline CI/CD** GitHub Actions (lint + tests à chaque push, job de déploiement) ;
5. ce **rapport technique**.

**Résultat clé.** Sur un jeu de test *gold* stratifié (n=300, décodage déterministe, sans fuite), le
fine-tuning fait passer le modèle d'**inutilisable à compétent** : **macro-F1 0,19 (Base non entraîné) →
0,82 (SFT servi)**. Le rappel sur la classe critique *urgence maximale* atteint **0,90 [IC 95 % : 0,83–0,95]**,
et le modèle produit toujours le format attendu et l'avertissement de sécurité (taux 1,00 contre 0,68 / 0,00
pour le Base).

**Positionnement (essentiel).** Ce POC démontre une **méthode** et un **signal de progrès**, **pas** un
trieur autonome déployable. Un rappel de 0,90 sur les urgences vitales (plancher d'intervalle de confiance
à 0,83) signifie qu'au moins 1 urgence sur 10 pourrait être manquée dans le pire cas — ce qui est
**inacceptable pour un triage autonome**. Je positionne donc l'agent comme un outil d'**aide à la décision
sous supervision humaine** (« human-in-the-loop ») : il assiste le personnel soignant, la décision finale
reste humaine, il ne pose pas de diagnostic et ne prescrit pas.

---

## 1. Contexte & cadrage

### 1.1 La mission

Le brief demande un **agent** qui accompagne le patient en (1) collectant ses symptômes via un
questionnaire intelligent adaptatif, (2) évaluant le niveau de priorité selon les protocoles médicaux,
(3) fournissant des explications claires, (4) s'intégrant au SIH existant, et (5) garantissant la
traçabilité pour les audits médicaux. La stratégie du CHSA est en trois phases : validation conceptuelle
sur Qwen3-1.7B-Base (phase 1), optimisation ciblée par SFT+LoRA puis DPO (phase 2), projection industrielle
vers des modèles 32B+ en cas de succès (phase 3).

### 1.2 La décision de périmètre : un agent complet, pas seulement un LLM affiné

Le brief demande un **agent** de triage (questionnaire adaptatif, intégration SIH, traçabilité), pas
seulement un LLM affiné. J'ai donc traité le fine-tuning du LLM comme le **cœur clinique**, puis — en
**dernière étape** — construit autour de lui la **chaîne fonctionnelle complète** du parcours patient :
collecte → anonymisation → prétraitement/validation → triage (le LLM) → explication → persistance →
intégration SIH. Chaque composant nommé dans le mail du Dr. Dubois est présent et démontrable en direct
(détails d'implémentation en §4).

### 1.3 Carte de conformité (livrable → où c'est démontré)

| Livrable / exigence du brief | Où c'est démontré dans ce projet | Statut |
|---|---|---|
| **L1 — Dataset bilingue anonymisé, versionné, SFT + DPO** | `data/kaggle_upload/*.jsonl` ; `DATA_CARD.md` ; passe Presidio (audit) ; §2 | ✓ |
| L1 — Schéma des métadonnées (symptômes / antécédents / constantes / source / confiance) | `METADATA_SCHEMA.md` ; §2.4 | ✓ |
| L1 — Justification RGPD | `DATA_CARD.md` (Recital 26) ; passe Presidio ; §2.5 | ✓ |
| L1 — Séparation train / éval (auditabilité, pas de fuite) | éval-gold disjointe, audit adverse ; §2, §5 | ✓ |
| **L2 — Modèle Qwen3-1.7B affiné SFT+LoRA puis aligné DPO, poids + métriques** | notebooks Kaggle ; adapter LoRA ; SFT v9 servi ; DPO corrigé (`rpo_alpha`) ; §3, §5 | ✓ (DPO : effondrement diagnostiqué → corrigé, +0,018 sans effondrement ; v9 servi en V1) |
| L2 — Traçabilité entraînement (logs, seed, checkpoints) | seed 3407, `adapter_config.json`, tracking W&B ; §3, §6 | ✓ |
| **L3 — Endpoint cloud vLLM, API de démo, inférence rapide** | wrapper FastAPI `/triage` conteneurisé ; agent FastAPI ; **endpoint vLLM déployé sur RunPod** (serverless, v9) ; §4, §6 | ✓ déployé (~2 s à chaud) |
| **L4 — CI/CD GitHub Actions (tests + déploiement)** | `.github/workflows/ci.yml` (ruff + pytest verts + job `deploy`) ; §6 | ✓ |
| **L5 — Rapport ≤ 20 p (méthodo données + entraînement · métriques · analyse · roadmap)** | ce document | ✓ |
| Agent — questionnaire adaptatif | `agent/questionnaire.py` ; §4 | ✓ |
| Agent — évaluation de priorité selon protocoles | rubrique ESI/MTS ; LLM affiné ; §2.2, §4 | ✓ |
| Agent — explications claires | nœud `explication` ; §4 | ✓ |
| Agent — intégration SIH | mock FHIR R4 ; §4 | ✓ (simulé) |
| Agent — traçabilité pour audits | dossier SQLite, `req-id`, `input_sha256` ; §4 | ✓ |

Le reste du rapport détaille chaque ligne, avec l'accent (selon la demande) sur les décisions **Données/RGPD**
(§2) et **Entraînement/DPO** (§3), présentées en **Problème → Options → Décision → Critère → Preuve**.

---

## 1.5 Architecture & pipelines (vue d'ensemble)

**Vue clinique — le parcours de dépistage du patient** (ce que voit le personnel soignant ; l'agent *assiste*, le soignant *décide*).

<svg width="1120" height="185" viewBox="0 0 1120 185" style="max-width:100%;height:auto;font-family:sans-serif" xmlns="http://www.w3.org/2000/svg"> <defs> <marker id="arP" markerWidth="9" markerHeight="9" refX="6" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill="#475569"/></marker> <marker id="arRed" markerWidth="9" markerHeight="9" refX="6" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill="#dc2626"/></marker> </defs><g fill="none" stroke-width="1.5"> <rect x="15" y="98" width="164" height="52" rx="8" fill="#f0fdfa" stroke="#0d9488"/> <rect x="197" y="98" width="164" height="52" rx="8" fill="#f0fdfa" stroke="#0d9488"/> <rect x="379" y="98" width="164" height="52" rx="8" fill="#f0fdfa" stroke="#0d9488"/> <rect x="561" y="98" width="164" height="52" rx="8" fill="#f0fdfa" stroke="#0d9488"/> <rect x="743" y="98" width="164" height="52" rx="8" fill="#fffbeb" stroke="#d97706" stroke-width="2"/> <rect x="925" y="98" width="164" height="52" rx="8" fill="#f0fdfa" stroke="#0d9488"/> </g> <rect x="330" y="10" width="440" height="40" rx="8" fill="#fef2f2" stroke="#dc2626" stroke-width="2"/> <text x="550" y="27" text-anchor="middle" font-size="12" font-weight="bold" fill="#b91c1c">Signe d'alerte (red-flag) détecté</text> <text x="550" y="42" text-anchor="middle" font-size="10" fill="#b91c1c">court-circuit → urgence maximale immédiate</text> <g stroke="#dc2626" stroke-width="1.6" fill="none"> <path d="M287,96 L330,45" marker-end="url(#arRed)"/> <path d="M770,32 L845,96" marker-end="url(#arRed)"/> </g> <g stroke="#475569" stroke-width="1.6" marker-end="url(#arP)"> <line x1="197" y1="125" x2="212" y2="125"/><line x1="379" y1="125" x2="394" y2="125"/> <line x1="561" y1="125" x2="576" y2="125"/><line x1="743" y1="125" x2="758" y2="125"/> <line x1="925" y1="125" x2="940" y2="125"/> </g> <g font-size="12" font-weight="bold" text-anchor="middle" fill="#0f766e"> <text x="97" y="120">Patient</text><text x="279" y="120">Collecte</text><text x="461" y="120">Évaluation IA</text> <text x="643" y="120">Priorité</text><text x="825" y="120" fill="#b45309">Revue soignant</text><text x="1007" y="120">Dossier SIH</text> </g> <g font-size="9.5" text-anchor="middle" fill="#475569"> <text x="97" y="138">motif</text><text x="279" y="138">questionnaire</text><text x="461" y="138">niveau d'urgence</text> <text x="643" y="138">+ explication</text><text x="825" y="138" fill="#b45309">HITL — décide</text><text x="1007" y="138">tracé / audit</text> </g>  </svg>

**Pipeline d'entraînement — des sources au modèle servi.**

<svg width="1120" height="118" viewBox="0 0 1120 118" style="max-width:100%;height:auto;font-family:sans-serif" xmlns="http://www.w3.org/2000/svg"> <defs><marker id="arT" markerWidth="9" markerHeight="9" refX="6" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill="#475569"/></marker></defs><g fill="none" stroke-width="1.5"> <rect x="12" y="25" width="124" height="60" rx="8" fill="#eff6ff" stroke="#2563eb"/> <rect x="150" y="25" width="124" height="60" rx="8" fill="#eff6ff" stroke="#2563eb"/> <rect x="288" y="25" width="124" height="60" rx="8" fill="#eff6ff" stroke="#2563eb"/> <rect x="426" y="25" width="124" height="60" rx="8" fill="#eff6ff" stroke="#2563eb"/> <rect x="564" y="25" width="124" height="60" rx="8" fill="#eff6ff" stroke="#2563eb"/> <rect x="702" y="25" width="124" height="60" rx="8" fill="#f3f4f6" stroke="#9ca3af" stroke-dasharray="5 3"/> <rect x="840" y="25" width="124" height="60" rx="8" fill="#eff6ff" stroke="#2563eb"/> <rect x="978" y="25" width="124" height="60" rx="8" fill="#dbeafe" stroke="#1d4ed8" stroke-width="2"/> </g> <g stroke="#475569" stroke-width="1.6" marker-end="url(#arT)"> <line x1="136" y1="55" x2="149" y2="55"/><line x1="274" y1="55" x2="287" y2="55"/> <line x1="412" y1="55" x2="425" y2="55"/><line x1="550" y1="55" x2="563" y2="55"/> <line x1="688" y1="55" x2="701" y2="55"/><line x1="826" y1="55" x2="839" y2="55"/> <line x1="964" y1="55" x2="977" y2="55"/> </g> <g font-size="12.5" font-weight="bold" text-anchor="middle" fill="#1e3a8a"> <text x="74" y="46">Sources</text><text x="212" y="46">Étiquetage</text><text x="350" y="46">Dataset SFT</text> <text x="488" y="46">SFT + LoRA</text><text x="626" y="46">Fusion</text><text x="764" y="46" fill="#6b7280">DPO</text> <text x="902" y="46">HF Hub</text><text x="1040" y="46">RunPod</text> </g> <g font-size="9.5" text-anchor="middle" fill="#475569"> <text x="74" y="72">MediQAl·MedQuAD</text><text x="212" y="72">consensus 3-LLM</text><text x="350" y="72">bilingue ~5 600</text> <text x="488" y="72">Unsloth · T4</text><text x="626" y="72">16-bit</text><text x="764" y="72" fill="#9ca3af">non retenu</text> <text x="902" y="72">privé (v9)</text><text x="1040" y="72">endpoint vLLM</text> </g>  </svg>

**Pipeline agentique — le parcours patient au *runtime*** (le nœud d'anonymisation est la **frontière RGPD** : au-delà, le **texte brut est effacé** — seul son hash subsiste pour la traçabilité — et seul le **texte anonymisé** circule : contenu clinique lisible, PII masquée).

<svg width="1120" height="118" viewBox="0 0 1120 118" style="max-width:100%;height:auto;font-family:sans-serif" xmlns="http://www.w3.org/2000/svg"> <defs><marker id="arR" markerWidth="9" markerHeight="9" refX="6" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill="#475569"/></marker></defs><g fill="none" stroke-width="1.5"> <rect x="12" y="25" width="124" height="60" rx="8" fill="#f0fdfa" stroke="#0d9488"/> <rect x="150" y="25" width="124" height="60" rx="8" fill="#f0fdfa" stroke="#0d9488"/> <rect x="288" y="25" width="124" height="60" rx="8" fill="#fef2f2" stroke="#dc2626" stroke-width="2.2"/> <rect x="426" y="25" width="124" height="60" rx="8" fill="#f0fdfa" stroke="#0d9488"/> <rect x="564" y="25" width="124" height="60" rx="8" fill="#f0fdfa" stroke="#0d9488"/> <rect x="702" y="25" width="124" height="60" rx="8" fill="#fffbeb" stroke="#d97706" stroke-width="2"/> <rect x="840" y="25" width="124" height="60" rx="8" fill="#f0fdfa" stroke="#0d9488"/> <rect x="978" y="25" width="124" height="60" rx="8" fill="#f0fdfa" stroke="#0d9488"/> </g> <g stroke="#475569" stroke-width="1.6" marker-end="url(#arR)"> <line x1="136" y1="55" x2="149" y2="55"/><line x1="274" y1="55" x2="287" y2="55"/> <line x1="412" y1="55" x2="425" y2="55"/><line x1="550" y1="55" x2="563" y2="55"/> <line x1="688" y1="55" x2="701" y2="55"/><line x1="826" y1="55" x2="839" y2="55"/> <line x1="964" y1="55" x2="977" y2="55"/> </g> <g font-size="12.5" font-weight="bold" text-anchor="middle" fill="#0f766e"> <text x="74" y="46">Patient</text><text x="212" y="46">Questionnaire</text><text x="350" y="46" fill="#b91c1c">Anonymisation</text> <text x="488" y="46">Prétraitement</text><text x="626" y="46">LLM triage</text><text x="764" y="46" fill="#b45309">Explication</text> <text x="902" y="46">Persistance</text><text x="1040" y="46">SIH</text> </g> <g font-size="9.5" text-anchor="middle" fill="#475569"> <text x="74" y="72">motif libre</text><text x="212" y="72">règles + red-flag</text><text x="350" y="72" fill="#b91c1c">Presidio · RGPD</text> <text x="488" y="72">red-flags</text><text x="626" y="72">vLLM (RunPod)</text><text x="764" y="72" fill="#b45309">override + confiance</text> <text x="902" y="72">SQLite · 0 PII</text><text x="1040" y="72">FHIR R4 (mock)</text> </g>  </svg>

---

## 2. Données & RGPD (méthodologie décision-driven)

### 2.1 Sources & licences

Toutes les sources sont **publiques** ; **aucune** n'est un dossier patient réel. La sélection a été guidée
par un critère de sûreté juridique (licence redistribuable) et de pertinence à la tâche.

| Source | Langue | Licence | Rôle | Statut |
|---|---|---|---|---|
| ANR-MALADES/**MediQAl** | FR | CC-BY-4.0 | QA médicale FR + (cas cliniques →) triage | active |
| keivalya/**MedQuAD** | EN | CC-BY-4.0 (Ben Abacha 2019) | QA médicale EN | active |
| TsinghuaC3I/**UltraMedical-Preference** | EN | MIT | paires DPO | *collecté pour un essai DPO initial, **non retenu** (hors-tâche)* |
| qanastek/**FrenchMedMCQA** | FR | Apache-2.0 | QA MCQ pharmacie FR | **désactivée** (loader incompatible `datasets≥3`) |
| **MIETIC / MIMIC** (triage) | EN | PhysioNet *credentialed* | triage EN réel | **exclue** |

**Problème (l'exclusion MIETIC/MIMIC).** Le seul corpus de triage « réel » facilement accessible (MIETIC)
dérive de MIMIC, sous licence PhysioNet **accréditée à non-redistribution**.

**Options.** (a) l'utiliser et redistribuer discrètement le dérivé ; (b) l'exclure et fabriquer un signal
de triage à partir de sources redistribuables.

**Décision.** **Exclure** MIETIC/MIMIC.

**Critère.** Protéger l'histoire « aucune donnée patient réelle / conforme RGPD » et respecter la licence —
un POC hospitalier ne peut pas s'affranchir d'une **DUA** (*Data Use Agreement*, accord d'utilisation des
données ; celle de PhysioNet interdit la redistribution).

**Preuve.** Exclusion documentée (`DATA_CARD.md`, recherche `00` §0b). Le signal de triage vient à la place
de **deux sources à provenance propre** : principalement les **cas cliniques réels de MediQAl** (questions
d'examen ECN) **étiquetés par le consensus de 3 LLM** (§2.2, standard « argent »), complétés par un petit
jeu de **vignettes fictives rédigées à la main** (bilingue — signal sécurité + triage EN).

### 2.2 Étiqueter le triage sans clinicien : le consensus de 3 LLM

**Problème.** MediQAl est du QA d'examen (QCM), **pas** du triage ; il n'existe **aucun** *gold* de triage
français validé, et **aucun clinicien** n'était disponible pour annoter. La première version utilisait une
heuristique de mots-clés (signaux d'alerte) — trop grossière, non défendable.

**Options.** (a) heuristique de mots-clés ; (b) générer des cas synthétiques ; (c) annoter des cas **réels**
par un ou plusieurs LLM de pointe guidés par une rubrique citée.

**Décision.** Option (c) : labelliser les **3 075 vignettes cliniques réelles** de MediQAl par un **consensus
de 3 LLM** — OpenAI `gpt-5.4`, Mistral `medium-3.5`, Anthropic `claude-sonnet-4-6` — chacun renvoyant, dans un
même appel, un **niveau 3-classes + son équivalent ESI 1-5**, contre une rubrique citée (`TRIAGE_CRITERIA.md`).
Je conserve les cas **unanimes et ESI-cohérents** comme *gold* (évaluation), et les cas majoritaires pour
l'entraînement.

**Critère.** Fiabilité maximale sans clinicien : un consensus multi-modèles réduit le biais d'un modèle
unique, la rubrique ancre les décisions dans les protocoles (ESI, MTS, échelles FR : FRENCH/CIMU), et l'accord
inter-annotateurs se **mesure** (Fleiss κ).

**Preuve.** Sur les 3 075 cas × 3 modèles : **Fleiss κ ≈ 0,67** (accord « substantiel »), **1 603 cas
unanimes + ESI-cohérents** retenus comme *gold*. Coût total maîtrisé (36,67 $ grâce au *prompt caching* :
73 % des *tokens* servis depuis le cache côté OpenAI, 91 % côté Anthropic). C'est **explicitement un standard
« argent »** (LLM-annotateur), pas une validation clinique — assumé (voir §7).

La rubrique collapse les 5 niveaux ESI vers 3 niveaux, une fois, de façon transparente et monotone :

| Niveau | ESI | Sens | Exemple de signal d'alerte (⇒ ce niveau) |
|---|---|---|---|
| **urgence maximale** | 1–2 | menace vitale/fonctionnelle, ne peut pas attendre | douleur thoracique (SCA), détresse respiratoire, AVC (FAST), sepsis/choc, GCS<13… |
| **urgence modérée** | 3 | symptomatique, à évaluer promptement, stable | infection fébrile focale, douleur modérée, fracture suspectée mais appui possible… |
| **urgence différée** | 4–5 | non urgent / ambulatoire / administratif | rhume léger, renouvellement d'ordonnance, certificat, conseil de prévention… |

Règle décisive : **en cas de doute, sur-trier** (choisir le niveau le plus urgent) — le sous-triage est
l'erreur dangereuse.

### 2.3 Composition du jeu final (livré, vérifiable)

Le jeu final n'est **pas** un corpus purement synthétique : c'est un **mélange de trois briques**, dont seule
l'*étiquette de triage* est générée.

1. **Q&A médical (3 557 lignes) — paires réelles des sources, telles quelles.** `mediqal_mcqu` (2 166),
   `mediqal_oeq` (396) et `medquad` (995) : leurs paires question→réponse **existantes**, reprises pour donner
   au modèle les **connaissances médicales** et le **bilinguisme**. Ces sources **ne portent aucune étiquette
   de triage** (ton intuition est juste) — on ne s'en sert donc que pour le Q&A.
2. **Triage (1 953 lignes `llm_triage`) — cas réels + étiquette générée.** Faute d'étiquettes de triage dans
   les sources, j'ai pris les **vrais cas cliniques de MediQAl** (champ `clinical_case`, vignettes d'examen)
   et **généré le niveau + la justification par le consensus de 3 LLM** (§2.2). → les **cas sont réels**,
   seules les **étiquettes** sont générées (standard « argent »). C'est *ça*, la partie « synthétique » — au
   niveau des labels, pas des cas.
3. **Vignettes rédigées à la main (88 lignes = ~11 × 8) — fictives, à provenance propre.** Un petit jeu que
   j'ai écrit à la main (signal sécurité + triage EN).

L'**éval gold (300)** provient aussi du consensus 3-LLM (les cas les plus fiables), **mise de côté** et
disjointe du train. Le DPO est un jeu de préférences de triage **équilibré par direction d'erreur**.

| Jeu | Train / Val | Détail |
|---|---|---|
| **SFT** | **5 598 / 562** | Langue : fr 4 571 / en 1 027 (~82 % FR). Type : triage 2 041 / QA 3 557. |
| ↳ sources SFT | | `mediqal_mcqu` 2 166 · `llm_triage` 1 953 · `medquad` 995 · `mediqal_oeq` 396 · `vignette` 88 |
| ↳ urgence (rows triage) | | maximale 1 098 · modérée 689 · différée 254 |
| **DPO** | **211 / 24** | équilibré par direction : `dpo_under` 103 · `dpo_mod` 50 · `dpo_over` 48 · `safety` 10 |
| **Éval gold** | **300** | stratifié 100 / 100 / 100 (maximale / modérée / différée) |

**Composition du jeu SFT (train, 5 598 lignes) — en un coup d'œil.**

<svg width="1120" height="210" viewBox="0 0 1120 210" style="max-width:100%;height:auto;font-family:sans-serif" xmlns="http://www.w3.org/2000/svg"> <text x="10" y="24" font-size="12.5" font-weight="bold" fill="#1f2937">Par type</text> <g> <rect x="10" y="34" width="712" height="40" fill="#eff6ff" stroke="#2563eb"/> <rect x="10" y="34" width="215" height="40" fill="#2563eb"/> <rect x="722" y="34" width="19" height="40" fill="#d97706"/> <text x="117" y="59" text-anchor="middle" font-size="11" font-weight="bold" fill="#fff">QA médical 3 557</text> <text x="473" y="59" text-anchor="middle" font-size="11" font-weight="bold" fill="#1e3a8a">Triage (cas réels + labels 3-LLM) 1 953</text> <text x="820" y="59" font-size="10" fill="#b45309">vignettes 88 (rédigées)</text> </g> <text x="10" y="112" font-size="12.5" font-weight="bold" fill="#1f2937">Urgence (lignes triage, 2 041)</text> <g> <rect x="10" y="122" width="538" height="40" fill="#fee2e2" stroke="#dc2626"/> <rect x="548" y="122" width="338" height="40" fill="#fef3c7" stroke="#d97706"/> <rect x="886" y="122" width="124" height="40" fill="#dcfce7" stroke="#16a34a"/> <text x="279" y="147" text-anchor="middle" font-size="11" font-weight="bold" fill="#991b1b">maximale 1 098</text> <text x="717" y="147" text-anchor="middle" font-size="11" font-weight="bold" fill="#92400e">modérée 689</text> <text x="948" y="147" text-anchor="middle" font-size="10.5" font-weight="bold" fill="#166534">différée 254</text> </g> <text x="10" y="192" font-size="10" fill="#6b7280">Seule l'étiquette de triage est générée (consensus 3-LLM) ; les cas et les paires QA sont réels. Le sur-triage du corpus d'examen (~47 % maximale) motive le rééquilibrage (§3.3).</text> </svg>

### 2.4 Schéma des métadonnées

Le brief demande explicitement un schéma couvrant **symptômes, antécédents, constantes, source, niveau de
confiance** (`METADATA_SCHEMA.md`). Je le documente pour les deux niveaux : les enregistrements de données
d'entraînement et le **dossier d'exécution** écrit par l'agent (même forme que l'enregistrement SIH).

| Champ (brief) | Clé | Provenance dans le POC |
|---|---|---|
| Symptômes | `symptoms` | Runtime : `motif` + suivi ; dataset : le texte de la vignette |
| Antécédents | `medical_history` | Runtime : texte libre optionnel ; absent du corpus d'examen (limite documentée) |
| **Constantes** | `vitals` | **Runtime : saisie optionnelle** dans le questionnaire (T°, TA, pouls, SpO₂), injectée dans le prompt et persistée (anonymisée) ; absente du corpus d'examen (dataset). Intégration **dispositif** en roadmap (§8) |
| Source | `source` | dataset : `mediqal_mcqu`/`_oeq`/`medquad`/`llm_triage`/`vignette` ; runtime : `chsa-triage-poc` |
| Niveau de confiance | `confidence_level` | **dataset** : consensus 3-LLM (unanime 3/3 → `high`, majorité 2/3 → `medium`). **Runtime** : score **dérivé** (accord règle↔modèle + succès du *parsing* → `high`/`medium`/`low` ; `low` → revue clinicien). Version calibrée (*logprobs*) en roadmap (§8) |

### 2.5 Anonymisation RGPD : la passe corpus + la frontière runtime

**Problème.** Le brief exige une conformité RGPD **prouvée** (Presidio recommandé), y compris un contrôle
qu'« aucune donnée personnelle identifiable ne subsiste ». Mais mes sources sont des questions d'examen et des
vignettes synthétiques : le risque théorique de PII est faible — reste à le **démontrer**.

**Décision (deux niveaux).**
- **Passe corpus (offline).** J'ai fait tourner Microsoft **Presidio + spaCy** sur **6 695 textes** du corpus,
  avec un journal d'audit par enregistrement (SHA-256 + entités, **sans stocker le texte brut**).
- **Anonymisation au *runtime* dans l'agent.** Le nœud `anonymisation` est la **frontière RGPD** : tout input
  patient est anonymisé *avant* tout stockage ; à partir de ce nœud le `raw_text` est effacé, et l'input brut
  ne survit que sous forme de hash **`input_sha256`** — « **hash pour la traçabilité, anonymisation pour le
  stockage** ».

**Critère.** (1) Garantir l'invariant « aucun identifiant direct dans le dossier persisté », indépendamment
de la qualité du modèle NER ; (2) interpréter les résultats de la passe **sans surinterpréter**.

**Preuve — les détections brutes** (audit `anonymization_audit.json`) :

| Entité | Détections |
|---|---|
| LOCATION | 13 298 |
| AGE | 3 624 |
| PERSON | 3 330 |
| PHONE_NUMBER | 470 |
| DATE_TIME | 97 |

**Interprétation (le point à clarifier).** Ces détections sont **massivement des
faux positifs cliniques**, pas de vrais patients : les LOCATION sont surtout des **localisations anatomiques**,
les PERSON des **éponymes cliniques** (maladies/signes nommés) et des **personas d'examen génériques**
(« Monsieur X », « Madame Y »), les AGE des âges d'énoncé. C'est cohérent avec la nature des sources (questions
ECN/pharmacie + vignettes synthétiques) : **il n'y a pas de vrai patient**, donc les données sont non
personnelles par construction → hors champ RGPD (**Recital 26**). La passe Presidio ne « nettoie » donc pas
des fuites : elle **confirme** l'hypothèse d'absence de PII réelle et documente le processus.

L'architecture de l'anonymiseur (`src/oc14_triage/anonymization.py`) est **à deux couches**, ce qui rend
l'invariant robuste. **(a) Presidio+spaCy** pour les entités sémantiques (PERSON, LOCATION) — deux rôles
distincts : **spaCy** fournit le modèle de **reconnaissance d'entités nommées (NER)** qui *repère* dans la
phrase ce qu'aucune regex ne peut cadrer (un nom, un lieu n'ont pas de forme fixe) ; **Presidio** (Microsoft)
est le **cadre d'orchestration** qui combine la sortie de spaCy avec ses propres reconnaisseurs à motifs,
attribue un **score de confiance** à chaque détection et pilote le remplacement. **(b)** un **filet
regex inconditionnel** pour les identifiants structurés (téléphone, e-mail, NIR français, IDs numériques
longs). Ce filet existe parce que le reconnaisseur de téléphone spaCy FR est peu fiable (score ~0,4) — la
regex **garantit** le masquage quel que soit le modèle. Une *allowlist* de noms communs français (motif,
douleur, urgence…) évite de corrompre l'entrée par de faux PERSON/LOCATION.

---

## 3. Entraînement (méthodologie décision-driven)

### 3.1 Modèle de départ : Base (primaire) + Instruct (comparaison)

**Problème.** Le brief nomme Qwen3-1.7B-**Base**. Or un modèle *Instruct* possède déjà un comportement de
sécurité/refus, précieux pour un POC clinique.

**Décision.** Affiner **les deux** sur les mêmes données ; **Base est le modèle primaire** (celui que je
DPO, fusionne et sers), **Instruct est le bras de comparaison**.

**Critère.** Honorer le brief tout en montrant le compromis Base-vs-Instruct.

**Preuve.** J'ai affiné **les deux**, mais **seul Base a été porté jusqu'au bout** (DPO, fusion, serving) ;
l'Instruct fut une **sonde exploratoire de départ**, pas un bras des phases finales (il n'apparaît pas dans le
tableau de bord final à 5 bras, §7). Le chiffre de référence n'est donc **pas** un Base-vs-Instruct, mais le
**gain d'entraînement**, mesuré sur le **même** jeu *gold* held-out de **n=300** (stratifié, sans fuite — cf.
§4) : **Base non entraîné macro-F1 0,19 → SFT v9 0,82** — même métrique, même jeu, seul le modèle change. Le
0,19 est le **vrai plancher** d'un Base *non affiné* (il ne sait pas suivre le format de triage), pas une
version volontairement faible. *(Une comparaison exploratoire initiale sur n=6, entachée de confusion de
gabarit, subsiste dans le journal ; elle est dépassée.)*

### 3.2 SFT + LoRA via Unsloth

- **SFT** (Supervised Fine-Tuning) : montrer au modèle de bons exemples (instruction → réponse) pour qu'il
  apprenne notre format de triage et sa persona. **LoRA** (Low-Rank Adaptation) n'entraîne qu'un petit greffon
  (~0,3 % des poids) — ce qui tient sur un **T4 Kaggle gratuit** de 16 GB.
- **Config.** r=16, α=16, 4-bit, 2 époques, **seed 3407**, gabarit **ChatML**, `train_on_responses_only`
  (on n'entraîne que sur la réponse, pas sur le prompt). **Unsloth** (kernels GPU custom, ~2× plus rapide)
  au-dessus de TRL/PEFT (fallback documenté).
- **Détail d'inférence appris à la dure (Problème #14/#15).** Qwen3-1.7B-Base **ne fournit aucun chat template**
  → j'ai posé un ChatML explicite. Et le modèle doit **s'arrêter sur `<|im_end|>`** (son `eos` est
  `<|endoftext|>`) et utiliser le **prompt système complet**, sinon il dégénère (répétitions). C'est aussi
  exactement ainsi que je le sers.
- **Résultat.** train_loss ~0,869 sur le jeu labellisé par consensus, ~79 min sur T4 (≈ **0 €** ; ~1,3 des
  ~30 GPU-heures/semaine gratuites).

### 3.3 Gestion du déséquilibre de classes

**Problème.** Le corpus d'examen sur-représente le grave (~47 % *maximale*). Un premier SFT « propre »
(après audit) obtenait macro-F1 0,65 avec un **effondrement de *différée*** (rappel 0,28) : 72 cas sur 100
poussés à *modérée*. Cause : l'exclusion (correcte) des cas non-consensuels avait **affamé** la classe
*différée* (les cas ambigus sont souvent des cas bas).

**Décision.** (a) Relâcher le filtre de consensus à `n_agree ≥ 2` (garder les majorités 2/3 légitimes, ne
retirer que les 347 vrais désaccords) ; (b) sur-échantillonner ×8 les 11 vignettes rédigées à la main
(porteuses de *différée* équilibrée et de triage EN).

**Critère.** Restaurer le signal *différée* sans réintroduire de fuite, mesuré sur le même *gold* greedy.

**Preuve.** *différée* rappel **0,28 → 0,71**, macro-F1 **0,65 → 0,82** (v9). C'est le modèle servi.

### 3.4 DPO : un négatif instructif, diagnostiqué… puis corrigé

- **DPO** (Direct Preference Optimization) : montrer au modèle des paires (meilleure, moins bonne) réponses
  pour qu'il préfère la meilleure — alignement sans modèle de récompense séparé.
- **Invariant d'ordonnancement.** Le DPO tourne sur le modèle SFT **adapter LoRA encore attaché** ; la fusion
  en poids 16-bit se fait **une seule fois, après DPO** — jamais entre les étapes.

**Tentative n°1 — l'effondrement.** Premières paires de triage « équilibrées par direction » (211/24 :
sous-triage 103 + safety 10 / sur-triage 48 / modérée 50), chosen = le bon niveau, rejected = un niveau
**adjacent** erroné. Résultat : macro-F1 **0,822 → 0,799**, extrêmes aiguisés (*différée* 0,71 → 0,96) mais
**milieu effondré** (*modérée* 0,85 → **0,55**).

**Diagnostic (le mécanisme, littérature à l'appui).** *modérée* est le niveau **rejected** pour **les deux**
types de paires (sous-triage depuis *maximale*, sur-triage depuis *différée*) : ~168× « la mauvaise réponse »
contre 56× « la bonne » → le DPO apprend à **éviter le milieu**. C'est un **déplacement de vraisemblance**
(*likelihood displacement*, Razin et al. [2410.08847]) : la perte DPO ne récompense que **l'écart**
log P(chosen) − log P(rejected) ; l'optimiseur peut l'agrandir en **écrasant P(rejected) au point de faire
aussi baisser P(chosen)** — la marge grandit, mais la probabilité *absolue* de la bonne réponse chute.

**Correction — de meilleures paires + une régularisation anti-effondrement :**
1. **Distribution rejetée équilibrée** — chaque niveau apparaît *choisi* ET *rejeté* en proportions saines,
   pour ne jamais pénaliser structurellement le milieu (invariant vérifié en test : *modérée* = 53× choisie /
   151× rejetée, plus jamais absente du côté « choisi »).
2. **Hard negatives** — des paires tirées des **vraies erreurs du SFT** (les cas qu'il rate), pas des
   inversions de niveau synthétiques.
3. **Régularisation anti-effondrement** — **`rpo_alpha` de TRL** (`DPOTrainer(rpo_alpha=1.0)`) : ajoute à la
   perte DPO un terme **NLL/SFT sur la réponse *choisie*** (« continuer le SFT sur la bonne réponse pendant le
   DPO »), qui **ancre P(chosen) vers le haut** et bloque l'effondrement. Équivalent : **DPO-Positive / Smaug**
   [2402.13228] (pénalité dès que P(chosen) passe sous celle du modèle de référence).

**Preuve — comparaison *pommes-à-pommes*** (mêmes 300 *gold*, même harness, mêmes poids de base, adapter DPO
activé/désactivé — seule variable : le DPO) :

| n=300, greedy, même harness | macro-F1 | rappel maximale | rappel modérée | rappel différée |
|---|--:|--:|--:|--:|
| **SFT v9** | 0,827 | 0,84 | 0,88 | 0,75 |
| **SFT v9 + DPO (`rpo_alpha`)** | **0,845** | **0,88** | 0,83 | **0,82** |

*(SFT v9 mesuré à 0,827 sur ce harness dédié — cohérent avec le 0,822 canonique de §5.2 à la marge de bruit ;
les deux modèles sont notés avec le **même** harness pour une comparaison valide.)*

**Plus d'effondrement** (les 3 classes 0,82–0,88, vs *modérée* 0,55 en tentative n°1) et un **gain net +0,018**,
porté par un meilleur rappel *maximale* (sécurité) et *différée* pour un coût modeste sur *modérée* : un
**profil orienté sécurité**, souhaitable en triage. Adapter persisté (HF `ghislaindelabie/oc14-qwen3-1.7b-triage-dpo-rpo`),
reproductible (seed 3407), journalisé sur W&B.

**Décision.** Le **SFT v9 reste servi en V1** : le gain +0,018 est **réel mais modeste** (dans la marge de
bruit à n=300) → pas de swap de production sur cette seule base. Mais la leçon **est** le résultat :
l'échec initial venait des **paires + l'absence de régularisation, pas de la méthode DPO**. Avec des paires
équilibrées et `rpo_alpha`, le DPO **améliore** le modèle sans effondrer le milieu — le levier d'alignement
clinique attendu par le brief, désormais **démontré fonctionnel** (et directement extensible : plus de hard
negatives, plus de pas).

---

## 4. L'agent de triage (architecture)

**La chaîne LangGraph (`StateGraph`) — nœuds réels + branche de sécurité.** L'état `TriageCase` circule de nœud en nœud ; le nœud `anonymisation` efface le texte brut (frontière RGPD) ; `explication` porte l'unique conditionnelle (override red-flag).

<svg width="1120" height="170" viewBox="0 0 1120 170" style="max-width:100%;height:auto;font-family:sans-serif" xmlns="http://www.w3.org/2000/svg"> <defs> <marker id="arG" markerWidth="9" markerHeight="9" refX="6" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill="#475569"/></marker> <marker id="arGr" markerWidth="9" markerHeight="9" refX="6" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill="#dc2626"/></marker> </defs><g fill="none" stroke-width="1.5"> <rect x="18" y="88" width="138" height="42" rx="8" fill="#fef2f2" stroke="#dc2626" stroke-width="2"/> <rect x="189" y="88" width="138" height="42" rx="8" fill="#f8fafc" stroke="#64748b"/> <rect x="360" y="88" width="138" height="42" rx="8" fill="#f8fafc" stroke="#64748b"/> <rect x="531" y="88" width="138" height="42" rx="8" fill="#fffbeb" stroke="#d97706" stroke-width="2"/> <rect x="702" y="88" width="138" height="42" rx="8" fill="#f8fafc" stroke="#64748b"/> <rect x="873" y="88" width="138" height="42" rx="8" fill="#f8fafc" stroke="#64748b"/> </g> <text x="10" y="18" font-size="10.5" fill="#6b7280">START → … → END · état partagé : TriageCase</text> <rect x="612" y="28" width="248" height="34" rx="7" fill="#fef2f2" stroke="#dc2626" stroke-width="1.8"/> <text x="736" y="49" text-anchor="middle" font-size="10.5" font-weight="bold" fill="#b91c1c">si red_flag → urgence maximale</text> <path d="M700,88 L700,62" stroke="#dc2626" stroke-width="1.6" fill="none" marker-end="url(#arGr)"/> <g stroke="#475569" stroke-width="1.6" marker-end="url(#arG)"> <line x1="156" y1="105" x2="171" y2="105"/><line x1="327" y1="105" x2="342" y2="105"/> <line x1="498" y1="105" x2="513" y2="105"/><line x1="669" y1="105" x2="684" y2="105"/> <line x1="840" y1="105" x2="855" y2="105"/><line x1="1011" y1="105" x2="1026" y2="105"/> </g> <g font-family="monospace" font-size="11" font-weight="bold" text-anchor="middle" fill="#334155"> <text x="85" y="103">anonymis.</text><text x="256" y="103">pretrait.</text><text x="427" y="103">triage</text> <text x="598" y="103">explication</text><text x="769" y="103">persist.</text><text x="940" y="103">sih</text><text x="1070" y="103" fill="#16a34a">END</text> </g> <g font-size="9" text-anchor="middle" fill="#475569"> <text x="85" y="120">RGPD · hash</text><text x="256" y="120">red-flags</text><text x="427" y="120">LLM vLLM</text> <text x="598" y="120">+ confiance</text><text x="769" y="120">SQLite</text><text x="940" y="120">FHIR mock</text> </g>  </svg>

L'agent **est** la chaîne du parcours patient exigée par le brief. Il est implémenté comme un `StateGraph`
LangGraph (linéaire + un unique override de sécurité conditionnel : la profondeur du framework n'est pas
l'exigence graduée — la chaîne l'est).

### 4.1 Le schéma de la chaîne (= le parcours patient)

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

### 4.2 Chaque composant du cahier des charges

- **Collecte — questionnaire adaptatif** (`questionnaire.py`). Trois champs cœur demandés dans l'ordre
  (motif → début → intensité 1-10). L'adaptativité : si le motif porte un signal d'alerte connu (ex.
  « douleur thoracique »), le questionnaire **insère une question de suivi ciblée** (« la douleur
  irradie-t-elle ? sueurs ? essoufflement ? ») avant de finir. Déterministe et testable — le raisonnement est
  laissé au LLM en aval. Une réponse vide ne remplit **pas** le champ (on re-pose la question) pour éviter un
  verdict confiant sur zéro information.
- **Anonymisation** (`anonymization.py`, nœud `anonymisation`). Frontière RGPD (voir §2.5) : masque les
  identifiants directs, remplace âge/date par `[AGE]`/`[DATE]` (gardés car pertinents au triage), efface le
  brut, calcule `input_sha256`.
- **Prétraitement / validation.** Détecte les **signes d'alerte** (`detect_red_flags`) et valide l'entrée
  (non vide).
- **Triage — le LLM fine-tuné** (`backend.py`). Backend **agnostique** : `stub` (réponse structurée canonique,
  sans GPU — CI/démo/fallback) ou `real` (endpoint vLLM OpenAI-compatible, mêmes réglages que le serving :
  temperature 0, stop `<|im_end|>`, thinking désactivé). Toute erreur backend → *fallback* structuré sûr
  (jamais de propagation d'exception).
- **Explication** (nœud `explication`). Parse la sortie en {niveau, justification, recommandation, disclaimer}.
  **Override de sécurité (l'unique conditionnel de la chaîne) : tout signe d'alerte détecté force l'urgence à
  *maximale*.** Une sortie non structurée retombe sur *modérée* par défaut (sûr) plutôt que de planter.
- **Persistance** (`store.py`, SQLite). Écrit le **dossier** post-anonymisation : `session_id`,
  `interaction_id`, timestamp, `model_version`, `symptoms_anon`, urgence, justification, recommandation,
  **`input_sha256`**, `disclaimer_present`, `latency_ms`, drapeau `deleted` (droit à l'effacement). **Aucun
  texte brut, aucune PII.**
- **Intégration SIH** (`sih.py`). Projette le cas dans un **Bundle FHIR R4** (Encounter + Observation) —
  **simulé** : c'est la *forme* d'un push SIH réel, pas une connexion hospitalière. Le `subject` est
  **pseudonyme** (le `session_id` synthétique, jamais un nom).

### 4.3 API, UI, traçabilité

- **API FastAPI** (`service.py`) : `/session/start`, `/session/answer` (pilote le questionnaire tour par
  tour puis lance la chaîne), `/session/{id}` (historique), `/sessions`, `/health`. Les réponses de session
  sont en mémoire (transitoires) ; tout ce qui est **persisté** passe d'abord par l'anonymisation.
- **UI patient Gradio** (`agent/ui.py`), avec un accordéon **Traçabilité** montrant en direct le `req-id`, le
  texte anonymisé stocké et l'absence de nom/téléphone — exactement la correction demandée à l'évaluation
  précédente.
- **Traçabilité pour audit** : chaque interaction est traçable par `req-id` + `input_sha256` **sans** conserver
  de donnée personnelle, avec la latence par nœud mesurée dans la trace.

---

## 5. Évaluation & métriques

### 5.1 Protocole

- **Jeu de test.** *Gold* **stratifié 100/100/100** (maximale/modérée/différée), **disjoint** du train
  (pas de fuite), issu du consensus 3-LLM unanime.
- **Décodage greedy** (temperature 0) → **déterministe et reproductible** (une version antérieure utilisait un
  décodage échantillonné, source d'inflation).
- **Métriques triage-first** (`eval/metrics.py`) : **macro-F1** en tête (l'accuracy brute est *gameable* en
  sur-prédisant *maximale* ; sur un test équilibré, macro-F1 ≈ accuracy), rappel/précision **par classe**,
  **intervalles de confiance de Wilson**, **matrice de confusion**, et contrôles comportementaux
  (présence du disclaimer, respect du format, absence de `<think>`).

### 5.2 Résultats (SFT v9 servi vs Base)

| Métrique (n=300, greedy, sans fuite) | Base (non entraîné) | **SFT v9 (servi)** |
|---|--:|--:|
| **macro-F1** | **0,19** | **0,82** |
| accuracy | 0,25 | 0,82 |
| rappel *urgence maximale* (sécurité) | 0,70 | **0,90 [IC 0,83–0,95]** |
| rappel *urgence modérée* | 0,05 | 0,85 |
| rappel *urgence différée* | 0,00 | 0,71 |
| Cohen κ vs gold | — | 0,73 |
| format / disclaimer | 0,68 / 0,00 | **1,00 / 1,00** |

**Matrice de confusion (v9, gold → prédit) :** maximale 90 ✓ / 9→modérée / **1→différée** ; modérée 85 ✓ /
12→maximale / 3→différée ; différée 71 ✓ / **29→modérée** / 0→maximale.

**Lecture.** Le fine-tuning achète **0,19 → 0,82** de macro-F1 et enseigne le format + le disclaimer **à
partir de zéro** (0,68/0,00 → 1,00/1,00). Le Base, lui, répond « maximale » ou **échoue à produire un niveau
exploitable dans ~32 %** des cas (96 réponses `(none)`), ne distingue jamais les niveaux bas, n'émet jamais
l'avertissement. Le modèle affiné **ne sous-trie presque jamais** (1 seule urgence rétrogradée à *différée*)
mais **sur-trie le bas** (*différée* : 29/100 poussés à *modérée*) — c'est le sens **sûr** du biais, payé en
précision.

### 5.3 L'audit adverse (la rigueur comme atout)

Un score antérieur de **0,81 a été retiré** après un audit adverse du pipeline : il était **gonflé** par (a)
une **fuite éval→train** (66 lignes de QA labellisées gold présentes aussi à l'entraînement, retirées), (b)
un **décodage échantillonné** qui tombait parfois par chance sur la bonne classe rare, et (c) des lignes
non-consensuelles bruitées. Une fois corrigé (greedy + sans fuite), un intermédiaire a mesuré 0,65,
révélant l'effondrement de *différée* ; le correctif de rééquilibrage (§3.3) l'a remonté à **0,82** —
reproductible et sans fuite, contrairement au 0,81 retiré. Je préfère un 0,82 défendable à un 0,81 flatteur.

### 5.4 Latence & pertinence

- **Latence.** La chaîne mesure la latence par nœud (trace `ms`) et la latence totale (`latency_ms` dans le
  dossier). En mode *stub* (CPU, sans modèle) la chaîne complète tourne en quelques millisecondes. La latence
  **de bout en bout avec le vrai modèle** a été mesurée sur l'endpoint **RunPod** : **~2 s à chaud**. Le
  cold-start serverless (scale-to-zero) est adressé par une requête de préchauffe (voir §6).
- **Pertinence.** Mesurée par le macro-F1 et le rappel par classe ci-dessus ; complétée par les contrôles
  comportementaux (format/disclaimer à 1,00) et l'override de sécurité sur les signes d'alerte.

### 5.5 Parcours de validation — étape par étape

Voici comment j'ai atteint le 0,82 servi. À **chaque** étape j'ai suivi le même rythme :
**je mesure → je constate un problème → je diagnostique la cause → je corrige → je re-mesure**. Ce
n'est pas une ligne droite : c'est une suite de mesures qui m'ont forcé à revenir sur mes pas. C'est,
à mon sens, la partie la plus solide de ce travail.

**Étape 1 — le plancher.** Avant tout entraînement, je mesure le modèle Base nu sur le même *gold* :
**macro-F1 0,19**. Il répond « maximale » par défaut ou échoue à produire un niveau exploitable
(~32 % de `(none)`), ne distingue jamais les classes basses, n'émet jamais l'avertissement. *Constat :*
ce chiffre fixe le plancher et la valeur ajoutée que le fine-tuning devra démontrer.

**Étape 2 — un premier SFT (v8) trop beau… que j'ai retiré moi-même.** L'éval initiale du premier SFT
sur labels de consensus donne **macro-F1 0,813**. Trop propre. Avant de dépenser plus de GPU, je lance
un **audit adverse** du pipeline (`docs/KNOWN_ISSUES.md`, 29 candidats → **16 constats confirmés**). Il
révèle trois vices qui gonflaient le chiffre : (a) une **fuite éval→train** — les mêmes cas cliniques
MediQAl, présents dans le *gold*, étaient indépendamment reformés en lignes de QA à l'entraînement ; (b)
un **décodage échantillonné non déterministe** (`do_sample=True, temp=0.3`, sans seed) qui tombait
parfois par chance sur la bonne classe rare ; (c) des **lignes d'entraînement bruitées** (cas
non-consensuels étiquetés par un tie-break arbitraire). *Décision :* **RETIRER le 0,813.** Point clé — le
correctif n'est pas venu d'un évaluateur, mais de mon propre audit : j'ai détecté et retiré mon meilleur
chiffre.

**Étape 3 — une éval corrigée… qui expose un nouveau problème.** Je corrige les trois vices :
dé-duplication garantissant `gold ∩ train = 0`, **décodage greedy déterministe** (temperature 0), et
métriques robustes (verdict ancré sur la ligne `Niveau d'urgence :`, IC de Wilson). Re-mesure :
**macro-F1 0,653.** Bien plus bas — mais *vrai*. Et cette éval corrigée met au jour un défaut que le
chiffre gonflé masquait : un **effondrement de la classe *différée*** (rappel **0,28** ; 72 cas sur 100
poussés à *modérée*). *Diagnostic :* le nettoyage (correct) des cas non-consensuels avait **affamé** la
classe *différée*, car les cas peu urgents sont sur-représentés parmi les désaccords retirés. Le modèle
**sur-triait** systématiquement le bas.

**Étape 4 — rééquilibrage → v9, le modèle livré.** Je cible la cause, pas le symptôme : (a) j'**assouplis
la règle de consensus** (`n_agree ≥ 2` — garder les majorités 2/3 légitimes, ne retirer que les vrais
désaccords), ce qui restaure le signal *différée* sans réintroduire de fuite ; (b) je **sur-échantillonne
×8 les vignettes cliniques** rédigées à la main (porteuses de *différée* équilibrée + triage EN).
Re-mesure sur le **même** *gold* greedy : **SFT v9 macro-F1 0,822** — rappel *différée* **0,28 → 0,71**,
*maximale* **0,90 [IC 0,83–0,95]**, format/disclaimer **1,00**. **C'est le modèle servi.**

**Étape 5 — DPO, un négatif instructif assumé.** Je tente d'aligner davantage via un DPO sur des paires
de préférences de triage. Re-mesure : **macro-F1 0,799 < 0,822.** *Diagnostic (littérature à l'appui) :*
la classe du milieu (*modérée*) était systématiquement le côté « rejeté » des paires (sous-triage depuis
*maximale*, sur-triage depuis *différée*) et beaucoup de paires étaient quasi-identiques → un
**déplacement de vraisemblance** (*likelihood displacement* — Razin et al. [2410.08847], Smaug/DPO-Positive
[2402.13228]) qui **effondre le milieu** (*modérée* 0,85 → 0,55). De plus, la cible d'alignement du DPO
(format, disclaimer, escalade) était **déjà saturée à 1,00 par le SFT** → aucune marge à gagner. *Décision :*
**conserver le SFT v9** en V1 ; documenter le DPO comme un **négatif bien diagnostiqué** — **repris et corrigé
à l'étape 7**.

**Étape 6 — v10, l'amélioration du *jeu de données*.** Le vrai levier du post-mortem DPO n'est pas une autre
méthode d'alignement, c'est un **meilleur jeu de données**. Pour réduire le biais de sur-triage résiduel, je
**rééquilibre le train** (part *maximale* **54 % → 35 %**, *différée* **12 % → 18 %** — je garde tous les cas
*différée* et *modérée*, je plafonne l'abondant *maximale*) et je **garantis une justification clinique réelle
par cas** (issue d'un annotateur du consensus, plus de canned fallback). **Entraîné puis évalué** sur le même
*gold* : **macro-F1 0,822**, parité stricte avec v9 (matrice identique). Le gain *visé* — la **qualité des
justifications** — n'est pas capté par la F1 par niveau : je le traite comme une **hypothèse non encore
mesurée** (cf. §5.6), donc **v9 reste servi** (`scripts/build_sft_v10.py`).

**Étape 7 — DPO, repris et corrigé (`rpo_alpha`).** Fort du diagnostic de l'étape 5, je **corrige les paires**
(distribution rejetée équilibrée — chaque niveau *choisi* ET *rejeté*) et j'ajoute la **régularisation
anti-effondrement `rpo_alpha`** (ancre NLL sur la réponse choisie). Comparaison **pommes-à-pommes** (mêmes 300
*gold*, même harness, mêmes poids de base, adapter DPO activé/désactivé) : **v9 0,827 → DPO 0,845 (+0,018)**,
**sans effondrement** (les 3 classes 0,82–0,88), profil orienté sécurité. Le DPO est donc **démontré
fonctionnel** ; le gain restant **modeste** (marge de bruit à n=300) → **v9 reste servi en V1**, le DPO corrigé
est persisté (HF) et directement extensible (cf. §3.4).

**Récapitulatif de la progression :**

| Étape | Modèle / éval | macro-F1 | Statut |
|---|---|--:|---|
| 1 | Base (non entraîné) | **0,19** | plancher |
| 2 | SFT v8 (première éval) | 0,813 | **RETIRÉ** (fuite + échantillonnage + bruit) |
| 3 | SFT v8 (éval corrigée) | 0,653 | intermédiaire — expose l'effondrement *différée* |
| 4 | **SFT v9** | **0,822** | **SERVI** |
| 5 | SFT v9 + DPO (v1) | 0,799 | négatif instructif — effondrement *modérée* diagnostiqué |
| 6 | SFT v10 | 0,822 | parité (amélioration *données* : justifications, hypothèse) |
| 7 | **SFT v9 + DPO (`rpo_alpha`)** | **0,845** | corrigé — +0,018 pommes-à-pommes, sans effondrement (v9 servi en V1) |

Ces bras sont journalisés côté **W&B** (projet `oc14-triage-eval`) dans un tableau de bord de **comparaison
d'expériences** : *base 0,19 · sft-v8 0,813 RETIRÉ · sft-v8-honnête 0,653 · **sft-v9 0,822 SERVI** · dpo 0,799 ·
**dpo-`rpo_alpha` 0,845***. Un **W&B Sweep** dédié (learning rate / LoRA r / warmup, early-stop Hyperband) y
ajoute les **courbes d'entraînement *live***. Provenance dans `config.kernel` de chaque run (voir §5.6).

### 5.6 Suivi d'expériences (W&B)

Un tableau de bord W&B compare les 5 bras : base 0,19 · sft-v8 0,813 **RETIRÉ** · sft-v8-honnête 0,653 ·
**sft-v9 0,822 SERVI** · dpo 0,799. Ce sont des **résumés d'éval finaux journalisés manuellement** (provenance
dans `config.kernel` de chaque run), **pas** des courbes d'entraînement *live* — leur capture continue est
câblée et constitue un point d'amélioration MLOps (§8).

> **Note (V1 servie = SFT v9).** Une itération **v10** — *jeu de données amélioré* (justifications de consensus
> réelles intégrées + rééquilibrage) — a été **entraînée puis évaluée** sur le même *gold* n=300 (T4, greedy,
> sans fuite) : **macro-F1 0,822**, soit une **parité stricte** avec v9 (rappels 0,90 / 0,85 / 0,71, κ 0,73 —
> matrice de confusion identique). Le gain *visé* de v10 est la **qualité des justifications** (rationales de
> consensus réelles) — non captée par la F1 par niveau, et **traitée ici comme une hypothèse non encore
> quantifiée** (une éval en préférence par juge LLM reste à mener). **Aucun gain de métrique ne justifiant un
> redéploiement, v9 reste servi** ; v10 est une amélioration de *données*, pas un ajustement de métrique.

---

## 6. Serving, CI/CD & déploiement

### 6.1 Architecture de serving

Deux couches, conteneurisées (Docker) :
- **Le modèle** tourne sur **vLLM** (serveur d'inférence rapide, API OpenAI-compatible), cible **RunPod
  serverless** (pay-per-second, scale-to-zero — le GPU le moins cher, cold-start assumé et préchauffé).
- **Un wrapper FastAPI `/triage`** (`serving/app.py`) devant vLLM, qui : (1) **injecte le prompt système de
  triage** (l'appelant n'envoie que le texte patient) ; (2) **force le non-thinking + stop sur `<|im_end|>`** ;
  (3) pose une **API-key gate** (header `X-API-Key`) et un **journal d'audit privacy-safe** (métadonnées
  seulement, jamais le texte patient). Testé unitairement contre un backend vLLM **mocké**.

L'**agent complet** (§4) est lui aussi exposé par FastAPI et conteneurisé ; le backend de triage bascule du
*stub* au vLLM réel via des variables d'environnement, sans changer le code.

### 6.2 CI/CD (GitHub Actions)

`.github/workflows/ci.yml` :
- **`lint-and-test`** (à chaque push / PR) : `uv sync`, **ruff** (lint), **pytest** (suite complète) — **verts**.
  La CI ne touche pas de GPU (pas d'entraînement dans le pipeline).
- **`deploy`** (job `workflow_dispatch` manuel) : build + push de l'image agent vers **GHCR**, puis un refresh
  d'endpoint **RunPod** gardé derrière le secret `RUNPOD_API_KEY` (no-op sans la clé). Déploiement automatisé
  et reproductible sans exiger le secret pour la CI de tous les jours.

Secrets : clés en `.env` / secrets GitHub, **jamais commités**.

### 6.3 Check-list go / no-go (avant pilote)

| Contrôle | Seuil / attendu | Statut |
|---|---|---|
| `/health` répond | `{"status":"ok"}` | ✓ |
| Chaîne end-to-end | 3 cas de démo donnent le bon niveau | ✓ (stub) |
| Anonymisation RGPD | aucun identifiant direct dans le dossier (test de fuite) | ✓ |
| Traçabilité | `req-id` + `input_sha256` par interaction, pas de texte brut | ✓ |
| Rappel *urgence maximale* | ≥ 0,83 (plancher IC) | ✓ (0,90 [0,83–0,95]) |
| Taux de disclaimer | = 1,00 | ✓ |
| Secrets protégés | clés en `.env`/secrets, jamais commit | ✓ |
| Endpoint cloud vLLM | déployé + latence mesurée | ✓ (RunPod, ~2 s à chaud) |
| Latence (à chaud) | mesurée sur l'endpoint réel | ✓ (~2 s, RunPod) |

---

## 7. Limites (assumées)

1. **Standard « argent », pas de vérité clinique.** Les étiquettes sont un consensus de 3 LLM (κ≈0,67), pas un
   jury de cliniciens. L'accord LLM↔clinicien n'est que *modéré* dans la littérature. À valider cliniquement
   avant tout usage réel.
2. **Circularité (le caveat clé d'éval).** Le *gold* = l'étiquette 3-LLM **unanime**, et le modèle est entraîné
   sur les **mêmes** étiquettes 3-LLM → la métrique mesure la **fidélité d'imitation** aux professeurs, pas la
   justesse clinique. Pire, le gold est le sous-ensemble **facile** (unanime) → chiffre **optimiste** vs la
   population complète.
3. **Compromis sur-triage ↔ précision.** Le modèle ne sous-trie presque jamais (sens sûr) mais sur-trie le bas
   (*différée* 0,71) ; l'échelle 3-niveaux concentre le biais résiduel dans la classe du milieu. À la limite,
   un trieur qui appelle tout « modéré-ou-pire » n'a plus de valeur de tri.
4. **Langue : FR-primaire, bilingue faiblement tenu.** Train 79 % FR / 21 % EN, mais l'EN est presque tout du
   QA général ; le triage EN à l'entraînement est mince (**32 lignes SFT-train**), et l'**évaluation est 100 %
   française**. La tâche de triage et son éval sont donc de fait FR-only.
5. **Représentativité du corpus.** Vignettes d'**examen** → sur-représentation du grave (~47 % maximale vs
   ~25-30 % en vraies urgences). Un système de production exigerait un jeu de triage ED **réel, collecté
   prospectivement**.
6. **Constantes vitales non collectées** (pas d'intégration dispositif) — champ réservé, item de roadmap.
7. **Intégration SIH simulée** (mock FHIR R4, pas de connexion hospitalière réelle).
8. **Puissance statistique.** n=100/classe → IC larges (maximale 0,90 → [0,83, 0,95]) ; le plancher de
   sécurité est la **borne basse 0,83**.
9. **Barre de sécurité.** ≥ 1 urgence sur 10 manquée au pire cas → **inacceptable pour un triage autonome**.
   Positionnement : **aide à la décision / human-in-the-loop**.
10. **Améliorations MLOps différées (assumées) :** capture *systématique* des **courbes d'entraînement W&B
    *live*** (un premier W&B Sweep live est en place, cf. §5.5) et une campagne de **latence p50/p95** à plus
    grande échelle.
11. **Robustesse hors-distribution.** Sur une entrée **absurde / non-clinique** (charabia), le modèle n'a pas de
    voie « information insuffisante » : il **confabule** un verdict (observé : charabia → *maximale* avec des
    signes d'alerte hallucinés). C'est un défaut de **robustesse OOD**, pas d'entraînement sur les vrais cas.
    Mitigation : un **garde-fou d'entrée déterministe** (rejeter le charabia *avant* le triage — symétrique de
    l'override red-flag : « garbage in → on refuse », pas « → maximale »), une voie d'**abstention**, et le **HITL**.

---

## 8. Roadmap de passage à l'échelle au CHSA

Suivant la phase 3 du brief (« projection industrielle »), voici les jalons pour passer du POC à un pilote,
puis à la production.

**La cible : un agent toujours *human-in-the-loop*, mais bien plus précis.** Le plafond du POC (rappel
*maximale* 0,90, précision ~0,88) tient à trois limites franchissables — des **étiquettes « argent »**
(consensus LLM, pas de vérité clinicienne), un **corpus d'examen** non représentatif du flux réel des
urgences, et une **petite taille** d'évaluation (n=300). En production, trois leviers combinés visent
**≥ 98 % sur la classe critique**, *tout en restant une aide à la décision, jamais un triage autonome* :

- **Données de l'hôpital.** Remplacer le standard argent par des cas de triage **réels, prospectifs et
  représentatifs du CHSA**, étiquetés et **validés par des cliniciens** (vérité terrain, non plus une
  pré-annotation LLM) — de loin le levier le plus fort sur la précision.
- **Amélioration continue (LLMOps).** Une boucle de **ré-entraînement** : chaque correction clinicienne (via
  le HITL) devient une donnée étiquetée ; monitoring de la dérive → ré-entraînement périodique ; versionnage
  des modèles/jeux + déploiement canari (le pipeline CI/CD est déjà en place).
- **Le human-in-the-loop lui-même.** L'agent **assiste**, le soignant **décide** et valide/corrige chaque
  alerte *maximale* : la précision *effective du système* est portée par le clinicien, et ses corrections
  **ré-entraînent** le modèle — un cercle vertueux qui rapproche progressivement l'agent seul de ce plafond.

Concrètement, les jalons :

1. **Modèle plus grand (32B+).** Passer de Qwen3-1.7B à un modèle 32B+ (le brief le prévoit) pour un gain de
   raisonnement clinique, avec un budget GPU serving revu (quantisation, batching vLLM).
2. **Données réelles + validation clinicienne.** Remplacer le standard argent par un jeu de triage ED **réel,
   collecté prospectivement et représentatif**, sous un **DPA** (accord de traitement des données) ; renforcer
   l'anonymisation (revue humaine d'un échantillon) ; faire **valider les étiquettes par des cliniciens** (le
   consensus LLM devient une pré-annotation, plus la vérité).
3. **Intégration SIH réelle.** Passer du mock FHIR R4 à une connexion **FHIR/HL7** authentifiée au SIH du CHSA
   (Encounter/Observation réels, identités gérées côté SIH, jamais dans le modèle).
4. **Intégration dispositif des constantes.** La **saisie libre** des constantes est déjà en place
   (questionnaire → prompt → `constantes`, POC) ; l'étape suivante est l'**intégration dispositif** (capture
   structurée T°/FC/TA/SpO₂ depuis les moniteurs) et des **seuils d'alerte vitaux** (ex. SpO₂ < 90 % →
   red-flag), au-delà du raisonnement libre actuel du modèle sur les constantes.
5. **Human-in-the-loop + monitoring/dérive.** Garder la décision finale humaine ; câbler en production les
   alertes sur taux d'erreur / latence p95, la **dérive du taux d'*urgence maximale*** (sur-triage), et une
   revue périodique d'un échantillon par un clinicien. Ajouter un **score de confiance au *runtime***, dérivé
   des **logprobs** de vLLM — la probabilité que le modèle attribue au *token* du niveau d'urgence qu'il a
   choisi (proba élevée → sûr ; hésitation entre deux niveaux → proba basse) — afin de **router automatiquement
   les cas peu sûrs vers une revue clinicienne**. C'est un signal bien plus fiable qu'une confiance
   auto-déclarée par le modèle (« je suis sûr à 80 % », notoirement mal calibrée).
6. **DPO d'alignement clinique.** Refaire un DPO ciblé (paires rééquilibrées pour ne pas écraser *modérée*)
   pour forcer le sous-triage résiduel vers zéro — c'est le rôle « conformité protocoles » attendu du DPO.
7. **MLOps.** Tracking W&B live (courbes d'entraînement, pas seulement les résumés), pipeline de
   **ré-entraînement** déclenché sur dérive ou nouvelle donnée validée, versionnage des modèles et des jeux,
   déploiement canari via le job CI/CD déjà en place.
8. **Recherche d'hyperparamètres.** Le brief invite à « itérer sur les hyperparamètres ». Pour le POC j'ai
   retenu des valeurs par défaut éprouvées (recommandations Unsloth pour Qwen3-1.7B) et concentré l'effort sur
   la **qualité des données** — le correctif *différée* a rapporté **+0,17 de macro-F1**, bien plus qu'un
   réglage fin ne l'aurait fait. Prochain levier : un **sweep systématique** (W&B Sweeps, grille ou bayésien)
   optimisant le macro-F1 et surtout le **rappel *maximale*** sur le gold — côté SFT sur le **rang LoRA `r` et
   `alpha`, le learning rate, le nombre d'époques, le dropout LoRA, les modules cibles, le warmup** ; côté DPO
   sur le **`beta` (force de régularisation) et le learning rate**. À brancher sur le tracking W&B et la boucle
   de ré-entraînement (point 7).
9. **Bilinguisme complet** (si requis) : grossir le triage EN à l'entraînement + ajouter une tranche d'éval
   EN indépendante (ex. `medical-triage-500`).
10. **Questionnaire adaptatif hybride (piste majeure — un projet à part entière).** Aujourd'hui la collecte est
   **déterministe** (table de règles + un suivi ciblé sur *red-flag*) : sûre et testable, mais figée. L'évolution
   serait un **workflow hybride** — un squelette déterministe qui **garantit les étapes obligatoires** (motif,
   antécédents, constantes, dépistage des *red-flags*) et, à l'intérieur de ce cadre, un **LLM entraîné à
   raffiner les questions** : reformuler, approfondir, et choisir la **prochaine question la plus informative**
   selon les réponses déjà données — pour une collecte d'information **plus riche** sans jamais sauter une étape
   critique. Cela suppose ses propres garde-fous (sécurité conversationnelle, évaluation de la *qualité de
   collecte*), d'où un chantier distinct du présent POC.

---

## 9. Conclusion

J'ai livré un **POC d'agent de triage médical** couvrant les cinq livrables du brief : un dataset bilingue
anonymisé et documenté (RGPD, Recital 26), un modèle Qwen3-1.7B spécialisé par SFT+LoRA (avec une tentative
de DPO analysée), un agent complet exposé par API et servi sur un endpoint vLLM/RunPod déployé, un pipeline
CI/CD vert, et ce rapport.

Le résultat central est un **signal de progrès net** : **macro-F1 0,19 → 0,82** entre le modèle
Base non entraîné et le SFT servi, avec un rappel de 0,90 sur les urgences vitales et un format/disclaimer
parfaits. La démarche a été **rigoureuse et transparente** : audit adverse, fuite éval→train éliminée,
intervalles de confiance reportés, un score gonflé (0,81) retiré, et un DPO documenté comme résultat négatif
instructif plutôt que survendu.

Surtout, je positionne ce système pour ce qu'il est : une **aide à la décision sous supervision humaine**,
**pas** un trieur autonome. La barre de sécurité n'est pas encore atteinte pour un usage autonome, et les
étiquettes restent un standard argent à valider cliniquement. La roadmap (§8) trace la voie du POC vers un
pilote crédible : modèle plus grand, données réelles validées par des cliniciens, intégration SIH réelle,
collecte des constantes, human-in-the-loop et MLOps. Le POC démontre la **faisabilité technique** et la
**valeur clinique potentielle** demandées — avec la rigueur méthodologique qu'exige un déploiement en santé.
