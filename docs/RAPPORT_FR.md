# Rapport technique — POC d'agent de triage médical (CHSA)

> **Livrable 5 — OpenClassrooms « Finetunez votre propre LLM ».**
> Auteur : Ghislain Delabie · IA Engineer junior · mission Centre Hospitalier Saint-Aurélien (CHSA).
> Dépôt : `github.com/ghislaindelabie/oc14-medical-triage-llm` (public).
> Tous les chiffres cités sont mesurés (sources : `DEVELOPMENT_JOURNAL.md`, `data/cards/DATA_CARD.md`,
> les logs de runs Kaggle, les fichiers `data/processed/_*_stats.json`). Rien n'est estimé sauf mention
> explicite. Ce document tient en ≤ 20 pages.

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

**Problème.** À la session de mentorat, un risque a été identifié : se concentrer uniquement sur le
fine-tuning du LLM et livrer un « bon score » sans l'**agent** que le brief exige (questionnaire adaptatif,
intégration SIH, traçabilité). Le LLM affiné n'est qu'une **brique** de la chaîne demandée.

**Décision.** Construire la **chaîne fonctionnelle complète** du parcours patient autour du LLM — collecte →
anonymisation → prétraitement/validation → triage (le LLM) → explication → persistance → intégration SIH —
et faire du LLM affiné le cœur clinique de cette chaîne. Le score de triage reste central (tâche graduée),
mais il est encapsulé dans un agent réel, testé de bout en bout.

**Critère.** Chaque composant nommé dans le mail du Dr. Dubois doit être présent et démontrable en direct.

**Preuve.** L'agent est implémenté en LangGraph (`src/oc14_triage/agent/graph.py`), exposé par une API
FastAPI (`agent/service.py`), doté d'une UI patient Gradio, et couvert par **97 tests** (TDD). La chaîne
tourne de bout en bout sans GPU en mode *stub*.

### 1.3 Carte de conformité (livrable → où c'est démontré)

| Livrable / exigence du brief | Où c'est démontré dans ce projet | Statut |
|---|---|---|
| **L1 — Dataset bilingue anonymisé, versionné, SFT + DPO** | `data/kaggle_upload/*.jsonl` ; `DATA_CARD.md` ; passe Presidio (audit) ; §2 | ✅ |
| L1 — Schéma des métadonnées (symptômes / antécédents / constantes / source / confiance) | `METADATA_SCHEMA.md` ; §2.4 | ✅ |
| L1 — Justification RGPD | `DATA_CARD.md` (Recital 26) ; passe Presidio ; §2.5 | ✅ |
| L1 — Séparation train / éval (auditabilité, pas de fuite) | éval-gold disjointe, audit adverse ; §2, §5 | ✅ |
| **L2 — Modèle Qwen3-1.7B affiné SFT+LoRA puis aligné DPO, poids + métriques** | notebooks Kaggle ; adapter LoRA ; SFT v9 servi ; DPO #2 analysé ; §3, §5 | ✅ (DPO = négatif honnête, non retenu) |
| L2 — Traçabilité entraînement (logs, seed, checkpoints) | seed 3407, `adapter_config.json`, tracking W&B ; §3, §6 | ✅ |
| **L3 — Endpoint cloud vLLM, API de démo, inférence rapide** | wrapper FastAPI `/triage` conteneurisé ; agent FastAPI ; plan RunPod vLLM ; §4, §6 | ⏳ endpoint *live* en attente d'une clé |
| **L4 — CI/CD GitHub Actions (tests + déploiement)** | `.github/workflows/ci.yml` (ruff + pytest verts + job `deploy`) ; §6 | ✅ (CI) / ⏳ (déploiement *live*) |
| **L5 — Rapport ≤ 20 p (méthodo données + entraînement · métriques · analyse · roadmap)** | ce document | ✅ |
| Agent — questionnaire adaptatif | `agent/questionnaire.py` ; §4 | ✅ |
| Agent — évaluation de priorité selon protocoles | rubrique ESI/MTS ; LLM affiné ; §2.2, §4 | ✅ |
| Agent — explications claires | nœud `explication` ; §4 | ✅ |
| Agent — intégration SIH | mock FHIR R4 ; §4 | ✅ (simulé) |
| Agent — traçabilité pour audits | dossier SQLite, `req-id`, `input_sha256` ; §4 | ✅ |

Le reste du rapport détaille chaque ligne, avec l'accent (selon la demande) sur les décisions **Données/RGPD**
(§2) et **Entraînement/DPO** (§3), présentées en **Problème → Options → Décision → Critère → Preuve**.

---

## 2. Données & RGPD (méthodologie décision-driven)

### 2.1 Sources & licences

Toutes les sources sont **publiques** ; **aucune** n'est un dossier patient réel. La sélection a été guidée
par un critère de sûreté juridique (licence redistribuable) et de pertinence à la tâche.

| Source | Langue | Licence | Rôle | Statut |
|---|---|---|---|---|
| ANR-MALADES/**MediQAl** | FR | CC-BY-4.0 | QA médicale FR + (cas cliniques →) triage | active |
| keivalya/**MedQuAD** | EN | CC-BY-4.0 (Ben Abacha 2019) | QA médicale EN | active |
| TsinghuaC3I/**UltraMedical-Preference** | EN | MIT | paires DPO | *seulement dans le DPO v1 abandonné* |
| qanastek/**FrenchMedMCQA** | FR | Apache-2.0 | QA MCQ pharmacie FR | **désactivée** (loader incompatible `datasets≥3`) |
| **MIETIC / MIMIC** (triage) | EN | PhysioNet *credentialed* | triage EN réel | **exclue** |

**Problème (l'exclusion MIETIC/MIMIC).** Le seul corpus de triage « réel » facilement accessible (MIETIC)
dérive de MIMIC, sous licence PhysioNet **accréditée à non-redistribution**.

**Options.** (a) l'utiliser et redistribuer discrètement le dérivé ; (b) l'exclure et fabriquer un signal
de triage à partir de sources redistribuables.

**Décision.** **Exclure** MIETIC/MIMIC.

**Critère.** Protéger l'histoire « aucune donnée patient réelle / conforme RGPD » et respecter la licence —
un POC hospitalier ne peut pas s'asseoir sur une DUA.

**Preuve.** Exclusion documentée (`DATA_CARD.md`, recherche `00` §0b) ; le signal de triage vient à la place
de vignettes cliniques MediQAl (questions d'examen) + vignettes rédigées à la main — provenance propre.

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

Le SFT combine le signal de triage (vignettes + cas MediQAl reformés dans la structure
niveau→justification→recommandation) et un socle de QA médicale (qui donne au modèle les connaissances pour
*justifier* un triage). Le DPO est un jeu de préférences de triage **équilibré par direction d'erreur**.

| Jeu | Train / Val | Détail |
|---|---|---|
| **SFT** | **5 598 / 562** | Langue : fr 4 571 / en 1 027 (~82 % FR). Type : triage 2 041 / QA 3 557. |
| ↳ sources SFT | | `mediqal_mcqu` 2 166 · `llm_triage` 1 953 · `medquad` 995 · `mediqal_oeq` 396 · `vignette` 88 |
| ↳ urgence (rows triage) | | maximale 1 098 · modérée 689 · différée 254 |
| **DPO** | **211 / 24** | équilibré par direction : `dpo_under` 103 · `dpo_mod` 50 · `dpo_over` 48 · `safety` 10 |
| **Éval gold** | **300** | stratifié 100 / 100 / 100 (maximale / modérée / différée) |

### 2.4 Schéma des métadonnées

Le brief demande explicitement un schéma couvrant **symptômes, antécédents, constantes, source, niveau de
confiance** (`METADATA_SCHEMA.md`). Je le documente pour les deux niveaux : les enregistrements de données
d'entraînement et le **dossier d'exécution** écrit par l'agent (même forme que l'enregistrement SIH).

| Champ (brief) | Clé | Provenance dans le POC |
|---|---|---|
| Symptômes | `symptoms` | Runtime : `motif` + suivi ; dataset : le texte de la vignette |
| Antécédents | `medical_history` | Runtime : texte libre optionnel ; absent du corpus d'examen (limite documentée) |
| **Constantes** | `vitals` | **Non collectées** dans le POC (pas d'intégration dispositif) — champ réservé, item de roadmap |
| Source | `source` | dataset : `mediqal_mcqu`/`_oeq`/`medquad`/`llm_triage`/`vignette` ; runtime : `chsa-triage-poc` |
| Niveau de confiance | `confidence_level` | dataset : dérivé du consensus 3-LLM — unanime (3/3) → `high`, majorité (2/3) → `medium` |

### 2.5 Anonymisation RGPD : la passe corpus + la frontière runtime

**Problème.** Le brief exige une conformité RGPD **prouvée** (Presidio recommandé), y compris un contrôle
qu'« aucune donnée personnelle identifiable ne subsiste ». Mais mes sources sont des questions d'examen et des
vignettes synthétiques : le risque théorique de PII est faible — reste à le **démontrer**, honnêtement.

**Décision (deux niveaux).**
- **Passe corpus (offline).** J'ai fait tourner Microsoft **Presidio + spaCy** sur **6 695 textes** du corpus,
  avec un journal d'audit par enregistrement (SHA-256 + entités, **sans stocker le texte brut**).
- **Anonymisation au *runtime* dans l'agent.** Le nœud `anonymisation` est la **frontière RGPD** : tout input
  patient est anonymisé *avant* tout stockage ; à partir de ce nœud le `raw_text` est effacé, et l'input brut
  ne survit que sous forme de hash **`input_sha256`** — « **hash pour la traçabilité, anonymisation pour le
  stockage** ».

**Critère.** (1) Garantir l'invariant « aucun identifiant direct dans le dossier persisté », indépendamment
de la qualité du modèle NER ; (2) interpréter les résultats de la passe **honnêtement**, sans surinterpréter.

**Preuve — les détections brutes** (audit `anonymization_audit.json`) :

| Entité | Détections |
|---|---|
| LOCATION | 13 298 |
| AGE | 3 624 |
| PERSON | 3 330 |
| PHONE_NUMBER | 470 |
| DATE_TIME | 97 |

**Interprétation honnête (c'est le point que je tenais à clarifier).** Ces détections sont **massivement des
faux positifs cliniques**, pas de vrais patients : les LOCATION sont surtout des **localisations anatomiques**,
les PERSON des **éponymes cliniques** (maladies/signes nommés) et des **personas d'examen génériques**
(« Monsieur X », « Madame Y »), les AGE des âges d'énoncé. C'est cohérent avec la nature des sources (questions
ECN/pharmacie + vignettes synthétiques) : **il n'y a pas de vrai patient**, donc les données sont non
personnelles par construction → hors champ RGPD (**Recital 26**). La passe Presidio ne « nettoie » donc pas
des fuites : elle **confirme** l'hypothèse d'absence de PII réelle et documente le processus.

L'architecture de l'anonymiseur (`src/oc14_triage/anonymization.py`) est **à deux couches**, ce qui rend
l'invariant robuste : (a) Presidio+spaCy pour les entités sémantiques (PERSON, LOCATION) ; (b) un **filet
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

**Preuve.** La comparaison n=6 initiale (avec analyse de confusion de gabarit) est conservée dans le journal
mais **dépassée** comme titre. Le chiffre défendable est le n=300 : **Base non entraîné macro-F1 0,19 → SFT
v9 0,82**.

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

### 3.4 DPO : deux tentatives, un négatif honnête, et un recadrage

- **DPO** (Direct Preference Optimization) : montrer au modèle des paires (meilleure, moins bonne) réponses
  pour qu'il préfère la meilleure — alignement sans modèle de récompense séparé.
- **Invariant d'ordonnancement.** Le DPO tourne sur le modèle SFT **adapter LoRA encore attaché** ; la fusion
  en poids 16-bit se fait **une seule fois, après DPO** — jamais entre les étapes.

**DPO #1 (négatif).** Sur UltraMedical-Preference (~99 % du jeu, anglais, hors-tâche). **Régression** :
l'accuracy chute, les deux urgences vitales du test sont manquées, apparition d'artefacts « GPT-isms ». Cause
vérifiable : le DPO a optimisé la **verbosité de style GPT-4**, pas la qualité de triage. Leçon : le problème
était la **composition des données**, pas la méthode.

**DPO #2 (le négatif instructif).**

**Problème.** Refaire le DPO proprement, sur des **paires de préférences de triage équilibrées par direction**
(211/24 : sous-triage 103 + safety 10 / sur-triage 48 / modérée 50), chosen = le bon niveau, rejected = un
niveau **adjacent** erroné.

**Preuve (n=300, greedy) :**

| n=300, greedy | macro-F1 | rappel maximale | rappel modérée | rappel différée | κ |
|---|--:|--:|--:|--:|--:|
| **SFT v9 (servi)** | **0,82** | 0,90 | **0,85** | 0,71 | 0,73 |
| SFT + DPO #2 | 0,80 | 0,92 | **0,55** | 0,96 | 0,72 |

**Analyse (le mécanisme).** Le DPO a **aiguisé les extrêmes** (*différée* 0,71→0,96, maximale 0,90→0,92) mais
**effondré le milieu** (*modérée* 0,85→0,55), pour un macro-F1 net **0,82→0,80**. Pourquoi ? *modérée* est le
niveau **rejected** pour **les deux** types de paires (sous-triage depuis maximale, sur-triage depuis
différée) : elle apparaît ~168× comme « la mauvaise réponse » contre 56× comme « la bonne » → le DPO apprend
à **éviter la classe du milieu**. Les paires de niveaux **adjacents** pénalisent structurellement le milieu.

**Décision.** **Livrer le SFT v9** (macro-F1 0,82, meilleur équilibre) ; reporter le DPO comme **résultat
négatif honnête**, technique démontrée et échec analysé.

**Recadrage clé (ce que j'ai clarifié après le mentorat).** Le brief attend le DPO pour l'**alignement
clinique / la conformité aux protocoles** (préférer une réponse plus sûre), **pas** pour gagner de l'accuracy.
C'est là sa vraie valeur : un levier de sécurité. Le SFT v9 est servi parce qu'il a le meilleur macro-F1 ;
le DPO reste le bon outil pour, à terme, forcer l'unique sous-triage résiduel (1 *maximale* → *différée*)
vers zéro, une fois les paires rééquilibrées pour ne pas écraser la classe intermédiaire.

---

## 4. L'agent de triage (architecture)

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
non-consensuelles bruitées. Une fois corrigé (greedy + sans fuite), un intermédiaire honnête a mesuré 0,65,
révélant l'effondrement de *différée* ; le correctif de rééquilibrage (§3.3) l'a remonté à **0,82 honnête** —
reproductible et sans fuite, contrairement au 0,81 retiré. Je préfère un 0,82 défendable à un 0,81 flatteur.

### 5.4 Latence & pertinence

- **Latence.** La chaîne mesure la latence par nœud (trace `ms`) et la latence totale (`latency_ms` dans le
  dossier). En mode *stub* (CPU, sans modèle) la chaîne complète tourne en quelques millisecondes. La latence
  **de bout en bout avec le vrai modèle** (p50/p95) sera mesurée sur l'endpoint vLLM réel — **en attente de la
  clé cloud** (⏳). Le cold-start serverless est adressé par une requête de préchauffe (voir §6).
- **Pertinence.** Mesurée par le macro-F1 et le rappel par classe ci-dessus ; complétée par les contrôles
  comportementaux (format/disclaimer à 1,00) et l'override de sécurité sur les signes d'alerte.

### 5.5 Suivi d'expériences (W&B)

Un tableau de bord W&B compare les 5 bras : base 0,19 · sft-v8 0,813 **RETIRÉ** · sft-v8-honnête 0,653 ·
**sft-v9 0,822 SERVI** · dpo 0,799. Ce sont des **résumés d'éval finaux journalisés manuellement** (provenance
dans `config.kernel` de chaque run), **pas** des courbes d'entraînement live — la capture live est câblée mais
**en attente** d'un secret `WANDB_API_KEY` côté Kaggle + un re-run (⏳).

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
- **`lint-and-test`** (à chaque push / PR) : `uv sync`, **ruff** (lint), **pytest** (97 tests) — **verts**.
  La CI ne touche pas de GPU (pas d'entraînement dans le pipeline).
- **`deploy`** (job `workflow_dispatch` manuel) : build + push de l'image agent vers **GHCR**, puis un refresh
  d'endpoint **RunPod** gardé derrière le secret `RUNPOD_API_KEY` (no-op sans la clé). Déploiement automatisé
  et reproductible sans exiger le secret pour la CI de tous les jours.

Secrets : clés en `.env` / secrets GitHub, **jamais commités**.

### 6.3 Check-list go / no-go (avant pilote)

| Contrôle | Seuil / attendu | Statut |
|---|---|---|
| `/health` répond | `{"status":"ok"}` | ✅ |
| Chaîne end-to-end | 3 cas de démo donnent le bon niveau | ✅ (stub) |
| Anonymisation RGPD | aucun identifiant direct dans le dossier (test de fuite) | ✅ |
| Traçabilité | `req-id` + `input_sha256` par interaction, pas de texte brut | ✅ |
| Rappel *urgence maximale* | ≥ 0,83 (plancher IC) | ✅ (0,90 [0,83–0,95]) |
| Taux de disclaimer | = 1,00 | ✅ |
| Secrets protégés | clés en `.env`/secrets, jamais commit | ✅ |
| Endpoint cloud vLLM | préchauffe OK + latence mesurée | ⏳ (RunPod, après clé) |
| Latence p50/p95 | mesurée en conditions réelles | ⏳ (endpoint réel) |

---

## 7. Limites (assumées, honnêtes)

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
10. **Éléments en attente (assumés) :** endpoint vLLM *live* (clé cloud), courbes d'entraînement W&B live,
    latence p50/p95 réelle.

---

## 8. Roadmap de passage à l'échelle au CHSA

Suivant la phase 3 du brief (« projection industrielle »), voici les jalons pour passer du POC à un pilote,
puis à la production.

1. **Modèle plus grand (32B+).** Passer de Qwen3-1.7B à un modèle 32B+ (le brief le prévoit) pour un gain de
   raisonnement clinique, avec un budget GPU serving revu (quantisation, batching vLLM).
2. **Données réelles + validation clinicienne.** Remplacer le standard argent par un jeu de triage ED **réel,
   collecté prospectivement et représentatif**, sous un **DPA** (accord de traitement des données) ; renforcer
   l'anonymisation (revue humaine d'un échantillon) ; faire **valider les étiquettes par des cliniciens** (le
   consensus LLM devient une pré-annotation, plus la vérité).
3. **Intégration SIH réelle.** Passer du mock FHIR R4 à une connexion **FHIR/HL7** authentifiée au SIH du CHSA
   (Encounter/Observation réels, identités gérées côté SIH, jamais dans le modèle).
4. **Collecte des constantes vitales.** Câbler la saisie (ou l'intégration dispositif) des constantes
   (T°, FC, TA, SpO₂) — le champ `vitals` est déjà réservé dans le schéma ; les zones de danger vitales sont
   déjà dans la rubrique.
5. **Human-in-the-loop + monitoring/dérive.** Garder la décision finale humaine ; câbler en production les
   alertes sur taux d'erreur / latence p95, la **dérive du taux d'*urgence maximale*** (sur-triage), et une
   revue périodique d'un échantillon par un clinicien.
6. **DPO d'alignement clinique.** Refaire un DPO ciblé (paires rééquilibrées pour ne pas écraser *modérée*)
   pour forcer le sous-triage résiduel vers zéro — c'est le rôle « conformité protocoles » attendu du DPO.
7. **MLOps.** Tracking W&B live (courbes d'entraînement, pas seulement les résumés), pipeline de
   **ré-entraînement** déclenché sur dérive ou nouvelle donnée validée, versionnage des modèles et des jeux,
   déploiement canari via le job CI/CD déjà en place.
8. **Bilinguisme complet** (si requis) : grossir le triage EN à l'entraînement + ajouter une tranche d'éval
   EN indépendante (ex. `medical-triage-500`).

---

## 9. Conclusion

J'ai livré un **POC d'agent de triage médical** couvrant les cinq livrables du brief : un dataset bilingue
anonymisé et documenté (RGPD, Recital 26), un modèle Qwen3-1.7B spécialisé par SFT+LoRA (avec une tentative
de DPO analysée honnêtement), un agent complet exposé par API et prêt à être servi sur vLLM/RunPod, un pipeline
CI/CD vert, et ce rapport.

Le résultat central est un **signal de progrès net et honnête** : **macro-F1 0,19 → 0,82** entre le modèle
Base non entraîné et le SFT servi, avec un rappel de 0,90 sur les urgences vitales et un format/disclaimer
parfaits. La démarche a été **rigoureuse et transparente** : audit adverse, fuite éval→train éliminée,
intervalles de confiance reportés, un score gonflé (0,81) retiré, et un DPO documenté comme résultat négatif
instructif plutôt que survendu.

Surtout, je positionne ce système pour ce qu'il est : une **aide à la décision sous supervision humaine**,
**pas** un trieur autonome. La barre de sécurité n'est pas encore atteinte pour un usage autonome, et les
étiquettes restent un standard argent à valider cliniquement. La roadmap (§8) trace la voie du POC vers un
pilote crédible : modèle plus grand, données réelles validées par des cliniciens, intégration SIH réelle,
collecte des constantes, human-in-the-loop et MLOps. Le POC démontre la **faisabilité technique** et la
**valeur clinique potentielle** demandées — avec l'honnêteté méthodologique qu'exige un déploiement en santé.
