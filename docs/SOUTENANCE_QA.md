# Soutenance — Préparation Q&R (OC14 · agent de triage médical CHSA)

**Format : 30 min = 15 min démo + 10 min discussion/challenge (évaluateur = « Dr. Marie Dubois ») + 5 min débrief.**
Principe : je **montre**, je **positionne honnêtement**, je **connais mes chiffres**. Chaque réponse ci-dessous est
faite pour être **dite à voix haute** (2–4 phrases), pas lue.

---

## 0. Le pitch d'ouverture (30 s — à dire au démarrage)

> « J'ai construit un **POC d'agent de triage médical** : le patient décrit ses symptômes, l'agent collecte
> l'information, **anonymise (RGPD)**, évalue un **niveau d'urgence** (maximale / modérée / différée), **explique**
> sa décision, la **trace**, et en produit une **sortie au format SIH — FHIR R4 (simulé)**. Le cœur est un **Qwen3-1.7B affiné (SFT+LoRA)**
> qui passe de **macro-F1 0,19 à 0,82** sur un jeu de test tenu à l'écart. Positionnement clé : c'est une **aide à
> la décision sous supervision humaine (HITL)**, **pas** un trieur autonome — la barre de sécurité pour l'autonomie
> n'est pas atteinte, et je l'assume. »

Cette phrase répond d'avance à 3 attentes de l'évaluateur : *agent complet*, *RGPD*, *positionnement sécurité*.

---

## 1. Déroulé de la démo (15 min)

**Avant de commencer :** ouvrir 2 onglets — la **démo HF Space** (`ghislaindelabie-oc14-triage-demo.hf.space`) et
le **rapport PDF**. **Pré-chauffer** l'endpoint : lancer 1 triage ~1 min avant (sinon 1ʳᵉ requête = message « le
modèle démarre, réessayez »). Si possible, **partager l'écran local** (plus fluide que le relais).

**Scénario (ordre conseillé) :**
1. **Cas red-flag** — « douleur thoracique qui irradie dans le bras gauche, sueurs ». Montrer : le questionnaire
   pose une **question de suivi ciblée** (irradiation/sueurs), puis verdict **urgence maximale** + justification +
   recommandation + disclaimer. **Point à dire** : « le *safety override* garantit qu'un signe d'alerte ne peut
   jamais être sous-trié, même si le modèle hésite ».
2. **Anonymisation en direct** — taper un nom/téléphone dans le motif (« Jean Dupont, 06… »). Montrer dans le
   **dossier de traçabilité** que c'est stocké **`[NOM]` / `[TEL]`** + un **hash SHA-256** du texte brut. **Point** :
   « frontière RGPD — le texte brut est effacé, seul le texte anonymisé (contenu clinique lisible) et un hash
   circulent ».
3. **Cas bénin** — « rhume léger depuis hier » → **différée**, avec la **confiance** affichée. Montrer la
   **saisie optionnelle de constantes** (SpO₂, TA…) injectée dans le raisonnement.
4. **Traçabilité** — ouvrir l'accordéon : le **dossier JSON** (req-id, timestamp, version modèle, niveau,
   confiance, **hash** du brut) = le **dossier de traçabilité (SQLite, anonymisé)** → auditabilité + « zéro PII
   stocké ». **À dire** : l'agent produit **en plus** une **sortie au format SIH FHIR R4** (Encounter+Observation),
   un **mock** du format qu'un vrai SIH ingérerait — **renvoyé par l'API, pas poussé** (aucune connexion hôpital).
5. **(si le temps)** le **rapport §5** (le tableau 0,19 → 0,82) + le **dashboard des bras** (base / v9 / DPO).

**Bonus « multi-tours » (une fois PR #6 fusionnée + Space redéployé) :** après un verdict **bénin**, taper un
**complément** — « en fait la douleur irradie maintenant à la mâchoire » → l'avis **se met à jour en *maximale***
(ré-escalade sur le complément). Puis « Rafraîchir le dossier » → l'**archive** montre les **deux tours**. Un seul
geste qui illustre le **multi-tours + le garde-fou red-flag + la traçabilité qui évolue**.

**Filet de sécurité démo :** si l'endpoint réel patine, dire « l'endpoint est *serverless* (scale-to-zero) — voici
le message honnête qu'il renvoie s'il démarre » — c'est **une fonctionnalité, pas un bug** (voir §F).

---

## 1bis. Tour de la codebase — « montrez-moi le code »

*Si on me demande de présenter le code : voici la carte + un mini-parcours de 5 arrêts. Le repo est un **package
Python** `src/oc14_triage/` organisé **par étape du pipeline**, avec des **tests TDD (139)**, les **notebooks
Kaggle** (le GPU), les scripts et le déploiement.*

### Carte à vol d'oiseau

| Dossier | Rôle |
|---|---|
| `src/oc14_triage/` | Le package — tout le code applicatif (détail ci-dessous) |
| `tests/` | **139 tests TDD** (17 fichiers) — LLM/backend mockés |
| `notebooks/` | Kaggle : SFT LoRA, merge, DPO, evals — **c'est là que vit le GPU** |
| `scripts/` | Construction datasets (`build_sft_v10`, `build_dpo_pairs`), log W&B, `runpod_experiments` |
| `data/` | Jeux versionnés (`kaggle_upload/*.jsonl`) + `cards/` (DATA_CARD, audit Presidio) |
| `serving/` | Wrapper FastAPI `/triage` devant vLLM |
| `deploy/hf-space/` | Source du Space + **auto-déploiement** (workflow CI) |
| `docs/` | Le rapport (`RAPPORT_FR.md`), ce doc |
| `.github/workflows/` | CI (ruff + pytest) + Deploy HF Space |

### Le package `src/oc14_triage/` — par étape du pipeline

| Étape | Fichiers | Rôle |
|---|---|---|
| **1. Données** | `data/{sources,download,build_sft,build_dpo,vignettes,templates,card}.py` | Sources, construction SFT/DPO, vignettes, DATA_CARD |
| **2. Étiquetage** | `labeling/{clients,rubric,cases,aggregate,run}.py` | **Consensus 3-LLM** : 3 clients, grille ESI/MTS, agrégation + Fleiss κ |
| **3. Anonymisation** | `anonymization.py` | **Frontière RGPD** : Presidio + spaCy bilingue + NIR + hash |
| **4. Config** | `config.py` | `SYSTEM_PROMPT`, `URGENCY_LEVELS`, chemins |
| **5. L'agent** | `agent/` (détail ↓) | La chaîne LangGraph du parcours patient |
| **6. Éval** | `eval/metrics.py` | `extract_urgency`, macro-F1, IC Wilson, matrice de confusion |
| **7. Serving** | `serving/app.py` | Wrapper `/triage` (inject prompt, stop, API-key, audit privacy-safe) |
| **8. RunPod** (PR #8) | `runpod/{launcher,sweep,results}.py` | Pods GPU mutualisés, sweep W&B, invariant d'équilibre des paires |

**Le sous-package `agent/` (le cœur) :**

| Fichier | Rôle |
|---|---|
| `graph.py` | **La chaîne LangGraph** — 6 nœuds + l'unique override sécurité. *Le fichier central.* |
| `state.py` | `TriageCase` (l'état) + le vocabulaire `RED_FLAGS` |
| `questionnaire.py` | Collecte adaptative (motif→début→intensité + suivi red-flag) + `detect_red_flags` |
| `backend.py` | Appel LLM — **stub** (CI/démo) \| **vLLM** réel ; parsing ; sentinelle `_UNAVAILABLE` |
| `sih.py` | Mock **FHIR R4** (Encounter + Observation) |
| `store.py` | **SQLite** — dossier de traçabilité, `input_sha256`, droit à l'effacement |
| `service.py` | **API FastAPI** — `/session/*`, `/trace` |
| `ui.py` | **UI Gradio** patient |

### Mini-parcours en direct (~4 min, 5 arrêts)

1. **`agent/graph.py`** → *« La chaîne : 6 nœuds, l'état `TriageCase` circule ; `anonymisation` efface le brut
   (frontière RGPD) ; l'unique conditionnelle, c'est l'override sécurité : un red-flag force *maximale*. »*
   Montrer `build_graph()` (nœuds + edges) et `_explication()` (override + confiance dérivée).
2. **`anonymization.py`** → *« La frontière RGPD : Presidio + spaCy masquent la PII, on hashe, on efface le
   brut — seul l'anonymisé circule. »* Montrer `anonymize()`.
3. **`agent/backend.py`** → *« Backend agnostique : `stub` sans GPU (CI/démo), `vLLM` réel ; toute erreur →
   sentinelle `_UNAVAILABLE`, jamais un verdict fabriqué. »*
4. **`eval/metrics.py`** → *« L'éval : `extract_urgency` ancré sur la ligne "Niveau d'urgence :", macro-F1 + IC
   de Wilson — pas l'accuracy gameable. »*
5. **`tests/test_graph.py`** → *« Tout est TDD : 139 tests, le LLM est mocké — j'y teste l'override,
   l'anonymisation, le questionnaire. »*

**Bonus DPO/sweep** (si la discussion y va) : `runpod/results.py` → l'invariant `check_balance` (chaque classe
choisie ET rejetée = le garde-fou anti-effondrement, en test) ; `runpod/launcher.py` → un pod GPU en ~1 appel
API, cost-cap + terminate.

### « Où est X ? » — réponses éclair

- **Prompt système ?** `config.py` (`SYSTEM_PROMPT`). · **Override sécurité ?** `agent/graph.py` → `_explication`.
- **Consensus 3-LLM ?** `labeling/` (`run.py`, `aggregate.py` = κ). · **Dossier / traçabilité ?** `agent/store.py` + `service.py` `/trace`.
- **Métriques ?** `eval/metrics.py`. · **DPO / `rpo_alpha` ?** `notebooks/oc14-dpo` + `runpod/` + `scripts/build_dpo_pairs.py`.
- **Config LoRA (r=16, α=16, 4-bit) ?** `notebooks/oc14-sft-lora`.

---

## 2. Questions anticipées & réponses

### A. Données & RGPD  *(mon point le plus challengé — à maîtriser)*

**Q : D'où viennent vos données ? Ce sont de vraies données patients ?**
> Non — et c'est **volontaire**. Trois briques : (1) **Q&A médical réel** repris tel quel de sources ouvertes
> (MediQAl = questions d'examen ECN/pharmacie FR ; MedQuAD = patient-education NIH EN) ; (2) le **signal de
> triage** = **vrais cas cliniques MediQAl** ré-étiquetés par un **consensus de 3 LLM** ; (3) une petite série de
> **vignettes rédigées à la main** (sécurité + anglais). Aucun vrai patient → **non-personnel par construction**.

**Q : Comment justifiez-vous la conformité RGPD ?**
> Trois niveaux. (1) **Recital 26** : les sources sont des questions d'examen + du synthétique → pas de donnée
> personnelle. (2) Je le **prouve** : passe **Presidio** sur les 6 695 textes, avec un audit par enregistrement
> (SHA-256 + entités, sans stocker le texte). (3) **Au runtime**, un nœud d'anonymisation est la **frontière RGPD** :
> le texte brut est effacé, seul le texte anonymisé et un hash de traçabilité sont persistés.

**Q : Presidio détecte des milliers de PERSON/LOCATION/AGE — donc il y a des PII ?**
> Ce sont **massivement des faux positifs cliniques** : LOCATION = localisations anatomiques, PERSON = éponymes
> de maladies + personas d'examen (« Monsieur X »), AGE = âges d'énoncé. La passe ne *nettoie* pas des fuites,
> elle **confirme** l'absence de PII réelle et **documente** le processus.

**Q : Vos étiquettes sont fiables ? Un LLM qui étiquette pour un autre LLM ?**
> C'est un **standard « argent »**, pas de l'or — je l'assume. Je le rends **robuste** : **consensus de 3 modèles**
> (GPT-5.4 + Mistral-Medium-3.5 + Sonnet-4.6), accord inter-annotateur **Fleiss κ ≈ 0,67** (substantiel), et
> l'**éval-gold** ne garde que les cas **unanimes** de haute confiance. La roadmap : faire **valider les étiquettes
> par des cliniciens** — le consensus LLM devient alors une pré-annotation.

**Q : Presidio et spaCy, c'est quoi la différence ?**
> **spaCy** = le modèle de **reconnaissance d'entités** (le linguiste : repère noms/lieux). **Presidio** = le
> **cadre d'orchestration** (Microsoft) qui combine spaCy + ses reconnaisseurs à motifs (regex téléphone/e-mail/NIR),
> score et pilote le masquage. J'ajoute un **filet regex inconditionnel** pour les identifiants structurés, car le
> reconnaisseur de téléphone FR de spaCy est peu fiable.

### B. Entraînement & DPO  *(mon point difficile #2)*

**Q : Pourquoi Qwen3-1.7B-**Base** et pas Instruct ?**
> Le brief le nomme. Base = **ardoise vierge** (pas de refus/persona pré-câblés à combattre), mais il **ne fournit
> pas de chat template** et son `eos` est `<|endoftext|>` — j'ai dû **imposer ChatML**, **arrêter sur `<|im_end|>`**
> et servir avec le **prompt système complet**, sinon il **dégénère** (répétitions). C'est exactement la taxe qu'on
> paie à choisir Base — et je sers le modèle **exactement** comme je l'ai entraîné.

**Q : Expliquez SFT + LoRA.**
> **SFT** = montrer de bons exemples (instruction → réponse) pour apprendre le format de triage. **LoRA** =
> n'entraîner qu'un **petit greffon** (~0,3 % des poids) → ça tient sur un **T4 gratuit** en ~80 min. Config :
> r=16, α=16, 4-bit, 2 époques, seed **3407**.

**Q : Le DPO a échoué ? Pourquoi le montrer ?**
> Au **premier essai**, oui — et je le présente comme un **négatif instructif**. Sur des paires de préférence de
> triage, le DPO a **aiguisé les extrêmes** mais **effondré la classe médiane** (*modérée* rappel 0,85 → 0,55),
> macro-F1 **0,822 → 0,799**. Cause **vérifiable** : *modérée* était le côté « rejeté » des **deux** directions
> d'erreur → **déplacement de vraisemblance** (*likelihood displacement*, Razin 2024). **Mais je l'ai corrigé** :
> paires **rééquilibrées** (chaque niveau choisi ET rejeté) + **`rpo_alpha`** (une ancre NLL sur la bonne réponse
> qui bloque l'effondrement). Comparaison **pommes-à-pommes** (mêmes 300 gold, même harness, adapter on/off) :
> **v9 0,827 → DPO 0,845 (+0,018), sans effondrement** (3 classes 0,82–0,88), profil **orienté sécurité**. Je
> **sers quand même le SFT v9 en V1** (gain modeste, dans le bruit à n=300), mais **le DPO est désormais démontré
> fonctionnel** — l'échec venait des **paires, pas de la méthode**. Adapter persisté sur HF, reproductible (seed 3407).

**Q : Et v10 ? Vous parlez d'un jeu amélioré.**
> v10 = un jeu de données **amélioré** (justifications de consensus réelles + rééquilibrage), **entraîné et
> évalué** sur le **même gold** : macro-F1 **0,822 = parité stricte** avec v9 (matrice identique). Donc **je n'ai
> pas redéployé** : aucun gain de métrique ne justifie de changer la production → **discipline MLOps**. Le gain
> *visé* est la **qualité des justifications** (que le macro-F1 par niveau ne mesure pas) — mais **je le présente
> comme une hypothèse, pas un fait mesuré** : je ne l'ai pas encore quantifié par un juge côte-à-côte. À parité,
> **v9 reste servi.**

**Q : Avez-vous fait une recherche d'hyperparamètres (le brief le suggère) ?**
> Oui — un **W&B Sweep** sur RunPod : *learning rate*, rang LoRA (r), *warmup*, avec **early-stop Hyperband**,
> loggé en direct. Signal net et lisible : `eval_loss` **−30 %** quand le LR monte vers 2–3e-4 et r vers 16–32.
> C'est un **grid proxy réduit** (runs courts) pour démontrer la méthode, **directement extensible** à un grid
> complet. Cela dit, j'ai **priorisé la qualité des données** d'abord (le correctif *différée* a rapporté **+0,17**
> de macro-F1, bien plus qu'un réglage fin) : v10 (parité) suggère que **les données sont le vrai plafond**, pas
> les hyperparamètres.

### C. Évaluation & métriques

**Q : Comment avez-vous évalué ?**
> Jeu de test **gold stratifié n=300** (100/100/100), **tenu à l'écart** (disjoint du train), **décodage greedy**
> (déterministe), **sans fuite**. Métrique tête d'affiche = **macro-F1** (traite les 3 classes équitablement malgré
> le déséquilibre), + **rappel par classe** + **IC de Wilson**.

**Q : macro-F1 0,82, c'est bon ?**
> C'est un **signal de progrès net** (0,19 → 0,82), et le rappel sur les **urgences vitales est 0,90 [IC 0,83–0,95]**.
> Mais je suis **honnête sur le plafond** : le plancher d'IC à 0,83 = au pire **1 urgence sur 6 manquée** →
> **inacceptable en autonomie** → d'où le **HITL**. n=300 donne des IC larges, c'est une limite assumée.

**Q : Un chiffre était plus élevé avant (0,81) — pourquoi l'avoir retiré ?**
> Parce qu'il était **gonflé** : fuite éval→train (66 lignes de QA gold aussi à l'entraînement), décodage
> échantillonné qui tombait parfois par chance, lignes bruitées. Corrigé (greedy, sans fuite), un intermédiaire
> a mesuré 0,65, exposant l'effondrement de *différée* ; le rééquilibrage l'a remonté à **0,82 reproductible**.
> **Je préfère un 0,82 défendable à un 0,81 flatteur.**

### D. Architecture de l'agent

**Q : Comment le questionnaire choisit-il ses questions ? C'est un LLM ?**
> Non — **règles déterministes** (par choix). 3 champs cœur (motif/début/intensité) + **un** suivi conditionnel :
> si le motif contient un **red-flag** (ex. douleur thoracique), on insère **une question ciblée**. Le LLM ne sert
> qu'au **diagnostic final**, pas à poser les questions — la collecte doit être **sûre et testable**. Évolution
> possible : un questionnaire **hybride** (squelette déterministe + LLM qui raffine les questions) — un projet à part.

**Q : « Agent » — vous utilisez un framework ?**
> Oui, **LangGraph** : une chaîne d'états (`StateGraph`) — anonymisation → prétraitement → triage (LLM) →
> explication → persistance → SIH. Volontairement **linéaire + une conditionnelle** (le safety override) : le
> livrable, c'est **la chaîne**, pas la profondeur du framework.

**Q : L'intégration SIH est réelle ?**
> Non, c'est un **mock FHIR R4** : je *façonne* la sortie dans le format standard (Bundle Encounter + Observation)
> qu'un vrai SIH ingérerait — mais **sans connexion hospitalière**. Deux frontières distinctes : pas de **données**
> hôpital (RGPD) **et** pas de **connexion** SIH (mock). La roadmap = FHIR/HL7 authentifié.

**Q : La traçabilité ?**
> Chaque interaction = une ligne SQLite : req-id, timestamp, version modèle, niveau, confiance, latence, **hash du
> brut** — **aucun texte brut**, tout post-anonymisation. Auditables via l'API `/session/{id}`.

### E. Déploiement, serving, MLOps

**Q : Où et comment le modèle est-il servi ?**
> **vLLM** (API compatible OpenAI) sur **RunPod serverless** (scale-to-zero → coût quasi nul au repos), modèle
> tiré d'un **repo HF public**. Un wrapper FastAPI injecte le prompt système, force le non-thinking, arrête sur
> `<|im_end|>`. Latence **~2 s à chaud**.

**Q : CI/CD ?**
> **GitHub Actions** : à chaque push, **ruff + pytest** (139 tests) ; **auto-déploiement du Space HF** à chaque
> push sur `main` touchant le runtime (vérifié : run vert ~13 s) ; un job de déploiement d'endpoint (gated). Tout
> est **reproductible** (seed 3407, `uv.lock`, notebooks Kaggle versionnés).

### F. Sécurité clinique & positionnement  *(le cœur de la discussion clinique)*

**Q : Utiliseriez-vous ça sur de vrais patients demain ?**
> **Non, et c'est le point le plus important.** C'est une **aide à la décision, HITL** : l'agent **assiste**, le
> soignant **décide**. Un rappel de 0,90 (plancher 0,83) sur les urgences vitales n'est **pas suffisant pour
> l'autonomie**. L'agent ne pose pas de diagnostic, ne prescrit pas, et affiche toujours un **disclaimer**.

**Q : Que se passe-t-il si le modèle est indisponible / lent ?**
> Il **ne fabrique jamais un verdict**. Après ~10 s, il renvoie un **message honnête** : « le modèle démarre →
> réessayez dans ~1 min ; en attendant, ce cas relève de l'évaluation d'un clinicien », avec `needs_review`. **Mais**
> un red-flag détecté **escalade quand même** en *maximale* (la règle de sécurité prime sur la panne).

**Q : Comment passer de 0,82 à un usage crédible (98 %+) ?**
> Trois leviers combinés, tous **HITL** : (1) **données réelles validées par des cliniciens** (le plus fort) ;
> (2) **modèle plus grand** (32B+, prévu au brief) ; (3) **boucle LLMOps** — chaque correction clinicienne devient
> une donnée d'entraînement → ré-entraînement. La précision *effective du système* est portée par le clinicien,
> et ses corrections rapprochent progressivement l'agent seul du plafond.

### G. Limites & roadmap  *(les dire AVANT qu'on me les sorte)*

Limites assumées : étiquettes **argent** (non validées cliniquement) · **n=300** (IC larges) · **synthétique** (pas
la vraie distribution des présentations) · SIH **mock** · pas de **recherche d'hyperparamètres** systématique ·
endpoint **serverless** (cold-start). Roadmap : données réelles + validation clinicienne · modèle 32B+ · SIH
FHIR/HL7 réel · constantes par intégration dispositif · **HITL + monitoring de dérive** · W&B Sweep · MLOps.

---

## 2bis. Précisions techniques — les questions pointues (à garder sous la main)

**Q : La config LoRA `r=16, α=16, 4-bit` ?**
> LoRA = on **gèle** le modèle et on entraîne un petit **greffon** (~0,3 % des poids). **r=16** = la
> **capacité/rang** du greffon ; **α=16** = son **échelle** d'application (effet = α/r = 1) ; **4-bit** = le modèle
> gelé est **quantisé** → tient sur un **T4 gratuit**. Full fine-tuning → gros GPU ; SFT+LoRA 4-bit → ~80 min,
> ~0 €. *(2 époques = 2 passages sur les données ; seed 3407 = reproductibilité.)*

**Q : Comment Presidio est-il configuré (quel modèle NLP) ?**
> **Pas du défaut.** Moteur spaCy **bilingue** (`fr_core_news_md` + `en_core_web_sm`), reconnaisseurs prédéfinis
> + un **NIR français** custom, **seuil relevé à 0,6**, une **allowlist** de noms communs FR (couper les faux
> positifs médicaux), et un **filet regex inconditionnel** pour les identifiants structurés (le reconnaisseur
> téléphone FR de spaCy est faible ~0,4). Nuance : `md` est un modèle *medium* — suffisant pour **prouver
> l'absence** de PII sur des données déjà non-personnelles ; pour du réel → transformer + revue clinicienne.
> *(Preuve dans le rapport §2.5 + `data/cards/anonymization_audit.json` : 6 695 textes audités.)*

**Q : À quoi sert le hash SHA-256 ? Pourquoi pas juste un UUID ?**
> Le SHA-256 est une **empreinte du contenu** du texte brut (l'UUID `interaction_id`, lui, est **aléatoire** et
> n'identifie que l'enregistrement). On garde l'empreinte **au lieu** du brut → *« hash pour la traçabilité,
> anonymisation pour le stockage »* (minimisation RGPD).
> **Ce qui est réellement exploitable :** (1) **déduplication** (même saisie → même hash) ; (2) **intégrité** —
> prouver qu'un texte présenté est **octet-pour-octet** celui traité.
> **Ne pas survendre (piège) :** le hash exige une **correspondance exacte** — un **souvenir** ou un **dossier
> papier re-saisi ne matcheront pas**. La vérification d'un texte contesté n'est possible que si le brut exact
> **existe ailleurs** (DPI/front-end qui l'a capté), **ou** en **reconstituant** le brut depuis notre **texte
> anonymisé** (squelette gardé octet-pour-octet) : on **re-remplit les placeholders** `[NOM]`/`[TEL]` avec les
> vraies PII, puis re-hachage → comparaison — **sans que nous ayons stocké la PII**. C'est surtout un **contrôle
> d'intégrité**, pas un moyen de « retrouver » le texte. *(Nuance crypto : texte court → attaque par dictionnaire
> possible.)*

**Q : L'override red-flag — c'est un hybride LLM + règles ?**
> Oui — un **garde-fou neuro-symbolique** : une **règle déterministe** pose un **plancher de sécurité** que le LLM
> ne peut pas passer sous silence (red-flag → au moins *maximale*). Le LLM gère le **nuancé**, la règle le
> **non-négociable**. Extensible : **seuils de constantes** (SpO₂ < 90…), détection red-flag **negation-aware**,
> règles **ESI/MTS**. **Insight fort** : pour la sécurité, **les garde-fous déterministes sont plus fiables que
> l'alignement par les poids** — c'est pourquoi l'override est une **règle**, pas dans le modèle (le DPO censé
> forcer ça a écrasé la classe médiane, cf. §B).

**Q : Le chatbot gère-t-il une conversation multi-tours ?**
> Oui (v. dossier / PR #6). Après le premier verdict, on peut **ajouter des informations** ; l'agent
> **ré-évalue sur toute la conversation** (contexte accumulé) et renvoie un **verdict mis à jour**. Un red-flag
> apparu dans un complément **ré-escalade** (ex. « la douleur irradie maintenant à la mâchoire » → *maximale*).
> Chaque tour est **tracé** → le dossier montre l'évaluation qui **évolue**. « Nouvelle consultation » = nouveau
> patient (nouvelle session).

**Q : Où voir le dossier de traçabilité (démo) ?**
> Déplier l'accordéon **« Traçabilité — dossier (anonymisé) »** → bouton **« Rafraîchir le dossier »** →
> l'**archive globale** (tous les cas/tours). Se remplit **après un verdict**. *(API : `GET /trace`, `/sessions`,
> `/session/{id}`.)*

**Q : « pousse au SIH » ? / l'intégration SIH est réelle ?**
> Non — l'agent **produit une sortie au format SIH FHIR R4** (Encounter + Observation), un **mock** du format
> qu'un vrai SIH ingérerait, **renvoyé par l'API — pas poussé**, aucune connexion hôpital. Deux frontières
> distinctes : pas de **données** patient (RGPD) **et** pas de **connexion** SIH (mock).

---

## 3. Questions dures / pièges — et comment les retourner

| Le piège | Ma réponse (posture : honnêteté factuelle) |
|---|---|
| « Votre modèle est nul, il rate 1 urgence sur 6 » | « Exact au pire cas de l'IC — c'est **pourquoi** je le positionne en HITL, pas en autonomie. Le POC démontre la *méthode* et un *signal*, pas un produit. » |
| « Vos données sont fausses/synthétiques » | « Les **cas** sont réels (MediQAl), seules les **étiquettes** sont générées par consensus. Le synthétique, je l'assume et je le documente ; la roadmap = validation clinicienne. » |
| « Le DPO a échoué, donc vous ne maîtrisez pas » | « J'ai **diagnostiqué** l'échec (déplacement de vraisemblance) **puis corrigé** : paires rééquilibrées + `rpo_alpha` → **+0,018 sans effondrement**. Diagnostiquer, corriger, et savoir que le gain reste modeste : c'est de la rigueur. » |
| « v10 = même score, à quoi bon ? » | « Justement : **pas de gain → pas de redéploiement**. v10 améliore les *justifications*, pas la F1 par niveau. » |
| « Pourquoi pas de vraie intégration SIH / vrai endpoint clinique ? » | « Périmètre POC en 4 semaines. Je **façonne** le format FHIR réel (démontrable) ; la connexion authentifiée est la 1ʳᵉ étape roadmap. » |
| « Un LLM qui étiquette un LLM, c'est circulaire » | « D'où le **consensus 3 modèles** + κ=0,67 + gold unanime. Et la vérité terrain reste la **validation clinicienne** (roadmap). » |

**Règle d'or :** ne jamais survendre. Chaque fois que je **reconnais une limite avant qu'on me la sorte**, je gagne
en crédibilité. L'honnêteté méthodologique **est** l'argument.

---

## 4. Chiffres & faits à mémoriser

- **macro-F1 : 0,19 (Base) → 0,82 (SFT v9 servi)**. DPO corrigé (`rpo_alpha`) : **0,845** vs v9 0,827 (pommes-à-pommes, **+0,018**, sans effondrement — v9 servi en V1). v10 : 0,822 (parité).
- **Rappel urgence maximale : 0,90 [IC 95 % 0,83–0,95]**. Rappels v9 : max 0,90 / modérée 0,85 / différée 0,71. **κ = 0,73**.
- **Éval : n=300** stratifié (100/100/100), greedy, sans fuite. Disclaimer/format : **1,00**.
- **Données : ~5 600 paires** SFT (~82 % FR). Étiquetage **3-LLM**, **Fleiss κ ≈ 0,67**. Presidio sur **6 695** textes.
- **Modèle : Qwen3-1.7B-Base**, SFT+LoRA (r=16, α=16, 4-bit, 2 époques, **seed 3407**), Unsloth/T4, ~80 min, ~0 €.
- **3 niveaux** : urgence **maximale / modérée / différée** (échelle FR, dérivée d'ESI/MTS).
- **Serving** : vLLM / RunPod serverless / HF public ; latence **~2 s à chaud**. **139 tests** verts.

---

## 5. À NE PAS faire

- ❌ Prétendre que c'est déployable en autonomie / « ça marche à 82 % donc c'est bon ».
- ❌ Cacher le DPO, la parité v10, ou le 0,81 retiré — au contraire, **les mettre en avant** comme preuves de rigueur.
- ❌ Confondre « pas de données patients » (RGPD) et « pas de connexion SIH » (mock) — ce sont **deux** frontières.
- ❌ Dire « le questionnaire est intelligent grâce au LLM » — non, il est **déterministe** (le LLM = le diagnostic).
- ❌ Rester silencieux sur une limite — **la nommer en premier**.

**Mantra de clôture :** *« Un POC qui démontre une méthode rigoureuse et un signal honnête, positionné en aide à la
décision humaine — avec une roadmap crédible vers un pilote clinique. »*
