# OC14 — Note de synthèse pour le mentor

> POC d'assistant de **triage médical** bilingue (FR/EN) pour un hôpital fictif (CHSA), par
> spécialisation de **Qwen3-1.7B** (SFT+LoRA → DPO), servi via vLLM, avec CI/CD et un rapport ≤20 p.
> Cette note résume le projet, la méthode, les résultats préliminaires (honnêtes), ce qu'il reste à
> finaliser, et pourquoi le travail est prêt à être présenté. Détails techniques : voir le rapport
> complet et `REPORT_LIMITATIONS` / `DEVELOPMENT_JOURNAL`.

## 1. Le projet & où nous en sommes
La tâche centrale est le **triage** (et non le Q&A médical) : à partir d'une présentation clinique,
classer l'urgence sur **3 niveaux** (maximale / modérée / différée). État d'avancement :
- **Données** collectées (MediQAl FR, MedQuAD EN, UltraMedical) ; anonymisation RGPD (Presidio) prévue.
- **Jeu d'évaluation rigoureux construit** : 3 075 vignettes cliniques réelles labellisées par
  **consensus de 3 LLM de pointe**, gold stratifié 100/100/100.
- **SFT (LoRA) entraîné et évalué** sur Kaggle (T4 gratuit) ; **DPO** diagnostiqué (échec initial
  compris, seconde tentative ciblée en cours).
- **Audit adverse** de tout le pipeline réalisé (16 défauts corrigés) → un **résultat honnête** établi.

## 2. Méthodologie
- **Labellisation par consensus** : GPT-5.4 + Mistral-Medium-3.5 + Claude-Sonnet-4.6 appliquent une
  rubrique **citée** (ESI / MTS / FRENCH / CIMU) et renvoient le niveau 3-classes **+ l'ESI 1-5** dans
  le même appel. On garde l'**unanime** comme *gold*. Accord inter-modèles **Fleiss κ ≈ 0,67**
  (« substantiel »). C'est un **standard argent** (pas de clinicien) — assumé.
- **Entraînement** : Qwen3-1.7B-**Base**, SFT+LoRA (Unsloth), 2 époques, T4 gratuit (~1 h30/run).
- **Évaluation** : gold **stratifié** (équilibré par classe), décodage **greedy** (déterministe,
  reproductible), **macro-F1** + rappel/précision par classe + **IC de Wilson** + matrice de confusion.
  **Baseline** (Base non entraîné) pour mesurer le gain réel du fine-tuning.

## 3. Résultats préliminaires (honnêtes)
**macro-F1 = 0,65** (sans fuite, reproductible). Comportement de format parfait
(disclaimer / structure / pas de `<think>` = 1,00).

| | gold → prédit | rappel (IC 95 %) |
|---|---|---|
| urgence maximale | 91 ✓ · 9→modérée · **0→différée** | **0,91** [0,84–0,95] |
| urgence modérée | 85 ✓ · 15→maximale · **0→différée** | 0,85 [0,77–0,91] |
| urgence différée | 28 ✓ · **72→modérée** · 0→maximale | 0,28 [0,20–0,38] |

**Lecture clé** : le modèle ne **sous-trie jamais** (0 urgence → différée) — c'est le biais
**sécuritaire voulu** — mais **sur-trie** le bas du spectre (rappel *différée* 0,28). Correction en
cours (rééquilibrage des données *différée* + DPO ciblé). Un score antérieur de 0,81 était **gonflé**
(fuite éval→train + décodage échantillonné) et a été **retiré** — la rigueur de l'audit est un point
fort, pas une faiblesse. *(Baseline Base nu : à insérer — montre le gain net du fine-tuning.)*

## 4. Ce qu'il reste à finaliser
1. **Corriger le sur-triage** : rééquilibrer le signal *différée* (en cours) + **DPO** ciblé
   (pénaliser sous- ET sur-triage, pondéré par le coût) ; ré-évaluer.
2. **Serving** : vLLM (RunPod ou Modal) + wrapper FastAPI (+ injection du prompt système) ; **étape de
   déploiement CI**.
3. **RGPD** : passe Presidio + journal d'audit + carte de données.
4. **Rapport ≤20 pages** (déjà tenu comme journal de bord).
5. *(optionnel)* éval **indépendante** sur `medical-triage-500` + tranche d'évaluation **EN**.

## 5. Pourquoi c'est prêt à être présenté
- **Les 5 livrables** (dataset RGPD, poids fine-tunés, endpoint cloud, CI/CD, rapport) sont couverts
  ou en voie de l'être.
- La **méthodologie est défendable et honnête** : audit adverse, fuites éliminées, IC reportés,
  limites explicites, résultat **non survendu**.
- Le POC démontre la **méthode** (spécialiser un petit LLM bilingue au triage + une évaluation
  honnête) et un **signal de progrès** ; il est positionné comme **aide à la décision /
  human-in-the-loop**, **pas** comme un trieur autonome.

## Limites assumées (résumé — détail dans `REPORT_LIMITATIONS`)
Standard argent (pas de clinicien) ; **circularité** (le gold = consensus 3-LLM unanime, donc on
mesure la *fidélité d'imitation* des modèles-professeurs sur le sous-ensemble *facile*, pas la
justesse clinique) ; corpus de **vignettes d'examen** (sur-représente le grave : ~47 % vs ~25-30 %
en vraies urgences) ; **FR-primaire** (triage EN mince, éval 100 % FR) ; **n=100/classe** (IC larges,
plancher de sécurité = la borne basse 0,84). **Barre de sécurité** : 0,91 de rappel sur les urgences
vitales reste **insuffisant pour un triage autonome** (≥1/10 manquée à la borne basse) → aide à la
décision sous supervision humaine.
