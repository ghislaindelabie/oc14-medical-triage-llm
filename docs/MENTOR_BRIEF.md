# OC14 — Note de synthèse pour le mentor

> POC d'assistant de **triage médical** bilingue (FR/EN) pour un hôpital fictif (CHSA), par
> spécialisation de **Qwen3-1.7B** (SFT+LoRA → DPO), servi via vLLM, avec CI/CD et un rapport ≤20 p.
> Cette note résume le projet, la méthode, les résultats préliminaires (honnêtes), ce qu'il reste à
> finaliser, et pourquoi le travail est prêt à être présenté. Détails techniques : voir le rapport
> complet et `REPORT_LIMITATIONS` / `DEVELOPMENT_JOURNAL`.

## 1. Le projet & où j'en suis
La tâche centrale est le **triage** (et non le Q&A médical) : à partir d'une présentation clinique,
classer l'urgence sur **3 niveaux** (maximale / modérée / différée). État d'avancement :
- J'ai **collecté les données** (MediQAl FR, MedQuAD EN, UltraMedical) ; l'anonymisation RGPD (Presidio) est prévue.
- J'ai **construit un jeu d'évaluation rigoureux** : 3 075 vignettes cliniques réelles labellisées par
  **consensus de 3 LLM de pointe**, gold stratifié 100/100/100.
- J'ai **entraîné et évalué le SFT (LoRA)** sur Kaggle (T4 gratuit) ; j'ai **diagnostiqué le DPO**
  (échec initial compris, seconde tentative ciblée en cours).
- J'ai mené un **audit adverse** de tout le pipeline (16 défauts corrigés) → un **résultat honnête** établi.

## 2. Méthodologie
- **Labellisation par consensus** : GPT-5.4 + Mistral-Medium-3.5 + Claude-Sonnet-4.6 appliquent une
  rubrique **citée** (ESI / MTS / FRENCH / CIMU) et renvoient le niveau 3-classes **+ l'ESI 1-5** dans
  le même appel. Je garde l'**unanime** comme *gold*. Accord inter-modèles **Fleiss κ ≈ 0,67**
  (« substantiel »). C'est un **standard argent** (pas de clinicien) — assumé.
- **Entraînement** : Qwen3-1.7B-**Base**, SFT+LoRA (Unsloth), 2 époques, T4 gratuit (~1 h30/run).
- **Évaluation** : gold **stratifié** (équilibré par classe), décodage **greedy** (déterministe,
  reproductible), **macro-F1** + rappel/précision par classe + **IC de Wilson** + matrice de confusion.
  **Baseline** (Base non entraîné) pour mesurer le gain réel du fine-tuning.

## 3. Résultats préliminaires (honnêtes)
**D'un modèle inutilisable à un trieur compétent.** Même gold stratifié (n=300, décodage *greedy*, sans fuite) :

| modèle | macro-F1 | rappel *maximale* | format / disclaimer |
|---|---|---|---|
| Base (non entraîné) | **0,19** | 0,70 | 0,68 / 0,00 |
| **SFT (LoRA)** | **0,82** | **0,90** [IC 0,83–0,95] | 1,00 / 1,00 |

Le fine-tuning fait passer le modèle d'**inutilisable** (le Base répond « maximale » ou échoue à produire
le format dans ~32 % des cas, ne distingue jamais les niveaux bas, n'émet jamais le *disclaimer*) à un
**trieur honnête** : macro-F1 **0,19 → 0,82**. Rappel par classe (SFT) : *maximale* 0,90 / *modérée* 0,85 /
**différée 0,71** [IC 0,62–0,79] ; Cohen κ 0,73 ; format/disclaimer parfaits (1,00).

**Transparence (la rigueur est un atout).** Un premier score de 0,81 était **gonflé** (fuite éval→train +
décodage échantillonné) → **retiré** après un **audit adverse** du pipeline ; le 0,82 actuel est
reproductible et sans fuite. **Compromis sécurité/précision assumé** : restaurer la classe *différée*
(rappel 0,28 → 0,71) a réintroduit un peu de sous-triage — **1 cas *urgence maximale* → *différée*** (une
urgence vitale rétrogradée). J'ai ensuite testé un **DPO** ciblé (préférences de triage équilibrées) : il
améliore les extrêmes (*différée* 0,71 → 0,96) mais **affaisse la classe intermédiaire** (*modérée*
0,85 → 0,55 ; macro-F1 0,80 < 0,82) → **je conserve le SFT**. Le DPO est livré comme **résultat honnête** :
la technique est démontrée et l'échec analysé (les paires de niveaux adjacents pénalisent structurellement
la classe du milieu).

## 4. Ce qu'il reste à finaliser
1. ✓ **Sur-triage corrigé** (rééquilibrage → *différée* 0,71) et **DPO testé** (résultat honnête,
   modèle SFT conservé) — fait.
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
plancher de sécurité = la borne basse 0,83). **Barre de sécurité** : 0,90 de rappel sur les urgences
vitales (borne basse 0,83) + 1 urgence rétrogradée en *différée* dans l'éval → **insuffisant pour un
triage autonome** → aide à la décision sous supervision humaine.
