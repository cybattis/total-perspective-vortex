# Explication de la Section "Score" - Total Perspective Vortex

## 📊 Critère de Notation Principal

### Exigences de Base

La section "Score" de la checklist indique :

> **There has to be a script executing training over each subject and computing the mean of scores over each subjects, by type of experiment runs.**
> 
> **The mean of the resulting four means (corresponding to the four types of experiment runs) has to be superior or equal to 75%.**
> 
> **Over 75% add a point for every 3%.**

### Traduction et Explication

**Ce qu'il faut faire :**

1. **Créer un script d'évaluation** qui :
   - Entraîne le modèle sur **chaque sujet** (subject) du dataset
   - Calcule le score moyen pour chaque sujet
   - Le fait pour **chaque type d'expérience** (experiment runs)

2. **Les 4 types d'expériences** du dataset PhysioNet Motor Imagery sont :
   - **Run 3** : Task 1 (ouvrir/fermer le poing gauche) vs Task 2 (ouvrir/fermer le poing droit)
   - **Run 7** : Task 1 (ouvrir/fermer le poing gauche) vs Task 2 (ouvrir/fermer le poing droit)
   - **Run 4** : Task 1 (ouvrir/fermer les deux poings) vs Task 3 (ouvrir/fermer les deux pieds)
   - **Run 8** : Task 1 (ouvrir/fermer les deux poings) vs Task 3 (ouvrir/fermer les deux pieds)
   
   OU selon l'interprétation :
   - **Runs 3, 7, 11** : Tâches réelles d'imagerie motrice
   - **Runs 4, 8, 12** : Tâches réelles d'imagerie motrice
   
   (Il faut vérifier exactement quels runs correspondent aux 4 types dans le sujet)

3. **Calcul du score final :**
   ```
   score_run_type_1 = moyenne des scores de tous les sujets pour le run type 1
   score_run_type_2 = moyenne des scores de tous les sujets pour le run type 2
   score_run_type_3 = moyenne des scores de tous les sujets pour le run type 3
   score_run_type_4 = moyenne des scores de tous les sujets pour le run type 4
   
   SCORE_FINAL = (score_run_type_1 + score_run_type_2 + score_run_type_3 + score_run_type_4) / 4
   ```

4. **Barème de notation :**
   - **< 75%** → **0 points** (échec)
   - **75%** → **1 points** (validation minimale)
   - **78%** → **2 point** (75% + 3%)
   - **81%** → **3 points** (75% + 6%)
   - **84%** → **4 points** (75% + 9%)
   - **87%** → **5 points** (75% + 12%)

### Note sur "Rate it from 0 (failed) through 5 (excellent)"

Cette partie est une **échelle de notation de 0 à 5** basée sur le score obtenu :
- 0 = échec (< 75%)
- 1 = passable (>=75%)
- 2 = acceptable (~78%)
- 3 = bien (~81%)
- 4 = très bien (~84%)
- 5 = excellent (~87%)

## 🎯 Ce que vous devez implémenter

### Script d'évaluation requis

Vous devez créer un script (par exemple `evaluate_all_subjects.py`) qui :

```python
# Pseudo-code
for each subject (S001 to S109):
    for each run_type in [run_type_1, run_type_2, run_type_3, run_type_4]:
        - Charger les données du sujet pour ce type de run
        - Entraîner le modèle avec cross-validation
        - Calculer le score de classification
        - Stocker le score
        
    # Calculer la moyenne pour ce sujet sur les 4 types
    subject_mean = moyenne des 4 scores
    
# Calculer la moyenne finale
for each run_type:
    run_type_mean = moyenne des scores de tous les sujets pour ce type
    
final_score = moyenne des 4 run_type_means

print(f"Score final : {final_score * 100:.2f}%")
```

## ⚠️ Points Importants

1. **Tous les sujets** : Vous devez tester sur tous les sujets disponibles (109 sujets dans le dataset PhysioNet), pas seulement S001
2. **4 types de runs** : Identifiez correctement les 4 types d'expériences dans le dataset
3. **Moyenne des moyennes** : Le score final est la moyenne des 4 moyennes (une par type de run)
4. **Seuil de 75%** : C'est le minimum pour valider le projet
5. **Script automatisé** : L'évaluation doit pouvoir être lancée automatiquement par l'évaluateur

## 📈 Votre Situation Actuelle

D'après votre code actuel :
- ✅ Vous avez un modèle qui fonctionne
- ✅ Vous obtenez 93.3% sur un seul sujet (S001)
- ❌ Vous n'avez pas de script d'évaluation sur tous les sujets
- ❌ Vous ne calculez pas la moyenne sur les 4 types de runs

## 🔧 Actions Nécessaires

1. **Créer un script d'évaluation complète** qui teste tous les sujets
2. **Identifier les 4 types de runs** dans le dataset
3. **Calculer les moyennes** selon la formule requise
4. **Afficher le score final** avec le format demandé

## 💡 Conseil

Votre score de 93.3% sur S001 est excellent ! Si vous obtenez des résultats similaires sur les autres sujets, vous devriez facilement atteindre 5/5 sur ce critère (≥90%).

