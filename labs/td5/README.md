# TP : Arbre de décision

## Dataset

Nous utilisons le dataset **Heart Disease** de l'UCI Repository.

**Objectif** : Prédire la présence de maladies cardiaques chez un patient.

| Feature | Description |
|---------|-------------|
| age | Âge du patient |
| sex | Sexe (1 = homme, 0 = femme) |
| cp | Type de douleur thoracique (0-3) |
| trestbps | Pression artérielle au repos |
| chol | Cholestérol sérique |
| fbs | Glycémie à jeun > 120 mg/dl |
| restecg | Résultats ECG au repos |
| thalach | Fréquence cardiaque maximale |
| exang | Angine induite par l'exercice |
| oldpeak | Dépression ST induite par l'exercice |
| slope | Pente du segment ST |
| ca | Nombre de vaisseaux colorés |
| thal | Thalassémie |
| **target** | Présence de maladie cardiaque (0/1) |

---

### Partie 1 : Arbre de Décision

**Objectif** : Implémenter un arbre de décision **from scratch** et le visualiser.

#### Rappel : Comment construire un arbre de décision ?

```
ALGORITHME : BuildTree(data, depth)
─────────────────────────────────────────────────────────
ENTRÉE : data (échantillons), depth (profondeur actuelle)
SORTIE : un nœud de l'arbre

1. SI critère d'arrêt atteint (depth=max OU data pur OU trop peu d'échantillons)
   └─→ RETOURNER une feuille avec la classe majoritaire

2. POUR chaque feature f :
   └─→ POUR chaque seuil t possible (valeurs uniques de f) :
       └─→ Séparer data en : gauche = {x | x[f] ≤ t}
                             droite = {x | x[f] > t}
       └─→ Calculer le GAIN d'information (ou réduction de Gini)

3. Sélectionner (best_feature, best_threshold) qui maximise le gain

4. Créer un nœud avec :
   ├─→ feature = best_feature
   ├─→ threshold = best_threshold
   ├─→ left = BuildTree(data_gauche, depth + 1)
   └─→ right = BuildTree(data_droite, depth + 1)

5. RETOURNER le nœud
─────────────────────────────────────────────────────────
```

#### Formules importantes

**Entropie** (mesure l'impureté d'un ensemble) :
$$H(S) = -\sum_{c \in classes} p_c \log_2(p_c)$$

**Indice de Gini** (alternative à l'entropie) :
$$Gini(S) = 1 - \sum_{c \in classes} p_c^2$$

**Gain d'information** :
$$Gain = H(parent) - \frac{|gauche|}{|parent|} H(gauche) - \frac{|droite|}{|parent|} H(droite)$$

#### Tâches

1. Implémenter la classe `DecisionTreeNode` (nœud ou feuille)
2. Implémenter le calcul de l'**entropie** ou du **Gini**
3. Implémenter la fonction `find_best_split()` (meilleure séparation)
4. Implémenter `build_tree()` récursivement
5. Implémenter `predict()` pour faire des prédictions
6. **Visualiser** l'arbre avec une fonction récursive
7. Comparer avec `sklearn.tree.DecisionTreeClassifier`

