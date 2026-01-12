# TP : Optimisation des Hyperparamètres

Installations nécessaires : 
```bash
pip install scikit-learn optuna
```

Utiliser le dataset `load_wine()` de scikit-learn pour tous les exercices (178 échantillons, 13 features, 3 classes de vins).

---

## Exercice 1 : Implémenter Grid Search

### 1.1 - Consigne

Implémenter Grid Search **from scratch**.

### 1.2 - Principe de l'algorithme

Grid Search est une méthode **exhaustive** qui teste toutes les combinaisons possibles :

1. **Définir** une liste de valeurs pour chaque hyperparamètre
2. **Générer** toutes les combinaisons (produit cartésien)
3. **Pour chaque** combinaison :
   - Créer un modèle avec ces hyperparamètres
   - Évaluer sa performance
   - Enregistrer le score
4. **Retourner** la combinaison avec le meilleur score

```
Exemple : max_depth=[2,5] et min_samples_split=[2,10]
→ Combinaisons : (2,2), (2,10), (5,2), (5,10)
→ 4 modèles à évaluer
```

### 1.3 - Questions

1. Créer une grille d'hyperparamètres pour `DecisionTreeClassifier` :
   - `max_depth` : `[2, 5, 10, 15, 20]`
   - `min_samples_split` : `[2, 5, 10]`

2. Utiliser `itertools.product` pour générer toutes les combinaisons.

3. Pour chaque combinaison, calculer le score avec `cross_val_score` (5-fold).

4. Retourner les meilleurs hyperparamètres.

5. Combien de combinaisons ont été testées ?
6. Combien d'entraînements au total (combinaisons × folds) ?

---

## Exercice 2 : Utiliser GridSearchCV

### 2.1 - Consigne

Utiliser `GridSearchCV` de scikit-learn pour reproduire l'exercice 1.

### 2.2 - Questions

1. Créer un `DecisionTreeClassifier`.
2. Définir la grille d'hyperparamètres.
3. Utiliser `GridSearchCV` avec `cv=5` et `scoring='accuracy'`.
4. Afficher `best_params_` et `best_score_`.

---

## Exercice 3 : Implémenter Random Search

### 3.1 - Consigne

Implémenter Random Search **from scratch**.

### 3.2 - Principe de l'algorithme

Random Search **échantillonne aléatoirement** au lieu de tester toutes les combinaisons :

1. **Définir** des distributions (plages de valeurs) pour chaque hyperparamètre
2. **Fixer** un budget : nombre d'essais `n_iter`
3. **Répéter** `n_iter` fois :
   - Tirer aléatoirement une valeur pour chaque hyperparamètre
   - Créer un modèle avec ces hyperparamètres
   - Évaluer sa performance
   - Enregistrer le score
4. **Retourner** la combinaison avec le meilleur score

```
Exemple avec n_iter=3 :
→ Essai 1 : max_depth=7, min_samples_split=12
→ Essai 2 : max_depth=23, min_samples_split=5
→ Essai 3 : max_depth=11, min_samples_split=18
```

**Avantage** : On contrôle le nombre d'évaluations (contrairement à Grid Search).

### 3.3 - Questions

1. Définir des distributions pour :
   - `max_depth` : entier entre 2 et 30
   - `min_samples_split` : entier entre 2 et 20

2. Échantillonner 20 combinaisons aléatoires.

3. Évaluer chaque combinaison par validation croisée.

4. Comparer le meilleur score avec Grid Search (15 vs 20 évaluations).

5. Exécuter plusieurs fois avec différents seeds. Le résultat est-il stable ?

---

## Exercice 4 : Utiliser RandomizedSearchCV

### 4.1 - Consigne

Utiliser `RandomizedSearchCV` de scikit-learn pour reproduire l'exercice 3.

### 4.2 - Questions

1. Définir les distributions pour `max_depth` et `min_samples_split`.
2. Utiliser `RandomizedSearchCV` avec `n_iter=20`.
3. Comparer le temps d'exécution avec `GridSearchCV`.

---

## Exercice 5 : Optimisation Bayésienne avec Optuna

### 5.1 - Consigne

Utiliser Optuna pour optimiser les hyperparamètres d'un `DecisionTreeClassifier` avec l'algorithme TPE.

### 5.2 - Principe

Optuna utilise TPE (Tree-structured Parzen Estimator) pour choisir **intelligemment** les prochains hyperparamètres à tester, en se basant sur les résultats précédents.

### 5.3 - Documentation

Suivre l'exemple de la documentation officielle :
- **Quickstart** : https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/001_first.html

### 5.4 - Questions

1. Installer Optuna avec `pip install optuna`.
2. Créer une fonction `objective(trial)` qui optimise `max_depth`, `min_samples_split` et `min_samples_leaf`.
3. Utiliser `TPESampler` et lancer 30 essais.
4. Afficher les meilleurs paramètres et le meilleur score.
5. Quel est l'avantage par rapport à Random Search ?

---


