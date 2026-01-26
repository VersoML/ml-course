# TP - Partie 2 : Bagging & Random Forest

## Consigne
Comprendre l'impact du Bagging sur la variance et implémenter une version simplifiée de Random Forest.

> **Prérequis** : Charger les données (voir `README.md`).

## 1. Bagging "From Scratch"

Implémentez votre propre version du Bagging (`my_bagging_predict`) en suivant ces étapes :

1.  Générez $B$ échantillons bootstrap (tirage avec remise).
2.  Entraînez un modèle de base (ex: Decision Tree) sur chaque échantillon.
3.  Pour la prédiction, faites voter les $B$ modèles (Majorité pour la classification).

Entraînez ce modèle avec 50 arbres et comparez l'accuracy avec un arbre unique.

```python
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample

def my_bagging_predict(X_train, y_train, X_test, n_estimators=50):
    # Liste pour stocker les modèles entraînés
    models = []
    
    # Entraînement
    for i in range(n_estimators):
        # 1. Bootstrap
        # ...
        
        # 2. Train model
        # ...
        pass

    # Prédiction (Vote)
    # ...
    return y_pred
```

## 2. Bagging avec Sklearn (Vérification)

Vérifiez vos résultats en utilisant `BaggingClassifier` de `sklearn.ensemble`.

```python
from sklearn.ensemble import BaggingClassifier
# ...
```

## 3. Random Forest

Utilisez `RandomForestClassifier` de sklearn pour voir si l'ajout du "Feature Sampling" améliore encore les performances.

## 4. (Bonus) Random Forest "From Scratch"

Implémenter un Random Forest complet et tester le sur vos données.

Pour cela, nous avons besoin de deux classes :
1.  **SimpleDecisionTree** : Un arbre de décision simplifié qui supporte le *feature sampling*.
2.  **LightRandomForest** : L'ensemble qui gère le *bagging* et agrège les résultats.

### Le Code à Compléter

Voici la structure complète. Votre tâche est de comprendre ce code, de remplir les trous.

#### 1. L'Arbre de Décision (The Base Learner)

```python
import numpy as np
from collections import Counter

class SimpleDecisionTree:
    def __init__(self, max_depth=5, feature_subsample=None):
        self.max_depth = max_depth
        self.feature_subsample = feature_subsample
        self.tree = None

    def fit(self, X, y):
        self.tree = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        # Stop if pure, empty, or at max depth
        # TODO: Implémenter la condition d'arrêt
        if ... or ... or ...:
            return Counter(y).most_common(1)[0][0]

        # Feature Sampling: Pick a subset of features
        # TODO: Sélectionner aléatoirement 'self.feature_subsample' indices
        indices = np.random.choice(...)
        
        # Find best split (Simplified: random split point for speed)
        # Pour cet exercice, on choisit une feature et un seuil au hasard parmi les indices sélectionnés
        best_feat = np.random.choice(indices)
        threshold = np.mean(X[:, best_feat])
        
        left_idx = X[:, best_feat] <= threshold
        right_idx = ~left_idx
        
        # If split is useless, return leaf
        if not any(left_idx) or not any(right_idx):
            return Counter(y).most_common(1)[0][0]

        return {
            'feat': best_feat,
            'threshold': threshold,
            'left': self._grow_tree(X[left_idx], y[left_idx], depth + 1),
            'right': self._grow_tree(X[right_idx], y[right_idx], depth + 1)
        }

    def _predict_one(self, x, tree):
        if not isinstance(tree, dict): return tree
        if x[tree['feat']] <= tree['threshold']:
            return self._predict_one(x, tree['left'])
        return self._predict_one(x, tree['right'])

    def predict(self, X):
        return np.array([self._predict_one(x, self.tree) for x in X])
```

#### 2. Le Random Forest

```python
class LightRandomForest:
    def __init__(self, n_trees=10, max_depth=5, max_features='sqrt'):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.max_features = max_features
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        n_samples, n_features = X.shape
        
        # Determine how many features to sample
        if self.max_features == 'sqrt':
            self.f_size = int(np.sqrt(n_features))
        else:
            self.f_size = n_features

        for _ in range(self.n_trees):
            # Bagging: Sample rows with replacement
            # TODO: Créer X_sample et y_sample par bootstrap
            indices = np.random.choice(...)
            X_sample, y_sample = ...
            
            tree = SimpleDecisionTree(max_depth=self.max_depth, feature_subsample=self.f_size)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        # Aggregate predictions (Majority Voting)
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        # Transpose to get [sample_index, tree_index]
        return np.array([Counter(sample_preds).most_common(1)[0][0] for sample_preds in tree_preds.T])
```


