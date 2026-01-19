"""
Exercice 1 : Implémenter Grid Search from scratch
"""
import numpy as np
from itertools import product
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier

# Charger les données
X, y = load_wine(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

def grid_search_manual(X, y, param_grid):
    """
    Implémente Grid Search manuellement.
    """
    results = []
    best_score = -np.inf
    best_params = None
    
    # Générer toutes les combinaisons
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    
    for values in product(*param_values):
        # Créer le dictionnaire de paramètres
        params = dict(zip(param_names, values))
        
        # Créer le modèle avec ces paramètres
        model = DecisionTreeClassifier(**params, random_state=42)
        
        # Calculer le score par validation croisée (5-fold)
        scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
        mean_score = scores.mean()
        
        # Enregistrer le résultat
        results.append({
            'params': params,
            'mean_score': mean_score,
            'std_score': scores.std()
        })
        
        # Mettre à jour le meilleur score
        if mean_score > best_score:
            best_score = mean_score
            best_params = params
        
        print(f"Params: {params} -> Score: {mean_score:.4f} (+/- {scores.std():.4f})")
    
    return best_params, best_score, results


# Définir la grille
param_grid = {
    'max_depth': [2, 5, 10, 15, 20],
    'min_samples_split': [2, 5, 10]
}

# Exécuter Grid Search
best_params, best_score, results = grid_search_manual(X_train, y_train, param_grid)

print(f"\n{'='*50}")
print(f"Nombre de combinaisons testées: {len(results)}")
print(f"Nombre d'entraînements total: {len(results) * 5}")
print(f"Meilleurs paramètres: {best_params}")
print(f"Meilleur score CV: {best_score:.4f}")

# Évaluer sur le test set
final_model = DecisionTreeClassifier(**best_params, random_state=42)
final_model.fit(X_train, y_train)
test_score = final_model.score(X_test, y_test)
print(f"Score sur test set: {test_score:.4f}")

