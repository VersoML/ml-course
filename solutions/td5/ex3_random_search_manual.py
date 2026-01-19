"""
Exercice 3 : Implémenter Random Search from scratch
"""
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier

# Charger les données
X, y = load_wine(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

def random_search_manual(X, y, param_distributions, n_iter=20, random_state=42):
    """
    Implémente Random Search manuellement.
    """
    np.random.seed(random_state)
    
    results = []
    best_score = -np.inf
    best_params = None
    
    for i in range(n_iter):
        # Échantillonner des paramètres aléatoires
        params = {}
        for name, distribution in param_distributions.items():
            if callable(distribution):
                params[name] = distribution()
            else:
                params[name] = np.random.choice(distribution)
        
        # Créer et évaluer le modèle
        model = DecisionTreeClassifier(**params, random_state=42)
        scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
        mean_score = scores.mean()
        
        results.append({
            'params': params.copy(),
            'mean_score': mean_score,
            'std_score': scores.std()
        })
        
        # Mettre à jour le meilleur
        if mean_score > best_score:
            best_score = mean_score
            best_params = params.copy()
        
        print(f"Iter {i+1}/{n_iter}: {params} -> Score: {mean_score:.4f}")
    
    return best_params, best_score, results


# Définir les distributions
param_distributions = {
    'max_depth': lambda: np.random.randint(2, 31),
    'min_samples_split': lambda: np.random.randint(2, 21)
}

# Exécuter Random Search avec 20 itérations
best_params, best_score, results = random_search_manual(
    X_train, y_train, param_distributions, n_iter=20
)

print(f"\n{'='*50}")
print(f"Meilleurs paramètres: {best_params}")
print(f"Meilleur score CV: {best_score:.4f}")

# Évaluer sur le test set
final_model = DecisionTreeClassifier(**best_params, random_state=42)
final_model.fit(X_train, y_train)
test_score = final_model.score(X_test, y_test)
print(f"Score sur test set: {test_score:.4f}")

# Test de stabilité avec différents seeds
print(f"\n{'='*50}")
print("Test de stabilité (5 seeds différents):")
for seed in [0, 1, 2, 3, 4]:
    best_params, best_score, _ = random_search_manual(
        X_train, y_train, param_distributions, n_iter=20, random_state=seed
    )
    print(f"  Seed {seed}: score={best_score:.4f}, params={best_params}")

