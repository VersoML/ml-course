"""
Exercice 4 : Utiliser RandomizedSearchCV
"""
import time
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import randint
import pandas as pd

# Charger les données
X, y = load_wine(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# === GRID SEARCH (pour comparaison) ===
print("GridSearchCV...")
param_grid = {
    'max_depth': [2, 5, 10, 15, 20],
    'min_samples_split': [2, 5, 10]
}

start_time = time.time()
grid_search = GridSearchCV(
    estimator=DecisionTreeClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)
grid_search.fit(X_train, y_train)
grid_time = time.time() - start_time

print(f"  Temps: {grid_time:.2f}s")
print(f"  Meilleur score CV: {grid_search.best_score_:.4f}")
print(f"  Score test: {grid_search.score(X_test, y_test):.4f}")

# === RANDOMIZED SEARCH ===
print("\nRandomizedSearchCV...")
param_distributions = {
    'max_depth': randint(2, 31),
    'min_samples_split': randint(2, 21)
}

start_time = time.time()
random_search = RandomizedSearchCV(
    estimator=DecisionTreeClassifier(random_state=42),
    param_distributions=param_distributions,
    n_iter=20,
    cv=5,
    scoring='accuracy',
    random_state=42,
    n_jobs=-1
)
random_search.fit(X_train, y_train)
random_time = time.time() - start_time

print(f"  Temps: {random_time:.2f}s")
print(f"  Meilleur score CV: {random_search.best_score_:.4f}")
print(f"  Meilleurs paramètres: {random_search.best_params_}")
print(f"  Score test: {random_search.score(X_test, y_test):.4f}")

# === COMPARAISON ===
print(f"\n{'='*60}")
print("COMPARAISON")
print("="*60)

comparison = pd.DataFrame({
    'Méthode': ['Grid Search', 'Random Search'],
    'Nb évaluations': [15, 20],
    'Temps (s)': [f"{grid_time:.3f}", f"{random_time:.3f}"],
    'Meilleur score CV': [f"{grid_search.best_score_:.4f}", f"{random_search.best_score_:.4f}"],
    'Score Test': [f"{grid_search.score(X_test, y_test):.4f}", f"{random_search.score(X_test, y_test):.4f}"]
})
print(comparison.to_string(index=False))

