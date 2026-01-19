"""
Exercice 2 : Utiliser GridSearchCV
"""
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
import pandas as pd

# Charger les données
X, y = load_wine(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Créer le modèle
model = DecisionTreeClassifier(random_state=42)

# Définir la grille
param_grid = {
    'max_depth': [2, 5, 10, 15, 20],
    'min_samples_split': [2, 5, 10]
}

# Créer GridSearchCV
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    verbose=1,
    n_jobs=-1
)

# Fit sur les données d'entraînement
grid_search.fit(X_train, y_train)

# Afficher les résultats
print(f"\nMeilleurs paramètres: {grid_search.best_params_}")
print(f"Meilleur score CV: {grid_search.best_score_:.4f}")

# Évaluer sur le test set
test_score = grid_search.score(X_test, y_test)
print(f"Score sur test set: {test_score:.4f}")

# Afficher les 5 meilleures combinaisons
results_df = pd.DataFrame(grid_search.cv_results_)
top_5 = results_df.nsmallest(5, 'rank_test_score')[
    ['params', 'mean_test_score', 'std_test_score', 'rank_test_score']
]
print("\nTop 5 combinaisons:")
print(top_5.to_string(index=False))

