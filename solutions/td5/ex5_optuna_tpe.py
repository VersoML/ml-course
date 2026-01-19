"""
Exercice 5 : Optimisation Bayésienne avec Optuna (TPE)
"""
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_wine

# Désactiver les logs verbeux d'Optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Charger les données
X, y = load_wine(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

def objective(trial):
    """Fonction objectif pour Optuna."""
    # Suggérer les hyperparamètres
    max_depth = trial.suggest_int('max_depth', 2, 30)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
    criterion = trial.suggest_categorical('criterion', ['gini', 'entropy'])
    
    # Créer le modèle
    model = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        criterion=criterion,
        random_state=42
    )
    
    # Retourner le score à maximiser
    return cross_val_score(model, X_train, y_train, cv=5).mean()


# Créer l'étude
study = optuna.create_study(
    direction='maximize',
    sampler=TPESampler(seed=42),
    study_name='tuning_decision_tree'
)

# Lancer l'optimisation
print("Optimisation en cours...")
study.optimize(objective, n_trials=30, show_progress_bar=True)

# Résultats
print(f"\n{'='*50}")
print(f"Meilleur score CV: {study.best_value:.4f}")
print(f"Meilleurs paramètres: {study.best_params}")

# Historique des essais
print("\nHistorique des 10 premiers essais:")
for i, trial in enumerate(study.trials[:10]):
    print(f"  Trial {i}: score={trial.value:.4f}, params={trial.params}")

# Évolution du meilleur score
print("\nConvergence:")
best_values = []
current_best = -float('inf')
for i, trial in enumerate(study.trials):
    if trial.value > current_best:
        current_best = trial.value
        print(f"  Trial {i}: nouveau meilleur score = {current_best:.4f}")
    best_values.append(current_best)

# Évaluation finale
print(f"\n{'='*50}")
print("ÉVALUATION FINALE")

best_model = DecisionTreeClassifier(
    max_depth=study.best_params['max_depth'],
    min_samples_split=study.best_params['min_samples_split'],
    min_samples_leaf=study.best_params['min_samples_leaf'],
    criterion=study.best_params['criterion'],
    random_state=42
)
best_model.fit(X_train, y_train)
final_score = best_model.score(X_test, y_test)
print(f"Score final sur test set: {final_score:.4f}")

# Visualisation (optionnel)
try:
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Évolution du score
    axes[0].plot(best_values, 'b-', linewidth=2)
    axes[0].set_xlabel('Essai')
    axes[0].set_ylabel('Meilleur score')
    axes[0].set_title('Convergence de l\'optimisation')
    axes[0].grid(True, alpha=0.3)
    
    # Distribution de max_depth
    max_depths = [t.params['max_depth'] for t in study.trials]
    scores = [t.value for t in study.trials]
    axes[1].scatter(max_depths, scores, alpha=0.6, c=range(len(scores)), cmap='viridis')
    axes[1].set_xlabel('max_depth')
    axes[1].set_ylabel('Score')
    axes[1].set_title('Exploration de max_depth')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('optuna_convergence.png', dpi=150)
    print("\nGraphique sauvegardé: optuna_convergence.png")
    plt.show()
except ImportError:
    print("\nMatplotlib non disponible pour la visualisation")

