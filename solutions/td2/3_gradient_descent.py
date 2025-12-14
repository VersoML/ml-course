import numpy as np


class LinearRegressionGradientDescent:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        """
        Initialise le modèle de régression linéaire avec descente de gradient.

        Paramètres:
        learning_rate : float - Taux d'apprentissage (eta)
        n_iterations : int - Nombre d'itérations de la descente de gradient
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.theta = None
        self.cost_history = []

    def _compute_cost(self, X_b, y):
        """
        Calcule le coût (MSE) pour les paramètres actuels.

        Paramètres:
        X_b : numpy array - Matrice des caractéristiques avec biais
        y : numpy array - Valeurs cibles

        Retourne:
        float - Coût MSE
        """
        m = len(y)
        predictions = X_b @ self.theta
        cost = (1 / (2 * m)) * np.sum((predictions - y) ** 2)
        return cost

    def fit(self, X, y):
        """
        Entraîne le modèle en utilisant la Descente de Gradient.

        Paramètres:
        X : numpy array de forme (m, n) - Matrice des caractéristiques
        y : numpy array de forme (m, 1) - Valeurs cibles

        Algorithme:
        1. Ajouter une colonne de 1 à X pour le terme de biais
        2. Initialiser theta aléatoirement
        3. Pour chaque itération:
           a. Calculer les prédictions: y_pred = X_b @ theta
           b. Calculer les gradients: gradients = (1/m) * X_b.T @ (y_pred - y)
           c. Mettre à jour theta: theta = theta - learning_rate * gradients
           d. (Optionnel) Enregistrer le coût pour visualisation
        """
        # 1. Ajouter une colonne de 1 à X pour le terme de biais (intercept)
        m = X.shape[0]
        X_b = np.c_[np.ones((m, 1)), X]
        n_features = X_b.shape[1]

        # 2. Initialiser theta aléatoirement
        np.random.seed(42)
        self.theta = np.random.randn(n_features, 1)

        # 3. Descente de gradient
        self.cost_history = []

        for iteration in range(self.n_iterations):
            # a. Calculer les prédictions
            predictions = X_b @ self.theta

            # b. Calculer les erreurs
            errors = predictions - y

            # c. Calculer les gradients
            # gradient = (1/m) * X^T * (predictions - y)
            gradients = (1 / m) * X_b.T @ errors

            # d. Mettre à jour theta
            self.theta = self.theta - self.learning_rate * gradients

            # Enregistrer le coût pour visualisation
            cost = self._compute_cost(X_b, y)
            self.cost_history.append(cost)

    def predict(self, X):
        """
        Prédit les valeurs cibles pour les caractéristiques données.

        Paramètres:
        X : numpy array de forme (m, n) - Matrice des caractéristiques

        Retourne:
        numpy array de forme (m, 1) - Prédictions
        """
        if self.theta is None:
            raise Exception("Ce modèle n'a pas encore été entraîné.")

        # Ajouter la colonne de biais aux nouvelles données
        m = X.shape[0]
        X_b = np.c_[np.ones((m, 1)), X]

        # Calculer les prédictions: y_pred = X_b @ theta
        return X_b @ self.theta


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LinearRegression

    # --- Génération des données synthétiques ---
    # y = 4 + 3x + bruit
    np.random.seed(42)
    X = 2 * np.random.rand(100, 1)
    y = 4 + 3 * X + np.random.randn(100, 1)

    # --- Utilisation de notre modèle ---
    model = LinearRegressionGradientDescent(learning_rate=0.1, n_iterations=1000)

    # Entraînement du modèle
    model.fit(X, y)

    # Prédiction pour X = 0 et X = 2
    X_new = np.array([[0], [2]])
    predictions = model.predict(X_new)

    # --- Affichage des résultats ---
    print("=" * 50)
    print("Notre implémentation (Descente de Gradient)")
    print("=" * 50)
    print(f"Intercept (θ₀) : {model.theta[0][0]:.6f} (attendu ~4)")
    print(f"Pente (θ₁)     : {model.theta[1][0]:.6f} (attendu ~3)")
    print(f"Prédiction pour X=0 : {predictions[0][0]:.6f}")
    print(f"Prédiction pour X=2 : {predictions[1][0]:.6f}")

    # --- Comparaison avec Scikit-Learn ---
    sklearn_model = LinearRegression()
    sklearn_model.fit(X, y)
    sklearn_predictions = sklearn_model.predict(X_new)

    print("\n" + "=" * 50)
    print("Scikit-Learn LinearRegression")
    print("=" * 50)
    print(f"Intercept (θ₀) : {sklearn_model.intercept_[0]:.6f}")
    print(f"Pente (θ₁)     : {sklearn_model.coef_[0][0]:.6f}")
    print(f"Prédiction pour X=0 : {sklearn_predictions[0][0]:.6f}")
    print(f"Prédiction pour X=2 : {sklearn_predictions[1][0]:.6f}")

    # --- Vérification ---
    print("\n" + "=" * 50)
    print("Vérification (différence entre les deux modèles)")
    print("=" * 50)
    diff_intercept = abs(model.theta[0][0] - sklearn_model.intercept_[0])
    diff_slope = abs(model.theta[1][0] - sklearn_model.coef_[0][0])
    print(f"Différence Intercept : {diff_intercept:.10f}")
    print(f"Différence Pente     : {diff_slope:.10f}")

    if diff_intercept < 0.01 and diff_slope < 0.01:
        print("\n✓ Les deux implémentations donnent des résultats similaires !")

    # --- Visualisation ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Graphique 1: Données et droite de régression
    axes[0].scatter(X, y, alpha=0.7, label="Données", color="steelblue")
    axes[0].plot(X_new, predictions, "r-", linewidth=2, label="Régression linéaire")
    axes[0].set_xlabel("X")
    axes[0].set_ylabel("y")
    axes[0].set_title("Régression Linéaire avec Descente de Gradient")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Graphique 2: Évolution du coût
    axes[1].plot(model.cost_history, color="steelblue", linewidth=1.5)
    axes[1].set_xlabel("Itération")
    axes[1].set_ylabel("Coût (MSE)")
    axes[1].set_title("Convergence de la Descente de Gradient")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("gradient_descent_plot.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("\n✓ Graphiques sauvegardés dans 'gradient_descent_plot.png'")
