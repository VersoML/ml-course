import numpy as np


class LogisticRegressionGD:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.theta = None

    def _sigmoid(self, z):
        """
        The Sigmoid activation function.
        """
        # Clip z to avoid overflow/underflow with exp
        z = np.clip(z, -250, 250)
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        """
        Fits the model using Gradient Descent.
        """
        m_samples, n_features = X.shape

        # 1. Add bias column (column of ones) to X
        X_b = np.c_[np.ones((m_samples, 1)), X]

        # 2. Initialize weights (theta) to zeros
        # Shape: (n_features + 1, 1)
        self.theta = np.zeros((n_features + 1, 1))

        # 3. Gradient Descent Loop
        for i in range(self.n_iterations):
            # a. Calculate linear prediction: z = X . theta
            linear_model = np.dot(X_b, self.theta)

            # b. Apply activation function
            y_predicted = self._sigmoid(linear_model)

            # c. Calculate Gradient: (1/m) * X.T * (y_pred - y)
            gradient = (1 / m_samples) * np.dot(X_b.T, (y_predicted - y))

            # d. Update weights
            self.theta -= self.learning_rate * gradient

    def predict_proba(self, X):
        """
        Returns probability (0 to 1) of class 1.
        """
        if self.theta is None:
            raise Exception("Model not fitted")

        m = X.shape[0]
        X_b = np.c_[np.ones((m, 1)), X]
        return self._sigmoid(np.dot(X_b, self.theta))

    def predict(self, X, threshold=0.5):
        """
        Returns class labels (0 or 1).
        """
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score

    # --- Génération des données synthétiques ---
    # Deux clusters de points
    np.random.seed(42)
    # Classe 0 : centrée en (2, 2)
    X0 = np.random.randn(50, 2) + 2
    # Classe 1 : centrée en (6, 6)
    X1 = np.random.randn(50, 2) + 6

    # Combinaison des données
    X = np.vstack((X0, X1))
    y = np.vstack((np.zeros((50, 1)), np.ones((50, 1))))

    # --- Entraînement de notre modèle ---
    model = LogisticRegressionGD(learning_rate=0.1, n_iterations=2000)
    model.fit(X, y)

    # --- Prédictions ---
    y_pred = model.predict(X)
    accuracy = np.mean(y_pred == y)

    # --- Affichage des résultats ---
    print("=" * 50)
    print("Notre implémentation (Régression Logistique)")
    print("=" * 50)
    print(f"Accuracy : {accuracy * 100:.2f}%")
    print(f"Biais    : {model.theta[0][0]:.6f}")
    print(f"Poids    : {model.theta[1:].flatten()}")

    # Points de test
    test_point_A = np.array([[2, 2]])
    test_point_B = np.array([[6, 6]])
    prob_A = model.predict_proba(test_point_A)
    prob_B = model.predict_proba(test_point_B)

    print(
        f"\nPrédiction pour (2,2) : Classe {model.predict(test_point_A)[0][0]} (Prob: {prob_A[0][0]:.4f})"
    )
    print(
        f"Prédiction pour (6,6) : Classe {model.predict(test_point_B)[0][0]} (Prob: {prob_B[0][0]:.4f})"
    )

    # --- Comparaison avec Scikit-Learn ---
    sklearn_model = LogisticRegression(max_iter=2000)
    sklearn_model.fit(X, y.ravel())
    y_pred_sklearn = sklearn_model.predict(X)
    accuracy_sklearn = accuracy_score(y, y_pred_sklearn)

    print("\n" + "=" * 50)
    print("Scikit-Learn LogisticRegression")
    print("=" * 50)
    print(f"Accuracy : {accuracy_sklearn * 100:.2f}%")
    print(f"Biais    : {sklearn_model.intercept_[0]:.6f}")
    print(f"Poids    : {sklearn_model.coef_[0]}")

    # --- Vérification ---
    print("\n" + "=" * 50)
    print("Vérification")
    print("=" * 50)
    print(f"Différence Accuracy : {abs(accuracy - accuracy_sklearn) * 100:.4f}%")

    if abs(accuracy - accuracy_sklearn) < 0.05:
        print("\n✓ Les deux implémentations donnent des résultats similaires !")

    # --- Visualisation ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Graphique 1: Données et frontière de décision
    axes[0].scatter(X0[:, 0], X0[:, 1], c="steelblue", label="Classe 0", alpha=0.7)
    axes[0].scatter(X1[:, 0], X1[:, 1], c="coral", label="Classe 1", alpha=0.7)

    # Frontière de décision : theta0 + theta1*x1 + theta2*x2 = 0
    # => x2 = -(theta0 + theta1*x1) / theta2
    x1_vals = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 100)
    x2_vals = -(model.theta[0][0] + model.theta[1][0] * x1_vals) / model.theta[2][0]
    axes[0].plot(x1_vals, x2_vals, "g-", linewidth=2, label="Frontière de décision")
    axes[0].set_xlim(X[:, 0].min() - 1, X[:, 0].max() + 1)
    axes[0].set_ylim(X[:, 1].min() - 1, X[:, 1].max() + 1)
    axes[0].set_xlabel("$x_1$")
    axes[0].set_ylabel("$x_2$")
    axes[0].set_title("Classification avec Régression Logistique")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Graphique 2: Fonction sigmoïde
    z = np.linspace(-10, 10, 200)
    sigmoid = 1 / (1 + np.exp(-z))
    axes[1].plot(z, sigmoid, color="steelblue", linewidth=2)
    axes[1].axhline(y=0.5, color="gray", linestyle="--", alpha=0.7)
    axes[1].axvline(x=0, color="gray", linestyle="--", alpha=0.7)
    axes[1].set_xlabel("z")
    axes[1].set_ylabel("$\\sigma(z)$")
    axes[1].set_title("Fonction Sigmoïde")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("logistic_regression_plot.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("\n✓ Graphiques sauvegardés dans 'logistic_regression_plot.png'")
