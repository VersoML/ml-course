import numpy as np


class LinearRegressionNormalEquation:
    def __init__(self):
        self.theta = None

    def fit(self, X, y):
        """
        Fits the model using the Normal Equation.

        Parameters:
        X : numpy array of shape (m, n) - Features
        y : numpy array of shape (m, 1) - Target values
        """
        # 1. Add a column of ones to X for the bias term (intercept)
        # m is number of samples, n is number of features
        m = X.shape[0]
        X_b = np.c_[np.ones((m, 1)), X]

        # 2. Apply the Normal Equation: theta = (X.T * X)^-1 * X.T * y
        # We use np.linalg.inv for inversion and @ for matrix multiplication
        X_transpose = X_b.T

        # Calculate (X^T * X)
        xtx = X_transpose @ X_b

        # Calculate Inverse of (X^T * X)
        try:
            xtx_inv = np.linalg.inv(xtx)
        except np.linalg.LinAlgError:
            # Fallback for singular matrices (using pseudo-inverse)
            xtx_inv = np.linalg.pinv(xtx)

        # Final calculation
        self.theta = xtx_inv @ X_transpose @ y

    def predict(self, X):
        """
        Predicts target values for given input features.
        """
        if self.theta is None:
            raise Exception("This model has not been fitted yet.")

        # Add the bias column to the new X data
        m = X.shape[0]
        X_b = np.c_[np.ones((m, 1)), X]

        # Calculate predictions: y_pred = X_b . theta
        return X_b @ self.theta


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LinearRegression

    # --- Generating Synthetic Data ---
    # y = 4 + 3x + noise
    np.random.seed(42)  # Pour la reproductibilité
    X = 2 * np.random.rand(100, 1)
    y = 4 + 3 * X + np.random.randn(100, 1)

    # --- Using our Model ---
    model = LinearRegressionNormalEquation()

    # Fit the model
    model.fit(X, y)

    # Make a prediction for X = 0 and X = 2
    X_new = np.array([[0], [2]])
    predictions = model.predict(X_new)

    # --- Output Results ---
    print("=" * 50)
    print("Notre implémentation (Équation Normale)")
    print("=" * 50)
    print(f"Intercept (θ₀) : {model.theta[0][0]:.6f} (attendu ~4)")
    print(f"Pente (θ₁)     : {model.theta[1][0]:.6f} (attendu ~3)")
    print(f"Prédiction pour X=0 : {predictions[0][0]:.6f}")
    print(f"Prédiction pour X=2 : {predictions[1][0]:.6f}")

    # --- Comparison with Scikit-Learn ---
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

    # --- Verification ---
    print("\n" + "=" * 50)
    print("Vérification (différence entre les deux modèles)")
    print("=" * 50)
    diff_intercept = abs(model.theta[0][0] - sklearn_model.intercept_[0])
    diff_slope = abs(model.theta[1][0] - sklearn_model.coef_[0][0])
    print(f"Différence Intercept : {diff_intercept:.10f}")
    print(f"Différence Pente     : {diff_slope:.10f}")

    if diff_intercept < 1e-10 and diff_slope < 1e-10:
        print("\n✓ Les deux implémentations donnent des résultats identiques !")

    # --- Visualization ---

    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, alpha=0.7, label="Données", color="steelblue")
    plt.plot(X_new, predictions, "r-", linewidth=2, label="Régression linéaire")
    plt.xlabel("X")
    plt.ylabel("y")
    plt.title("Régression Linéaire avec l'Équation Normale")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("linear_regression_plot.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("\n✓ Graphique sauvegardé dans 'linear_regression_plot.png'")
