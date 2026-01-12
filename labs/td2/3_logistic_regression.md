# TP : Régression Logistique



## Rappel Théorique

### La Régression Logistique

La régression logistique est un algorithme de **classification binaire** qui modélise la probabilité qu'une observation appartienne à une classe donnée.

### La Fonction Sigmoïde

La fonction sigmoïde transforme une valeur réelle en une probabilité entre 0 et 1 :

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

où $z = \theta^T x$ est la combinaison linéaire des caractéristiques.

### Hypothèse

La probabilité que $y = 1$ sachant $x$ est :

$$h_\theta(x) = \sigma(\theta^T x) = \frac{1}{1 + e^{-\theta^T x}}$$

### Fonction de Coût (Log-Loss)

La fonction de coût pour la régression logistique est l'entropie croisée binaire :

$$J(\theta) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(h_\theta(x^{(i)})) + (1 - y^{(i)}) \log(1 - h_\theta(x^{(i)})) \right]$$

### Gradient

Le gradient de la fonction de coût est :

$$\nabla J(\theta) = \frac{1}{m} X^T (\sigma(X\theta) - y)$$

---

## Partie 1 : Préparation de l'environnement

### Import des bibliothèques

```python
import numpy as np
import matplotlib.pyplot as plt
```

---

## Partie 2 : Implémentation de la classe `LogisticRegressionGD`

Complétez la classe suivante. Vous devez implémenter les méthodes `_sigmoid`, `fit`, `predict_proba` et `predict`.

```python
import numpy as np


class LogisticRegressionGD:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        """
        Initialise le modèle de régression logistique.

        Paramètres:
        learning_rate : float - Taux d'apprentissage
        n_iterations : int - Nombre d'itérations
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.theta = None

    def _sigmoid(self, z):
        """
        Fonction d'activation sigmoïde.

        Paramètres:
        z : numpy array - Valeurs d'entrée

        Retourne:
        numpy array - Valeurs entre 0 et 1

        Formule: sigma(z) = 1 / (1 + exp(-z))

        Conseil: Utilisez np.clip(z, -250, 250) avant exp pour éviter
        les overflow/underflow.
        """
        # TODO: Implémentez cette méthode
        pass

    def fit(self, X, y):
        """
        Entraîne le modèle avec la descente de gradient.

        Paramètres:
        X : numpy array de forme (m, n) - Matrice des caractéristiques
        y : numpy array de forme (m, 1) - Labels (0 ou 1)

        Étapes à suivre:
        1. Ajouter une colonne de 1 à X pour le biais
        2. Initialiser theta à zéro (np.zeros)
        3. Pour chaque itération:
           a. Calculer z = X_b @ theta
           b. Calculer les prédictions avec sigmoid: y_pred = sigmoid(z)
           c. Calculer le gradient: gradient = (1/m) * X_b.T @ (y_pred - y)
           d. Mettre à jour theta: theta = theta - learning_rate * gradient
        """
        # TODO: Implémentez cette méthode
        pass

    def predict_proba(self, X):
        """
        Retourne la probabilité d'appartenir à la classe 1.

        Paramètres:
        X : numpy array de forme (m, n) - Caractéristiques

        Retourne:
        numpy array de forme (m, 1) - Probabilités

        Étapes:
        1. Ajouter la colonne de biais
        2. Retourner sigmoid(X_b @ theta)
        """
        # TODO: Implémentez cette méthode
        pass

    def predict(self, X, threshold=0.5):
        """
        Retourne les labels prédits (0 ou 1).

        Paramètres:
        X : numpy array - Caractéristiques
        threshold : float - Seuil de décision (défaut: 0.5)

        Retourne:
        numpy array - Labels prédits

        Étapes:
        1. Calculer les probabilités avec predict_proba
        2. Retourner 1 si prob >= threshold, sinon 0
        """
        # TODO: Implémentez cette méthode
        pass
```

---

## Partie 3 : Application sur des données synthétiques

### Génération des données

Nous allons créer deux clusters de points pour la classification binaire.

```python
# Génération des données synthétiques
np.random.seed(42)

# Classe 0 : centrée en (2, 2)
X0 = np.random.randn(50, 2) + 2

# Classe 1 : centrée en (6, 6)
X1 = np.random.randn(50, 2) + 6

# Combinaison des données
X = np.vstack((X0, X1))
y = np.vstack((np.zeros((50, 1)), np.ones((50, 1))))

# Visualisation des données
plt.figure(figsize=(10, 6))
plt.scatter(X0[:, 0], X0[:, 1], c='steelblue', label='Classe 0', alpha=0.7)
plt.scatter(X1[:, 0], X1[:, 1], c='coral', label='Classe 1', alpha=0.7)
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('Données synthétiques pour la classification binaire')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### Entraînement du modèle

```python
# Création et entraînement du modèle
model = LogisticRegressionGD(learning_rate=0.1, n_iterations=2000)
model.fit(X, y)

# Affichage des paramètres appris
print(f"Biais : {model.theta[0][0]:.4f}")
print(f"Poids : {model.theta[1:].flatten()}")

# Calcul de l'accuracy
y_pred = model.predict(X)
accuracy = np.mean(y_pred == y)
print(f"Accuracy : {accuracy * 100:.2f}%")
```

### Prédictions

```python
# Points de test
test_point_A = np.array([[2, 2]])  # Devrait être classe 0
test_point_B = np.array([[6, 6]])  # Devrait être classe 1

prob_A = model.predict_proba(test_point_A)
prob_B = model.predict_proba(test_point_B)

print(f"Point (2,2) : Classe {model.predict(test_point_A)[0][0]}, Prob={prob_A[0][0]:.4f}")
print(f"Point (6,6) : Classe {model.predict(test_point_B)[0][0]}, Prob={prob_B[0][0]:.4f}")
```

### Visualisation de la frontière de décision

```python
# Frontière de décision : theta0 + theta1*x1 + theta2*x2 = 0
# => x2 = -(theta0 + theta1*x1) / theta2

plt.figure(figsize=(10, 6))
plt.scatter(X0[:, 0], X0[:, 1], c='steelblue', label='Classe 0', alpha=0.7)
plt.scatter(X1[:, 0], X1[:, 1], c='coral', label='Classe 1', alpha=0.7)

x1_vals = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 100)
x2_vals = -(model.theta[0][0] + model.theta[1][0] * x1_vals) / model.theta[2][0]
plt.plot(x1_vals, x2_vals, 'g-', linewidth=2, label='Frontière de décision')

plt.xlim(X[:, 0].min() - 1, X[:, 0].max() + 1)
plt.ylim(X[:, 1].min() - 1, X[:, 1].max() + 1)
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('Classification avec Régression Logistique')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

---

## Partie 4 : Vérification avec Scikit-Learn

Comparez votre implémentation avec celle de Scikit-Learn :

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Modèle sklearn
sklearn_model = LogisticRegression(max_iter=2000)
sklearn_model.fit(X, y.ravel())

# Prédictions
y_pred_sklearn = sklearn_model.predict(X)
accuracy_sklearn = accuracy_score(y, y_pred_sklearn)

print("Comparaison des résultats :")
print(f"\nNotre modèle :")
print(f"  Accuracy : {accuracy * 100:.2f}%")
print(f"  Biais    : {model.theta[0][0]:.6f}")
print(f"  Poids    : {model.theta[1:].flatten()}")

print(f"\nScikit-Learn :")
print(f"  Accuracy : {accuracy_sklearn * 100:.2f}%")
print(f"  Biais    : {sklearn_model.intercept_[0]:.6f}")
print(f"  Poids    : {sklearn_model.coef_[0]}")
```

---

## Partie 5 : Visualisation de la fonction sigmoïde

```python
# Fonction sigmoïde
z = np.linspace(-10, 10, 200)
sigmoid = 1 / (1 + np.exp(-z))

plt.figure(figsize=(10, 6))
plt.plot(z, sigmoid, color='steelblue', linewidth=2)
plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, label='Seuil = 0.5')
plt.axvline(x=0, color='gray', linestyle='--', alpha=0.7)
plt.xlabel('z')
plt.ylabel('$\sigma(z)$')
plt.title('Fonction Sigmoïde')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

