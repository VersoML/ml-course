# TP : Régression Linéaire : Descente de Gradient



## Rappel Théorique

### La Descente de Gradient

La descente de gradient est un algorithme d'optimisation itératif qui cherche à minimiser une fonction de coût en suivant la direction opposée au gradient.

Pour la régression linéaire, la fonction de coût est l'erreur quadratique moyenne (MSE) :

$$J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2$$

où $h_\theta(x) = \theta^T x$ est notre fonction hypothèse.

### Algorithme

À chaque itération, on met à jour les paramètres selon :

$$\theta := \theta - \alpha \nabla J(\theta)$$

où :
- $\alpha$ est le **taux d'apprentissage** (learning rate)
- $\nabla J(\theta)$ est le gradient de la fonction de coût

Le gradient pour la régression linéaire est :

$$\nabla J(\theta) = \frac{1}{m} X^T (X\theta - y)$$

---

## Partie 1 : Préparation de l'environnement

### Import des bibliothèques

```python
import numpy as np
import matplotlib.pyplot as plt
```

---

## Partie 2 : Implémentation de la classe `LinearRegressionGradientDescent`

Complétez la classe suivante. Vous devez implémenter les méthodes `_compute_cost`, `fit` et `predict`.

```python
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

        Formule: J(theta) = (1 / 2m) * sum((predictions - y)^2)
        """
        # TODO: Implémentez cette méthode
        pass

    def fit(self, X, y):
        """
        Entraîne le modèle en utilisant la Descente de Gradient.

        Paramètres:
        X : numpy array de forme (m, n) - Matrice des caractéristiques
        y : numpy array de forme (m, 1) - Valeurs cibles

        Étapes à suivre:
        1. Ajouter une colonne de 1 à X pour le terme de biais
           Indice: utilisez np.c_[np.ones((m, 1)), X]

        2. Initialiser theta aléatoirement
           Indice: utilisez np.random.randn(n_features, 1)

        3. Pour chaque itération (boucle for):
           a. Calculer les prédictions: predictions = X_b @ theta
           b. Calculer les erreurs: errors = predictions - y
           c. Calculer les gradients: gradients = (1/m) * X_b.T @ errors
           d. Mettre à jour theta: theta = theta - learning_rate * gradients
           e. (Optionnel) Enregistrer le coût dans cost_history
        """
        # TODO: Implémentez cette méthode
        pass

    def predict(self, X):
        """
        Prédit les valeurs cibles pour les caractéristiques données.

        Paramètres:
        X : numpy array de forme (m, n) - Matrice des caractéristiques

        Retourne:
        numpy array de forme (m, 1) - Prédictions

        Étapes à suivre:
        1. Vérifier que le modèle a été entraîné (self.theta n'est pas None)
        2. Ajouter une colonne de 1 à X pour le terme de biais
        3. Calculer les prédictions: y_pred = X_b @ theta
        """
        # TODO: Implémentez cette méthode
        pass
```

---

## Partie 3 : Application sur des données synthétiques

### Génération des données

Nous allons générer des données qui suivent la relation linéaire : $y = 4 + 3x + \epsilon$

où $\epsilon$ est un bruit gaussien.

```python
# Génération des données synthétiques
np.random.seed(42)  # Pour la reproductibilité

# Génération de 100 points aléatoires entre 0 et 2
X = 2 * np.random.rand(100, 1)

# Génération de y avec la relation: y = 4 + 3x + bruit
y = 4 + 3 * X + np.random.randn(100, 1)

# Visualisation des données
plt.figure(figsize=(10, 6))
plt.scatter(X, y, alpha=0.7, label='Données')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Données synthétiques pour la régression linéaire')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### Entraînement du modèle

```python
# Création et entraînement du modèle
model = LinearRegressionGradientDescent(learning_rate=0.1, n_iterations=1000)
model.fit(X, y)

# Affichage des paramètres appris
print(f"Intercept (θ₀) calculé : {model.theta[0][0]:.4f} (attendu ~4)")
print(f"Pente (θ₁) calculée : {model.theta[1][0]:.4f} (attendu ~3)")
```

### Visualisation de la convergence

Un avantage de la descente de gradient est de pouvoir visualiser l'évolution du coût au fil des itérations.

```python
# Visualisation de l'évolution du coût
plt.figure(figsize=(10, 6))
plt.plot(model.cost_history, color='steelblue', linewidth=1.5)
plt.xlabel('Itération')
plt.ylabel('Coût (MSE)')
plt.title('Convergence de la Descente de Gradient')
plt.grid(True, alpha=0.3)
plt.show()
```

### Prédictions et visualisation

```python
# Prédictions pour X = 0 et X = 2
X_new = np.array([[0], [2]])
predictions = model.predict(X_new)

print(f"\nPrédiction pour X=0 : {predictions[0][0]:.4f}")
print(f"Prédiction pour X=2 : {predictions[1][0]:.4f}")

# Visualisation de la droite de régression
plt.figure(figsize=(10, 6))
plt.scatter(X, y, alpha=0.7, label='Données')
plt.plot(X_new, predictions, 'r-', linewidth=2, label='Régression linéaire')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Régression Linéaire avec Descente de Gradient')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

---

## Partie 4 : Vérification avec Scikit-Learn

Comparez votre implémentation avec celle de Scikit-Learn :

```python
from sklearn.linear_model import LinearRegression

# Modèle sklearn
sklearn_model = LinearRegression()
sklearn_model.fit(X, y)

print("Comparaison des résultats :")
print(f"\nNotre modèle :")
print(f"  Intercept : {model.theta[0][0]:.6f}")
print(f"  Pente     : {model.theta[1][0]:.6f}")

print(f"\nScikit-Learn :")
print(f"  Intercept : {sklearn_model.intercept_[0]:.6f}")
print(f"  Pente     : {sklearn_model.coef_[0][0]:.6f}")
```

---

## Partie 5 : Expérimentation avec les hyperparamètres

### Impact du taux d'apprentissage

Testez différentes valeurs du taux d'apprentissage et observez l'impact sur la convergence.

```python
learning_rates = [0.001, 0.01, 0.1, 0.5]

plt.figure(figsize=(12, 8))

for lr in learning_rates:
    model_test = LinearRegressionGradientDescent(learning_rate=lr, n_iterations=100)
    model_test.fit(X, y)
    plt.plot(model_test.cost_history, label=f'lr = {lr}', linewidth=1.5)

plt.xlabel('Itération')
plt.ylabel('Coût (MSE)')
plt.title('Impact du taux d\'apprentissage sur la convergence')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

**Questions à explorer :**
- Que se passe-t-il si le taux d'apprentissage est trop petit ?
- Que se passe-t-il si le taux d'apprentissage est trop grand ?

