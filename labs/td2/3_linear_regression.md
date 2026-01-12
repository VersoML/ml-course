# TP : Régression Linéaire : L'Équation Normale



## Rappel Théorique

### L'Équation Normale

La régression linéaire cherche à trouver les paramètres $\theta$ qui minimisent l'erreur quadratique moyenne entre les prédictions et les valeurs réelles.

L'équation normale nous donne directement la solution optimale :

$$\theta = (X^T X)^{-1} X^T y$$

Où :
- $X$ est la matrice des caractéristiques (avec une colonne de 1 pour le biais)
- $y$ est le vecteur des valeurs cibles
- $\theta$ contient les coefficients du modèle (intercept + pentes)

### Prédiction

Une fois le modèle entraîné, la prédiction se fait simplement par :

$$\hat{y} = X \cdot \theta$$

---

## Partie 1 : Préparation de l'environnement

### Import des bibliothèques

```python
import numpy as np
import matplotlib.pyplot as plt
```

---

## Partie 2 : Implémentation de la classe `LinearRegressionNormalEquation`

Complétez la classe suivante. Vous devez implémenter les méthodes `fit` et `predict`.

```python
import numpy as np


class LinearRegressionNormalEquation:
    def __init__(self):
        self.theta = None

    def fit(self, X, y):
        """
        Entraîne le modèle en utilisant l'Équation Normale.

        Paramètres:
        X : numpy array de forme (m, n) - Matrice des caractéristiques
        y : numpy array de forme (m, 1) - Valeurs cibles

        Étapes à suivre:
        1. Ajouter une colonne de 1 à X pour le terme de biais (intercept)
           Indice: utilisez np.c_[np.ones((m, 1)), X]
        
        2. Appliquer l'équation normale: theta = (X^T * X)^-1 * X^T * y
           Indices:
           - X.T donne la transposée de X
           - @ est l'opérateur de multiplication matricielle
           - np.linalg.inv() calcule l'inverse d'une matrice
           - np.linalg.pinv() calcule la pseudo-inverse (plus stable)
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
        3. Calculer les prédictions: y_pred = X_b . theta
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
model = LinearRegressionNormalEquation()
model.fit(X, y)

# Affichage des paramètres appris
print(f"Intercept (θ₀) calculé : {model.theta[0][0]:.4f} (attendu ~4)")
print(f"Pente (θ₁) calculée : {model.theta[1][0]:.4f} (attendu ~3)")
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
plt.title('Régression Linéaire avec l\'Équation Normale')
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

