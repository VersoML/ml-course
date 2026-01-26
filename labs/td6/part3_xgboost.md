# TP - Partie 3 : Boosting & XGBoost

## Consigne
Découvrir la puissance du Boosting et manipuler la librairie XGBoost.

> **Prérequis** : Charger les données (voir `README.md`).

## 0. Le Gradient Boosting

### Principe
Contrairement au Bagging qui entraîne des modèles en parallèle pour réduire la variance, le **Boosting** entraîne des modèles **séquentiellement** pour réduire le biais (et l'erreur globale).

Dans le **Gradient Boosting** :
1.  On entraîne un premier modèle $h_0(x)$ (souvent très simple).
2.  On calcule les **erreurs** (ou "résidus") de ce modèle : $r_0 = y - h_0(x)$.
3.  On entraîne un deuxième modèle $h_1(x)$ pour prédire ces résidus $r_0$.
4.  On combine les modèles : $H(x) = h_0(x) + \alpha \cdot h_1(x)$.
5.  On répète l'opération. 

Chaque nouveau modèle corrige les erreurs des précédents. C'est comme une descente de gradient, mais dans l'espace des fonctions !

## 1. Gradient Boosting avec Sklearn

**GradientBoostingClassifier (Sklearn)** est l'implémentation "classique". Elle est efficace pour des petits/moyens datasets, mais séquentielle et parfois lente.
Utilisez `GradientBoostingClassifier` de `sklearn.ensemble`.

```python
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

# TODO 1 : Entraîner le modèle de base avec les paramètres par défaut
# TODO 2 : Varier le learning_rate (0.01, 0.1, 1.0). Quel est l'impact ?
# TODO 3 : Varier n_estimators (10, 100, 500). Attention au surapprentissage si combiné avec un fort learning rate !
```

## 2. XGBoost (Extreme Gradient Boosting)

**XGBoost** est une librairie externe optimisée pour :
*   **Rapidité** : Calculs parallélisés, approximations intelligentes pour trouver les splits.
*   **Régularisation** : Ajoute des pénalités (L1/L2) pour éviter le surapprentissage (ce que Sklearn ne fait pas par défaut).
*   **Hardware** : Optimisé pour le cache et support GPU.
*   **Valeurs Manquantes** : Les gère nativement.

```python
import xgboost as xgb
model_xgb = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    use_label_encoder=False,
    eval_metric='logloss'
)

# TODO : Fit et Predict
```

## 3. Early Stopping

L'un des grands avantages de XGBoost est l'**Early Stopping** : on arrête d'ajouter des arbres si la performance sur un set de validation ne s'améliore plus. Cela évite de devoir "deviner" le bon `n_estimators`.

```python
# Créez un eval_set (ici on triche un peu en utilisant le test set pour l'exemple, 
# en pratique on utiliserait un set de validation séparé)
eval_set = [(X_test_scaled, y_test)]

model_xgb.fit(
    X_train_scaled, 
    y_train, 
    early_stopping_rounds=10, 
    eval_set=eval_set, 
    verbose=True
)

# Observe à quel indice l'entraînement s'est arrêté.
```
