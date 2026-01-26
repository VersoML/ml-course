# TP - Partie 4 : Stacking

## Consigne
Implémenter une architecture de Stacking simple pour combiner les forces de plusieurs modèles hétérogènes.

> **Prérequis** : Charger les données (voir `README.md`).

## Architecture
- **Niveau 0 (Base Models)** :
    - Logistic Regression
    - Random Forest
    - XGBoost
- **Niveau 1 (Meta Model)** :
    - Logistic Regression

## 1 : Stacking avec Sklearn
Utilisez `StackingClassifier` de `sklearn.ensemble`.

```python
from sklearn.ensemble import StackingClassifier
from sklearn.svm import SVC

estimators = [
    ('rf', RandomForestClassifier(...)),
    ('svr', SVC(...)),
    ('lr', LogisticRegression(...))
]

# TODO: Créer le StackingClassifier avec final_estimator=LogisticRegression()
# TODO: Entraîner et comparer avec les modèles individuels
```
