# TP : Méthodes d'Ensemble

Ce TP explore différentes méthodes d'ensemble pour améliorer la performance de modèles de classification.
Nous explorerons le Voting, le Bagging, le Boosting et le Stacking.

## Dataset & Setup Commun

Nous utilisons le dataset **Heart Disease** (UCI).
Voici le code commun à exécuter au début de chaque exercice pour charger et préparer les données :

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Chargement du dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
           'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']

df = pd.read_csv(url, names=columns, na_values='?')
df = df.dropna()
df['target'] = (df['target'] > 0).astype(int)

# Préparation X, y
X = df.drop('target', axis=1).values
y = df['target'].values

# Split Train/Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scaling (nécessaire pour certains modèles comme SVM ou Logistic Regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Dataset chargé : {X_train.shape[0]} train, {X_test.shape[0]} test")
```

## Exercices

### [Partie 1 : Voting Classifier](part1_voting.md)
Implémentez vous-même le **Hard Voting** (vote majoritaire) et le **Soft Voting** (moyenne des probabilités).
Vous comparerez ces méthodes d'ensemble "maison" avec les modèles individuels (Logistic Regression, Decision Tree, Random Forest).

### [Partie 2 : Bagging & Random Forest](part2_bagging_rf.md)
Comprenez comment le **Bagging** réduit la variance en agrégeant des modèles entraînés sur des échantillons bootstrap.
Vous implémenterez ensuite une simulation simplifiée de **Random Forest** (Bagging + Feature Sampling).

### [Partie 3 : Boosting (XGBoost)](part3_xgboost.md)
Découvrez le principe du Boosting (correction séquentielle des erreurs) avec **Gradient Boosting** et **XGBoost**.
Vous explorerez l'impact des hyperparamètres (learning rate) et le mécanisme d'Early Stopping.

### [Partie 4 : Stacking](part4_stacking.md)
Mettez en place une architecture de **Stacking** à deux niveaux.
Plusieurs modèles hétérogènes (Niveau 0) génèrent des prédictions qui servent d'entrée à un méta-modèle (Niveau 1) pour la décision finale.
