# TP - Partie 1 : Voting Classifier from scratch

## Consigne
Implémentez deux fonctions de vote (**Hard Voting** et **Soft Voting**) et testez-les en combinant 3 modèles de `sklearn` (par exemple : Logistic Regression, Decision Tree, Random Forest).

## 1. Chargement des données

> **Important** : Exécutez d'abord le code de chargement des données présent dans le fichier [`README.md`](README.md).


## 2. À vous de jouer

1.  Entraînez 3 modèles différents (ex: LogisticRegression, DecisionTreeClassifier, RandomForestClassifier).
2.  Implémentez `hard_voting_predict(models, X)` qui retourne la classe majoritaire pour chaque échantillon.
3.  Implémentez `soft_voting_predict(models, X)` qui retourne la classe ayant la plus haute probabilité moyenne.
4.  Comparez les scores (Accuracy) des modèles individuels vs vos méthodes de Voting.

```python
# Vos implémentations ici
```

