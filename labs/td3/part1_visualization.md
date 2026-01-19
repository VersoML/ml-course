
## Partie 1 : Intuition Visuelle et Frontières de Décision

**Objectif :** Comprendre les limites de la linéarité et la nécessité des couches cachées sans écrire de code.
**Outil :** [TensorFlow Playground](https://playground.tensorflow.org/) (Navigateur Web)

### Exercice 1.1 : Le Perceptron Simple (La limite linéaire)

1. Allez sur TensorFlow Playground.
2. Sélectionnez le jeu de données **"Gaussian"** (les deux taches, une bleue et une orange).
3. Configurez le réseau pour avoir **0 hidden layers** (cela équivaut à un Perceptron simple ou régression logistique).
4. Lancez l'entraînement.
* *Question :* La frontière de décision (la ligne qui sépare les couleurs) est-elle courbe ou droite ? Le modèle arrive-t-il à séparer les données ?

### Exercice 1.2 : Le Problème du XOR

1. Changez le jeu de données pour le **"XOR"** (4 quarts : bleu en haut à gauche/bas à droite, orange ailleurs).
2. Gardez **0 hidden layers**. Lancez l'entraînement.
* *Question :* Le modèle parvient-il à converger ? Pourquoi le "Loss" (l'erreur) ne descend-il pas ?

3. **Défi :** Ajoutez maintenant **1 hidden layer**.
* Commencez avec 1 neurone, puis 2, puis 3.
* *Question :* Quel est le nombre **minimum** de neurones nécessaires dans la couche cachée pour résoudre ce problème correctement ?

### Exercice 1.3 : L'importance de la Non-Linéarité

1. Gardez la configuration qui a résolu le XOR.
2. Changez la fonction d'activation de "Tanh" ou "ReLU" à **"Linear"**.
3. Relancez l'entraînement.
* *Question :* Même avec plusieurs couches, si l'activation est linéaire, le modèle peut-il résoudre le XOR ? Que pouvez-vous en conclure sur l'empilement de couches linéaires ?
