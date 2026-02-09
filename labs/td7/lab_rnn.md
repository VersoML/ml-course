# TP : Réseaux de Neurones Récurrents (RNN)

```bash
pip install jax flax optax numpy matplotlib
```



Lancez `main.py` dans chaque dossier d'exercice pour tester et entraîner.

---

## Exercice 1 : RNN from scratch

### 1.1 - Rappel mathématique

À chaque pas de temps $t$, un RNN calcule :

$$h_t = \tanh(W_{xh} \cdot x_t + W_{hh} \cdot h_{t-1} + b_h)$$

$$y_t = W_{hy} \cdot h_t + b_y$$

Où :
- $x_t \in \mathbb{R}^{d}$ : entrée au temps $t$
- $h_t \in \mathbb{R}^{H}$ : état caché au temps $t$
- $y_t \in \mathbb{R}^{o}$ : sortie au temps $t$
- $W_{xh} \in \mathbb{R}^{H \times d}$, $W_{hh} \in \mathbb{R}^{H \times H}$, $W_{hy} \in \mathbb{R}^{o \times H}$ : poids
- $b_h \in \mathbb{R}^{H}$, $b_y \in \mathbb{R}^{o}$ : biais

On déroule le réseau sur $T$ pas de temps et on rétropropage les gradients à travers toute la chaîne (**BPTT**). Les mêmes poids étant partagés à chaque pas, les gradients **s'additionnent**.

### 1.2 - À faire

Dans `simple_rnn.py` :

1. **`step(x_t, h_prev)`** — Implémentez un seul pas de temps du RNN (les 2 formules ci-dessus).

2. **`forward(X)`** — Déroulez le RNN sur une séquence de $T$ pas. Initialisez $h_0 = \vec{0}$, bouclez en appelant `step`, stockez les sorties et états cachés.

Dans `flax_rnn.py` :

3. **`jax_rnn_forward`** — Implémentez le même forward pass en JAX pur avec `jax.lax.scan`. Les poids NumPy sont passés en paramètre.


---

## Exercice 2 : GRU from scratch

### 2.1 - Rappel mathématique

Le GRU résout le problème des **gradients évanescents** grâce à deux portes : la **porte de mise à jour** ($z_t$) et la **porte de réinitialisation** ($r_t$).

À chaque pas de temps $t$ :

$$z_t = \sigma(W_{xz} \cdot x_t + W_{hz} \cdot h_{t-1} + b_z) \quad \text{(porte de mise à jour)}$$

$$r_t = \sigma(W_{xr} \cdot x_t + W_{hr} \cdot h_{t-1} + b_r) \quad \text{(porte de réinitialisation)}$$

$$\tilde{h}_t = \tanh(W_{xh} \cdot x_t + W_{hh} \cdot (r_t \odot h_{t-1}) + b_h) \quad \text{(candidat)}$$

$$h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t \quad \text{(nouvel état caché)}$$

$$y_t = W_{hy} \cdot h_t + b_y \quad \text{(sortie)}$$

Où $\sigma$ est la sigmoïde et $\odot$ le produit élément par élément.

**Intuition :** quand $z_t \approx 0$, l'état est **copié tel quel** ($h_t \approx h_{t-1}$), créant un raccourci pour les gradients. C'est ce qui permet au GRU de mémoriser sur de longues séquences.

### 2.2 - À faire

Dans `simple_gru.py` :

1. **`step(x_t, h_prev)`** — Implémentez un seul pas de temps du GRU (les 5 formules ci-dessus). Retournez $h_t$, $y_t$, $z_t$, $r_t$, $\tilde{h}_t$.

2. **`forward(X)`** — Déroulez le GRU sur $T$ pas. Stockez les sorties, états cachés, et les valeurs intermédiaires (portes et candidat) dans un dictionnaire `cache`.

Dans `flax_gru.py` :

3. **`jax_gru_forward`** — Implémentez le même forward pass en JAX pur avec `jax.lax.scan`.


