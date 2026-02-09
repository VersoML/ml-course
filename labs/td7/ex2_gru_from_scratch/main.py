"""
Exercice 2 : Implémenter un GRU from scratch avec NumPy.
Version JAX : la vérification finale utilise JAX/Flax au lieu de PyTorch.

Script principal — tests, entraînement et vérification.
"""
import numpy as np
import matplotlib.pyplot as plt

from simple_gru import SimpleGRU, numerical_gradient
from flax_gru import jax_gru_forward


# ============================================================
# PARTIE 1 : TEST DE LA CELLULE GRU (un pas de temps)
# ============================================================
print("=" * 60)
print("Partie 1 : Test de la cellule GRU (un pas de temps)")
print("=" * 60)

np.random.seed(42)
gru = SimpleGRU(input_size=3, hidden_size=5, output_size=1)

x = np.random.randn(3, 1)
h = np.zeros((5, 1))

h_new, y, z, r, h_tilde = gru.step(x, h)
print(f"h_new shape   : {h_new.shape}  (attendu : (5, 1))")
print(f"y shape       : {y.shape}  (attendu : (1, 1))")
print(f"z shape       : {z.shape}  (attendu : (5, 1))")
print(f"z in [0, 1]   : {np.all((z >= 0) & (z <= 1))}  (True car sigmoid)")
print(f"r in [0, 1]   : {np.all((r >= 0) & (r <= 1))}  (True car sigmoid)")
print(f"h_tilde in [-1,1] : {np.all(np.abs(h_tilde) <= 1)}  (True car tanh)")

# ============================================================
# PARTIE 2 : TEST DU FORWARD PASS
# ============================================================
print("\n" + "=" * 60)
print("Partie 2 : Test du forward pass (séquence de 10 pas)")
print("=" * 60)

T = 10
X = np.random.randn(T, 3, 1)
outputs, hiddens, cache = gru.forward(X)
print(f"outputs shape  : {outputs.shape}  (attendu : (10, 1, 1))")
print(f"hiddens shape  : {hiddens.shape}  (attendu : (11, 5, 1))")
print(f"gates_z shape  : {cache['gates_z'].shape}  (attendu : (10, 5, 1))")
print(f"gates_r shape  : {cache['gates_r'].shape}  (attendu : (10, 5, 1))")
print(f"hiddens[0] est zéros : {np.allclose(hiddens[0], 0)}")

# ============================================================
# PARTIE 3 : TEST DE BPTT + VÉRIFICATION NUMÉRIQUE
# ============================================================
print("\n" + "=" * 60)
print("Partie 3 : Test de BPTT et vérification numérique")
print("=" * 60)

np.random.seed(123)
gru_check = SimpleGRU(input_size=2, hidden_size=3, output_size=1)

T_check = 4
X_check = np.random.randn(T_check, 2, 1)
targets_check = np.random.randn(T_check, 1, 1)

outputs_check, hiddens_check, cache_check = gru_check.forward(X_check)
grads = gru_check.backward(
    X_check, outputs_check, hiddens_check, cache_check, targets_check,
    clip_value=1e10,
)

for name, g in grads.items():
    print(f"  {name}: shape={g.shape}, |max|={np.abs(g).max():.6f}")

print("\nVérification numérique (erreur relative) :")
param_names = [
    "W_xz", "W_hz", "b_z",
    "W_xr", "W_hr", "b_r",
    "W_xh", "W_hh", "b_h",
    "W_hy", "b_y",
]
grad_keys = ["d" + p for p in param_names]

for pname, gkey in zip(param_names, grad_keys):
    grad_num = numerical_gradient(gru_check, X_check, targets_check, pname)
    grad_ana = grads[gkey]
    denom = np.maximum(np.abs(grad_num) + np.abs(grad_ana), 1e-8)
    rel_err = np.max(np.abs(grad_num - grad_ana) / denom)
    status = "OK" if rel_err < 1e-5 else "ERREUR"
    print(f"  {pname:<5s} : erreur relative max = {rel_err:.2e}  [{status}]")

# ============================================================
# PARTIE 4 : ENTRAÎNEMENT SUR UN SINUS
# ============================================================
print("\n" + "=" * 60)
print("Partie 4 : Entraînement — prédiction d'un sinus")
print("=" * 60)

np.random.seed(42)
T_period = 20
n_steps = 500
t = np.arange(n_steps)
signal = np.sin(2 * np.pi * t / T_period)

seq_length = 25
gru = SimpleGRU(input_size=1, hidden_size=16, output_size=1)

lr = 0.005
n_epochs = 200
losses = []

for epoch in range(n_epochs):
    total_loss = 0
    n_seqs = 0

    for i in range(0, len(signal) - seq_length - 1, seq_length):
        X_seq = signal[i : i + seq_length].reshape(seq_length, 1, 1)
        Y_seq = signal[i + 1 : i + seq_length + 1].reshape(seq_length, 1, 1)

        outputs, hiddens, cache = gru.forward(X_seq)
        loss = 0.5 * np.sum((outputs - Y_seq) ** 2)
        total_loss += loss
        n_seqs += 1

        grads = gru.backward(X_seq, outputs, hiddens, cache, Y_seq)
        gru.update_params(grads, lr)

    avg_loss = total_loss / n_seqs
    losses.append(avg_loss)
    if (epoch + 1) % 40 == 0:
        print(f"  Époque {epoch+1:>3d}/{n_epochs} — Loss : {avg_loss:.6f}")

plt.figure(figsize=(10, 4))
plt.plot(losses, color="steelblue", linewidth=1.5)
plt.title("Convergence de l'entraînement (GRU from scratch)")
plt.xlabel("Époque")
plt.ylabel("Loss (MSE)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("gru_scratch_loss.png", dpi=150, bbox_inches="tight")
plt.show()

test_start = 400
X_test = signal[test_start : test_start + seq_length].reshape(seq_length, 1, 1)
Y_test = signal[test_start + 1 : test_start + seq_length + 1]

preds, _, _ = gru.forward(X_test)
preds = preds.flatten()

plt.figure(figsize=(10, 4))
plt.plot(Y_test, label="Réel", color="steelblue", linewidth=2)
plt.plot(preds, label="Prédiction (GRU from scratch)", color="orangered",
         linewidth=2, linestyle="--")
plt.title("Prédictions du GRU from scratch sur un sinus")
plt.xlabel("Pas de temps")
plt.ylabel("y(t)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("gru_scratch_predictions.png", dpi=150, bbox_inches="tight")
plt.show()

# ============================================================
# PARTIE 5 : VÉRIFICATION AVEC JAX/FLAX
# ============================================================
print("\n" + "=" * 60)
print("Partie 5 : Vérification avec JAX/Flax")
print("=" * 60)

X_verify = signal[0:seq_length].reshape(seq_length, 1, 1)
our_outputs, our_hiddens, _ = gru.forward(X_verify)

jax_all_h = jax_gru_forward(
    gru.W_xz, gru.W_hz, gru.b_z,
    gru.W_xr, gru.W_hr, gru.b_r,
    gru.W_xh, gru.W_hh, gru.b_h,
    X_verify, seq_length,
)

our_all_h = our_hiddens[1:].reshape(seq_length, -1)  # ignorer h_0
jax_all_h_np = np.array(jax_all_h)

diff_all = np.max(np.abs(our_all_h - jax_all_h_np))
print(f"Différence max sur tous les états cachés : {diff_all:.2e}")

if diff_all < 1e-5:
    print("Les deux implémentations produisent les mêmes résultats !")
else:
    print("Différence détectée => vérifiez votre implémentation.")

our_last_h = our_hiddens[-1].flatten()
jax_last_h = np.array(jax_all_h[-1])

diff = np.max(np.abs(our_last_h - jax_last_h))
print(f"Différence max sur le dernier état caché : {diff:.2e}")

# ============================================================
# PARTIE 6 : NOMBRE DE PARAMÈTRES — GRU vs RNN
# ============================================================
print("\n" + "=" * 60)
print("Partie 6 : Comparaison du nombre de paramètres")
print("=" * 60)

input_size, hidden_size, output_size = 1, 16, 1

# RNN : W_xh (H*d) + W_hh (H*H) + b_h (H) + W_hy (o*H) + b_y (o)
rnn_params = (hidden_size * input_size + hidden_size * hidden_size
              + hidden_size + output_size * hidden_size + output_size)

# GRU : 3 * (W_x (H*d) + W_h (H*H) + b (H)) + W_hy (o*H) + b_y (o)
gru_params = 3 * (hidden_size * input_size + hidden_size * hidden_size
                  + hidden_size) + output_size * hidden_size + output_size

print(f"RNN : {rnn_params} paramètres")
print(f"GRU : {gru_params} paramètres")
print(f"Ratio GRU/RNN : {gru_params / rnn_params:.2f}x")
print("\nLe GRU a environ 3x plus de paramètres que le RNN simple")
print("(3 ensembles de poids pour les portes z, r et le candidat h_tilde).")
