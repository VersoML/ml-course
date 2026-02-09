"""
Vérification JAX/Flax du RNN from scratch.

À COMPLÉTER : fonction `jax_rnn_forward`.
"""
import jax
import jax.numpy as jnp
import flax.linen as nn


class FlaxRNNCell(nn.Module):
    """
    Cellule RNN simple Flax qui reproduit exactement notre implémentation.
    h_t = tanh(W_xh @ x_t + W_hh @ h_{t-1} + b_h)
    """
    hidden_size: int

    @nn.compact
    def __call__(self, carry, x):
        h_prev = carry
        h_new = jnp.tanh(
            nn.Dense(self.hidden_size, use_bias=False, name="W_xh")(x)
            + nn.Dense(self.hidden_size, use_bias=True, name="W_hh")(h_prev)
        )
        return h_new, h_new  # (new_carry, output)


class FlaxSimpleRNN(nn.Module):
    """RNN simple complet en Flax, utilisant nn.scan sur FlaxRNNCell."""
    hidden_size: int

    @nn.compact
    def __call__(self, x):
        """x: (seq_len, input_size) — pas de dimension batch."""
        h0 = jnp.zeros(self.hidden_size)

        scan_fn = nn.scan(
            FlaxRNNCell,
            variable_broadcast="params",
            split_rngs={"params": False},
            in_axes=0,
            out_axes=0,
        )
        scanner = scan_fn(hidden_size=self.hidden_size, name="cell")
        h_final, all_h = scanner(h0, x)
        return all_h  # (seq_len, hidden_size)


def jax_rnn_forward(W_xh, W_hh, b_h, X, seq_length):
    """
    Forward pass RNN en JAX pur, avec les mêmes poids que l'implémentation NumPy.

    Paramètres:
    W_xh : numpy array (hidden_size, input_size)
    W_hh : numpy array (hidden_size, hidden_size)
    b_h  : numpy array (hidden_size, 1)
    X    : numpy array (seq_length, input_size, 1)
    seq_length : int

    Retourne:
    all_h : jax array (seq_length, hidden_size)

    Indice: utilisez jax.lax.scan avec une fonction qui calcule
    """
    # TODO: Implémentez cette fonction
    # 1. Convertir les poids NumPy en jnp arrays (et flatten b_h)
    # 2. Définir scan_fn(h_prev, x_t) qui calcule un pas RNN
    # 3. Initialiser h0 à zéros
    # 4. Reshaper X en (seq_length, input_size)
    # 5. Appeler jax.lax.scan(scan_fn, h0, X_j)
    raise NotImplementedError("Implémentez jax_rnn_forward()")
