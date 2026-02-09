"""
Vérification JAX/Flax du GRU from scratch.

À COMPLÉTER : fonction `jax_gru_forward`.
"""
import jax
import jax.numpy as jnp
import flax.linen as nn


class FlaxGRUCell(nn.Module):
    """
    Cellule GRU Flax qui reproduit exactement notre implémentation.
    """
    hidden_size: int

    @nn.compact
    def __call__(self, carry, x):
        h_prev = carry

        # Porte de mise à jour (z)
        z = jax.nn.sigmoid(
            nn.Dense(self.hidden_size, use_bias=False, name="W_xz")(x)
            + nn.Dense(self.hidden_size, use_bias=True, name="W_hz")(h_prev)
        )

        # Porte de réinitialisation (r)
        r = jax.nn.sigmoid(
            nn.Dense(self.hidden_size, use_bias=False, name="W_xr")(x)
            + nn.Dense(self.hidden_size, use_bias=True, name="W_hr")(h_prev)
        )

        # Candidat
        h_tilde = jnp.tanh(
            nn.Dense(self.hidden_size, use_bias=False, name="W_xh")(x)
            + nn.Dense(self.hidden_size, use_bias=True, name="W_hh")(r * h_prev)
        )

        # Nouvel état caché
        h_new = (1 - z) * h_prev + z * h_tilde
        return h_new, h_new  # (new_carry, output)


class FlaxSimpleGRU(nn.Module):
    """GRU complet en Flax, utilisant nn.scan sur FlaxGRUCell."""
    hidden_size: int

    @nn.compact
    def __call__(self, x):
        """x: (seq_len, input_size) — pas de dimension batch."""
        h0 = jnp.zeros(self.hidden_size)

        scan_fn = nn.scan(
            FlaxGRUCell,
            variable_broadcast="params",
            split_rngs={"params": False},
            in_axes=0,
            out_axes=0,
        )
        scanner = scan_fn(hidden_size=self.hidden_size, name="cell")
        h_final, all_h = scanner(h0, x)
        return all_h  # (seq_len, hidden_size)


def jax_gru_forward(W_xz, W_hz, b_z, W_xr, W_hr, b_r,
                    W_xh, W_hh, b_h, X, seq_length):
    """
    Forward pass GRU en JAX pur, avec les mêmes poids que l'implémentation NumPy.

    Paramètres:
    W_xz, W_hz : numpy array (hidden_size, input_size/hidden_size)
    b_z         : numpy array (hidden_size, 1)
    W_xr, W_hr : numpy array (hidden_size, input_size/hidden_size)
    b_r         : numpy array (hidden_size, 1)
    W_xh, W_hh : numpy array (hidden_size, input_size/hidden_size)
    b_h         : numpy array (hidden_size, 1)
    X           : numpy array (seq_length, input_size, 1)
    seq_length  : int

    Retourne:
    all_h : jax array (seq_length, hidden_size)

    Indice: utilisez jax.lax.scan avec une fonction qui calcule les
            portes z, r, le candidat h_tilde et le nouvel état h_new.
    """
    # TODO: Implémentez cette fonction
    # 1. Convertir les 9 poids NumPy en jnp arrays (et flatten les biais)
    # 2. Définir scan_fn(h_prev, x_t) qui calcule un pas GRU complet
    # 3. Initialiser h0 à zéros
    # 4. Reshaper X en (seq_length, input_size)
    # 5. Appeler jax.lax.scan(scan_fn, h0, X_j)
    raise NotImplementedError("Implémentez jax_gru_forward()")
