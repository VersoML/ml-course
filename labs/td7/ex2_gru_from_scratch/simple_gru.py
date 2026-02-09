"""
Implémentation d'un GRU from scratch avec NumPy.

Classe SimpleGRU : forward pass, BPTT, mise à jour des paramètres.

À COMPLÉTER : `__init__`, `step` et `forward`.
"""
import numpy as np


class SimpleGRU:
    def __init__(self, input_size, hidden_size, output_size):
        """
        Initialise les poids du GRU avec initialisation Xavier.

        Paramètres:
        input_size  : int - dimension d'entrée (d)
        hidden_size : int - dimension cachée (H)
        output_size : int - dimension de sortie (o)
        """
        self.hidden_size = hidden_size

        # TODO: Initialisation Xavier
        # scale = sqrt(2 / (fan_in + fan_out))
        scale_xh = ...
        scale_hh = ...
        scale_hy = ...

        # TODO: Porte de mise à jour (z)
        # W_xz : (hidden_size, input_size)
        # W_hz : (hidden_size, hidden_size)
        # b_z  : (hidden_size, 1) — initialisé à zéros
        self.W_xz = ...
        self.W_hz = ...
        self.b_z = ...

        # TODO: Porte de réinitialisation (r)
        self.W_xr = ...
        self.W_hr = ...
        self.b_r = ...

        # TODO: Candidat (h_tilde)
        self.W_xh = ...
        self.W_hh = ...
        self.b_h = ...

        # TODO: Sortie
        # W_hy : (output_size, hidden_size)
        # b_y  : (output_size, 1) — initialisé à zéros
        self.W_hy = ...
        self.b_y = ...

    def sigmoid(self, x):
        """Sigmoïde numériquement stable."""
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    def step(self, x_t, h_prev):
        """
        Calcule un seul pas de temps du GRU.

        Paramètres:
        x_t    : (input_size, 1)
        h_prev : (hidden_size, 1)

        Retourne:
        h_t     : (hidden_size, 1) - nouvel état caché
        y_t     : (output_size, 1) - sortie
        z_t     : (hidden_size, 1) - porte de mise à jour
        r_t     : (hidden_size, 1) - porte de réinitialisation
        h_tilde : (hidden_size, 1) - candidat


        """
        # TODO: Implémentez cette méthode (5 lignes)
        raise NotImplementedError("Implémentez step()")

    def forward(self, X):
        """
        Déroule le GRU sur une séquence complète.

        Paramètres:
        X : (T, input_size, 1) - séquence d'entrées

        Retourne:
        outputs : (T, output_size, 1) - toutes les sorties
        hiddens : (T+1, hidden_size, 1) - tous les états cachés
                  hiddens[0] = h_0 = zéros
        cache   : dict avec :
                  'gates_z'  : (T, hidden_size, 1)
                  'gates_r'  : (T, hidden_size, 1)
                  'h_tildes' : (T, hidden_size, 1)

        """
        # TODO: Implémentez cette méthode (~15 lignes)
        raise NotImplementedError("Implémentez forward()")

    # ================================================================
    # Les méthodes ci-dessous sont DÉJÀ IMPLÉMENTÉES.
    # Vous n'avez pas besoin de les modifier.
    # ================================================================

    def backward(self, X, outputs, hiddens, cache, targets, clip_value=5.0):
        """
        BPTT pour le GRU : calcule les gradients de tous les paramètres.
        (Déjà implémenté — ne pas modifier.)
        """
        T = X.shape[0]
        gates_z = cache["gates_z"]
        gates_r = cache["gates_r"]
        h_tildes = cache["h_tildes"]

        dW_xz = np.zeros_like(self.W_xz)
        dW_hz = np.zeros_like(self.W_hz)
        db_z = np.zeros_like(self.b_z)

        dW_xr = np.zeros_like(self.W_xr)
        dW_hr = np.zeros_like(self.W_hr)
        db_r = np.zeros_like(self.b_r)

        dW_xh = np.zeros_like(self.W_xh)
        dW_hh = np.zeros_like(self.W_hh)
        db_h = np.zeros_like(self.b_h)

        dW_hy = np.zeros_like(self.W_hy)
        db_y = np.zeros_like(self.b_y)

        dh_next = np.zeros((self.hidden_size, 1))

        for t in reversed(range(T)):
            h_prev = hiddens[t]
            h_t = hiddens[t + 1]
            z_t = gates_z[t]
            r_t = gates_r[t]
            h_tilde = h_tildes[t]

            dy = outputs[t] - targets[t]
            dW_hy += dy @ h_t.T
            db_y += dy

            dh = self.W_hy.T @ dy + dh_next

            dh_tilde = dh * z_t
            dz = dh * (h_tilde - h_prev)
            dh_prev = dh * (1 - z_t)

            d_tanh = dh_tilde * (1 - h_tilde ** 2)
            dW_xh += d_tanh @ X[t].T
            dW_hh += d_tanh @ (r_t * h_prev).T
            db_h += d_tanh

            d_rh = self.W_hh.T @ d_tanh
            dr_from_htilde = d_rh * h_prev
            dh_prev += d_rh * r_t

            d_sig_z = dz * z_t * (1 - z_t)
            dW_xz += d_sig_z @ X[t].T
            dW_hz += d_sig_z @ h_prev.T
            db_z += d_sig_z
            dh_prev += self.W_hz.T @ d_sig_z

            d_sig_r = dr_from_htilde * r_t * (1 - r_t)
            dW_xr += d_sig_r @ X[t].T
            dW_hr += d_sig_r @ h_prev.T
            db_r += d_sig_r
            dh_prev += self.W_hr.T @ d_sig_r

            dh_next = dh_prev

        grads = {
            "dW_xz": dW_xz, "dW_hz": dW_hz, "db_z": db_z,
            "dW_xr": dW_xr, "dW_hr": dW_hr, "db_r": db_r,
            "dW_xh": dW_xh, "dW_hh": dW_hh, "db_h": db_h,
            "dW_hy": dW_hy, "db_y": db_y,
        }
        for key in grads:
            np.clip(grads[key], -clip_value, clip_value, out=grads[key])

        return grads

    def update_params(self, grads, lr):
        """Met à jour les poids par descente de gradient. (Déjà implémenté.)"""
        for param_name in [
            "W_xz", "W_hz", "b_z",
            "W_xr", "W_hr", "b_r",
            "W_xh", "W_hh", "b_h",
            "W_hy", "b_y",
        ]:
            param = getattr(self, param_name)
            param -= lr * grads["d" + param_name]


def numerical_gradient(gru, X, targets, param_name, eps=1e-5):
    """Calcule le gradient numérique par différences finies. (Déjà implémenté.)"""
    param = getattr(gru, param_name)
    grad = np.zeros_like(param)
    for i in range(param.shape[0]):
        for j in range(param.shape[1]):
            param[i, j] += eps
            out_p, _, _ = gru.forward(X)
            loss_p = 0.5 * np.sum((out_p - targets) ** 2)
            param[i, j] -= 2 * eps
            out_m, _, _ = gru.forward(X)
            loss_m = 0.5 * np.sum((out_m - targets) ** 2)
            param[i, j] += eps
            grad[i, j] = (loss_p - loss_m) / (2 * eps)
    return grad
