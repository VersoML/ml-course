"""
Implémentation d'un RNN simple from scratch avec NumPy.

Classe SimpleRNN : forward pass, BPTT, mise à jour des paramètres.

À COMPLÉTER : `__init__`, `step` et `forward`.
"""
import numpy as np


class SimpleRNN:
    def __init__(self, input_size, hidden_size, output_size):
        """
        Initialise les poids du RNN avec initialisation Xavier.

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

        # TODO: Poids et biais
        # W_xh : (hidden_size, input_size)
        # W_hh : (hidden_size, hidden_size)
        # W_hy : (output_size, hidden_size)
        # b_h  : (hidden_size, 1) — initialisé à zéros
        # b_y  : (output_size, 1) — initialisé à zéros
        self.W_xh = ...
        self.W_hh = ...
        self.W_hy = ...
        self.b_h = ...
        self.b_y = ...

    def step(self, x_t, h_prev):
        """
        Calcule un seul pas de temps du RNN.

        Paramètres:
        x_t    : (input_size, 1)
        h_prev : (hidden_size, 1)

        Retourne:
        h_t : (hidden_size, 1) - nouvel état caché
        y_t : (output_size, 1) - sortie
        """
        # TODO: Implémentez cette méthode (2 lignes)
        raise NotImplementedError("Implémentez step()")

    def forward(self, X):
        """
        Déroule le RNN sur une séquence complète.

        Paramètres:
        X : (T, input_size, 1) - séquence d'entrées

        Retourne:
        outputs : (T, output_size, 1) - toutes les sorties
        hiddens : (T+1, hidden_size, 1) - tous les états cachés
                  hiddens[0] = h_0 = zéros

        Algorithme:
        1. Initialiser h_0 à zéros
        2. Pour chaque pas t : appeler self.step(X[t], hiddens[t])
        3. Stocker h_t et y_t
        """
        # TODO: Implémentez cette méthode (~10 lignes)
        raise NotImplementedError("Implémentez forward()")

    # ================================================================
    # Les méthodes ci-dessous sont DÉJÀ IMPLÉMENTÉES.
    # Vous n'avez pas besoin de les modifier.
    # ================================================================

    def backward(self, X, outputs, hiddens, targets, clip_value=5.0):
        """
        BPTT : calcule les gradients de tous les paramètres.
        (Déjà implémenté — ne pas modifier.)
        """
        T = X.shape[0]

        dW_xh = np.zeros_like(self.W_xh)
        dW_hh = np.zeros_like(self.W_hh)
        dW_hy = np.zeros_like(self.W_hy)
        db_h = np.zeros_like(self.b_h)
        db_y = np.zeros_like(self.b_y)

        delta_next = np.zeros((self.hidden_size, 1))

        for t in reversed(range(T)):
            dy = outputs[t] - targets[t]
            dW_hy += dy @ hiddens[t + 1].T
            db_y += dy

            dh = self.W_hy.T @ dy + self.W_hh.T @ delta_next
            delta_t = dh * (1 - hiddens[t + 1] ** 2)

            dW_xh += delta_t @ X[t].T
            dW_hh += delta_t @ hiddens[t].T
            db_h += delta_t

            delta_next = delta_t

        grads = {
            "dW_xh": dW_xh,
            "dW_hh": dW_hh,
            "dW_hy": dW_hy,
            "db_h": db_h,
            "db_y": db_y,
        }
        for key in grads:
            np.clip(grads[key], -clip_value, clip_value, out=grads[key])

        return grads

    def update_params(self, grads, lr):
        """Met à jour les poids par descente de gradient. (Déjà implémenté.)"""
        self.W_xh -= lr * grads["dW_xh"]
        self.W_hh -= lr * grads["dW_hh"]
        self.W_hy -= lr * grads["dW_hy"]
        self.b_h -= lr * grads["db_h"]
        self.b_y -= lr * grads["db_y"]


def numerical_gradient(rnn, X, targets, param_name, eps=1e-5):
    """Calcule le gradient numérique par différences finies. (Déjà implémenté.)"""
    param = getattr(rnn, param_name)
    grad = np.zeros_like(param)
    for i in range(param.shape[0]):
        for j in range(param.shape[1]):
            param[i, j] += eps
            out_p, _ = rnn.forward(X)
            loss_p = 0.5 * np.sum((out_p - targets) ** 2)
            param[i, j] -= 2 * eps
            out_m, _ = rnn.forward(X)
            loss_m = 0.5 * np.sum((out_m - targets) ** 2)
            param[i, j] += eps
            grad[i, j] = (loss_p - loss_m) / (2 * eps)
    return grad
