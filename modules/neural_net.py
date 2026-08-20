"""
neural_net.py
-------------
Lightweight feedforward neural network (2 → 6 → 1) built from scratch
using only NumPy. No PyTorch, no Keras, no external ML libraries.

Architecture
------------
  Input  layer : 2 neurons  — x(t-1), x(t)
  Hidden layer : 6 neurons  — sigmoid activation
  Output layer : 1 neuron   — sigmoid activation → predicted x(t+1)

Training
--------
  Loss      : Mean Squared Error (MSE)
  Optimiser : Stochastic Gradient Descent (SGD) with backpropagation
"""

import numpy as np


class NeuralNetwork:
    def __init__(self, lr: float = 0.01, seed: int = 42):
        rng = np.random.default_rng(seed)
        # Xavier initialisation for stable sigmoid gradients
        self.W1 = rng.normal(0, np.sqrt(2 / 2), (6, 2))   # (hidden, input)
        self.b1 = np.zeros((6, 1))
        self.W2 = rng.normal(0, np.sqrt(2 / 6), (1, 6))   # (output, hidden)
        self.b2 = np.zeros((1, 1))
        self.lr = lr
        self.loss_history: list[float] = []

    # ── Activation ────────────────────────────────────────────────────────
    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(z, -15, 15)))

    # ── Forward pass ──────────────────────────────────────────────────────
    def forward(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        x : shape (2, 1)
        Returns (h, y_hat) where h is hidden activations, y_hat is output.
        """
        z1 = self.W1 @ x + self.b1          # (6,1)
        h  = self._sigmoid(z1)              # (6,1)
        z2 = self.W2 @ h  + self.b2         # (1,1)
        y  = self._sigmoid(z2)              # (1,1)
        return h, y

    # ── Backward pass (backpropagation) ───────────────────────────────────
    def _backward(self, x, h, y_hat, target):
        """Single SGD step. Returns scalar MSE loss."""
        err    = y_hat - target                           # (1,1)
        dL_dy  = 2 * err                                  # MSE gradient
        dy_dz2 = y_hat * (1 - y_hat)                     # sigmoid derivative
        delta2 = dL_dy * dy_dz2                           # (1,1)

        dh_dz1 = h * (1 - h)                             # (6,1)
        delta1 = (self.W2.T @ delta2) * dh_dz1           # (6,1)

        # Weight updates
        self.W2 -= self.lr * (delta2 @ h.T)
        self.b2 -= self.lr * delta2
        self.W1 -= self.lr * (delta1 @ x.T)
        self.b1 -= self.lr * delta1

        return float((err ** 2).item())

    # ── Training loop ─────────────────────────────────────────────────────
    def train_epoch(self, data: np.ndarray) -> float:
        """
        One full pass over the normalised time-series.
        data : 1-D array, values in [0,1]
        Returns mean MSE for this epoch.
        """
        losses = []
        for i in range(1, len(data) - 1):
            x      = np.array([[data[i - 1]], [data[i]]])
            target = np.array([[data[i + 1]]])
            h, y_hat = self.forward(x)
            loss = self._backward(x, h, y_hat, target)
            losses.append(loss)
        mean_loss = float(np.mean(losses))
        self.loss_history.append(mean_loss)
        return mean_loss

    def train(self, data: np.ndarray, epochs: int = 500,
              verbose: bool = True, log_every: int = 50) -> list[float]:
        """
        Train for `epochs` epochs.
        Prints loss at every `log_every` epochs if verbose=True.
        Returns full loss history.
        """
        print(f"  Training 2→6→1 NN for {epochs} epochs  (lr={self.lr})")
        for ep in range(1, epochs + 1):
            loss = self.train_epoch(data)
            if verbose and ep % log_every == 0:
                pred = self.predict_next(data)
                print(f"  Epoch {ep:>4d} | MSE: {loss:.5f} | "
                      f"Next-step prediction: {pred:.4f}")
        return self.loss_history

    # ── Inference ─────────────────────────────────────────────────────────
    def predict_next(self, data: np.ndarray) -> float:
        """Predict the value after the last two in `data`."""
        x = np.array([[data[-2]], [data[-1]]])
        _, y = self.forward(x)
        return float(y.item())

    def predict_series(self, data: np.ndarray) -> np.ndarray:
        """Return one-step-ahead predictions for the whole series."""
        preds = np.full(len(data), np.nan)
        for i in range(1, len(data) - 1):
            x = np.array([[data[i - 1]], [data[i]]])
            _, y = self.forward(x)
            preds[i + 1] = float(y.item())
        return preds
