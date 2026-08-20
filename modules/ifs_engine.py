"""
ifs_engine.py
-------------
Iterated Function System (IFS) fractal engine.

Implements:
  - Pure IFS (Sierpinski Triangle attractor)
  - Data-anchored IFS  (real-world data bends the attractor)
  - NN-feedback IFS    (neural prediction closes the loop)
  - Chaos Game         (classic Sierpinski construction)
  - Lorenz Attractor   (deterministic chaos, 2-trajectory divergence)
"""

import numpy as np


# ── IFS transformations (Sierpinski Triangle) ─────────────────────────────
_IFS_TRANSFORMS = [
    dict(a=0.5, b=0.0, c=0.0, d=0.5, e=0.00,  f=0.000),
    dict(a=0.5, b=0.0, c=0.0, d=0.5, e=0.50,  f=0.000),
    dict(a=0.5, b=0.0, c=0.0, d=0.5, e=0.25,  f=0.433),
]
_IFS_PROBS = np.array([1/3, 1/3, 1/3])


def run_chaos_game(n_points: int = 50_000,
                   seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """
    Classic chaos game → Sierpinski Triangle.
    Returns (x_coords, y_coords), both in [0,1].
    """
    rng = np.random.default_rng(seed)
    vertices = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3)/2]])
    x, y = 0.5, 0.5
    xs, ys = np.empty(n_points), np.empty(n_points)
    choices = rng.integers(0, 3, size=n_points)
    for i, v in enumerate(choices):
        x = (x + vertices[v, 0]) / 2
        y = (y + vertices[v, 1]) / 2
        xs[i], ys[i] = x, y
    return xs, ys


class IFSPredictor:
    """
    Data-anchored IFS with optional neural network feedback.

    Parameters
    ----------
    n_particles  : number of particles tracked simultaneously
    noise        : stochastic perturbation magnitude [0, 1]
    data_weight  : how strongly real-world anchors bend the attractor [0, 1]
    nn_weight    : how strongly the NN prediction bends the attractor [0, 1]
    seed         : RNG seed for reproducibility
    """

    def __init__(self, n_particles: int = 200,
                 noise: float = 0.02,
                 data_weight: float = 0.30,
                 nn_weight: float = 0.25,
                 seed: int = 0):
        self.rng         = np.random.default_rng(seed)
        self.n_particles = n_particles
        self.noise       = noise
        self.data_weight = data_weight
        self.nn_weight   = nn_weight

        # Particle state: shape (n_particles, 2)
        self.particles   = self.rng.random((n_particles, 2))
        self.particle_age = np.zeros(n_particles, dtype=int)

        # External signals
        self.anchors: np.ndarray | None = None   # shape (M, 2)
        self.nn_anchor: np.ndarray      = np.array([0.5, 0.5])

    # ── Configuration ─────────────────────────────────────────────────────
    def set_data_anchors(self, normalised_series: np.ndarray):
        """
        Map a normalised time-series into IFS coordinate space.
        Each time step becomes one anchor point.
        """
        n = len(normalised_series)
        self.anchors = np.column_stack([
            normalised_series,
            np.linspace(0, 1, n)
        ])  # shape (n, 2)  — x=data value, y=time position

    def set_nn_anchor(self, prediction: float):
        """Update the neural network prediction anchor."""
        self.nn_anchor = np.array([prediction, prediction])

    # ── Core step ─────────────────────────────────────────────────────────
    def _apply_ifs(self, pos: np.ndarray) -> np.ndarray:
        """Apply one random IFS transformation to a batch of positions."""
        n = len(pos)
        choices = self.rng.choice(3, size=n, p=_IFS_PROBS)
        out = np.empty_like(pos)
        for i, t in enumerate(_IFS_TRANSFORMS):
            mask = choices == i
            out[mask, 0] = t['a'] * pos[mask, 0] + t['b'] * pos[mask, 1] + t['e']
            out[mask, 1] = t['c'] * pos[mask, 0] + t['d'] * pos[mask, 1] + t['f']
        return out

    def step(self, n_steps: int = 1) -> np.ndarray:
        """
        Advance all particles by n_steps.
        Returns current particle positions, shape (n_particles, 2).
        """
        for _ in range(n_steps):
            p = self._apply_ifs(self.particles)

            # Noise
            if self.noise > 0:
                p += self.rng.normal(0, self.noise * 0.05, p.shape)

            # Data anchors
            if self.anchors is not None and self.data_weight > 0:
                idx = self.rng.integers(0, len(self.anchors), size=self.n_particles)
                p = (p * (1 - self.data_weight * 0.35) +
                     self.anchors[idx] * self.data_weight * 0.35)

            # Neural network anchor
            if self.nn_weight > 0:
                p = p * (1 - self.nn_weight * 0.35) + self.nn_anchor * self.nn_weight * 0.35

            self.particles = np.clip(p, 0, 1)
            self.particle_age += 1

        return self.particles.copy()

    # ── Analytics ─────────────────────────────────────────────────────────
    @property
    def convergence(self) -> float:
        """Fraction of particles that have converged (age > 50)."""
        return float((self.particle_age > 50).mean())

    @property
    def attractor_centre(self) -> np.ndarray:
        """Mean position of all particles."""
        return self.particles.mean(axis=0)

    @property
    def attractor_spread(self) -> float:
        """Std deviation of particle positions (proxy for attractor width)."""
        return float(self.particles.std())

    def reset(self, seed: int | None = None):
        rng = np.random.default_rng(seed)
        self.particles    = rng.random((self.n_particles, 2))
        self.particle_age = np.zeros(self.n_particles, dtype=int)


# ── Lorenz Attractor ──────────────────────────────────────────────────────

def lorenz_trajectory(x0: float = 0.1, y0: float = 0.0, z0: float = 0.0,
                      sigma: float = 10.0, rho: float = 28.0,
                      beta: float = 8/3, dt: float = 0.005,
                      n_steps: int = 10_000
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Integrate the Lorenz system. Returns (x, y, z) arrays."""
    xs = np.empty(n_steps); ys = np.empty(n_steps); zs = np.empty(n_steps)
    x, y, z = x0, y0, z0
    for i in range(n_steps):
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        x += dx * dt; y += dy * dt; z += dz * dt
        xs[i], ys[i], zs[i] = x, y, z
    return xs, ys, zs


def lorenz_divergence(epsilon: float = 1e-4,
                      n_steps: int = 10_000,
                      **kwargs) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Two Lorenz trajectories with initial separation `epsilon`.
    Returns (trajectory_A_xz, trajectory_B_xz, divergence_over_time).
    """
    xA, yA, zA = lorenz_trajectory(x0=0.1,           **kwargs, n_steps=n_steps)
    xB, yB, zB = lorenz_trajectory(x0=0.1 + epsilon, **kwargs, n_steps=n_steps)
    div = np.sqrt((xA - xB)**2 + (yA - yB)**2 + (zA - zB)**2)
    trajA = np.column_stack([xA, zA])
    trajB = np.column_stack([xB, zB])
    return trajA, trajB, div
