"""
cellular_automata.py
--------------------
1-D elementary cellular automaton (Wolfram).
Rule 90 produces a Sierpinski Triangle from a single active cell,
demonstrating emergence independent of the chaos game.
"""

import numpy as np


def _rule_output(rule_number: int, left: int, centre: int, right: int) -> int:
    idx = (left << 2) | (centre << 1) | right
    return (rule_number >> idx) & 1


def evolve(rule_number: int = 90,
           width: int = 201,
           generations: int = 100,
           seed_centre: bool = True) -> np.ndarray:
    """
    Evolve a 1-D cellular automaton for `generations` steps.

    Parameters
    ----------
    rule_number  : Wolfram rule (0–255)
    width        : number of cells
    generations  : number of time steps
    seed_centre  : if True start with single active centre cell;
                   if False use random initialisation

    Returns
    -------
    grid : np.ndarray of shape (generations, width), dtype uint8
           row 0 = initial state, row n = state after n steps
    """
    grid = np.zeros((generations, width), dtype=np.uint8)
    if seed_centre:
        grid[0, width // 2] = 1
    else:
        rng = np.random.default_rng(42)
        grid[0] = rng.integers(0, 2, size=width, dtype=np.uint8)

    for g in range(1, generations):
        for c in range(width):
            L = int(grid[g - 1, (c - 1) % width])
            C = int(grid[g - 1, c])
            R = int(grid[g - 1, (c + 1) % width])
            grid[g, c] = _rule_output(rule_number, L, C, R)

    return grid


def active_cell_counts(grid: np.ndarray) -> np.ndarray:
    """Return number of active cells per generation."""
    return grid.sum(axis=1)


def fractal_dimension_estimate(grid: np.ndarray) -> float:
    """
    Rough box-counting estimate of fractal dimension for a binary grid.
    Compare to theoretical 1.585 for Sierpinski.
    """
    counts = []
    sizes  = []
    g = grid.astype(float)
    for scale in [1, 2, 4, 8, 16]:
        h, w = g.shape
        bh, bw = h // scale, w // scale
        if bh < 1 or bw < 1:
            break
        resized = g[:bh*scale, :bw*scale].reshape(bh, scale, bw, scale)
        boxes   = (resized.sum(axis=(1, 3)) > 0).sum()
        counts.append(np.log(boxes + 1e-9))
        sizes.append(np.log(1.0 / scale))
    if len(counts) < 2:
        return float('nan')
    coeffs = np.polyfit(sizes, counts, 1)
    return float(abs(coeffs[0]))
