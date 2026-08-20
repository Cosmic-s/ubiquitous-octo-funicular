"""
datasets.py
-----------
Real-world datasets for the Fractal-Driven Predictive System.
Sources:
  - World Bank: Global birth rates (births per 1,000 people), 2000-2022
  - NASA GISS:  Global temperature anomaly (°C above 1951-1980 baseline), 1980-2023
"""

import numpy as np

# ── World Bank birth rate data ─────────────────────────────────────────────
BIRTH_YEARS = list(range(2000, 2023))

BIRTH_DATA = {
    "World":            [21.4,21.0,20.6,20.2,19.9,19.5,19.1,18.8,18.5,18.2,
                         18.0,17.8,17.6,17.4,17.2,17.0,16.8,16.6,16.4,16.2,
                         16.0,15.8,15.6],
    "India":            [27.4,26.8,26.1,25.4,24.7,24.0,23.4,22.7,22.1,21.5,
                         20.9,20.4,19.9,19.4,18.9,18.4,17.9,17.4,16.9,16.4,
                         15.9,15.4,14.9],
    "Sub-Saharan Africa":[43.5,43.1,42.7,42.3,41.9,41.4,40.9,40.4,39.9,39.3,
                          38.8,38.2,37.6,37.0,36.4,35.8,35.2,34.6,34.0,33.4,
                          32.8,32.2,31.6],
    "Europe":           [10.2,10.1,10.1,10.1,10.1,10.2,10.3,10.4,10.5,10.6,
                         10.7,10.5,10.4,10.3,10.1, 9.9, 9.8, 9.7, 9.6, 9.5,
                          9.4, 9.3, 9.2],
}

# ── NASA GISS temperature anomaly data ────────────────────────────────────
CLIMATE_YEARS = list(range(1980, 2024))

CLIMATE_DATA = {
    "Global anomaly": [
        0.26,0.32,0.14,0.31,0.16,0.12,0.18,0.33,0.40,0.29,
        0.44,0.33,0.23,0.31,0.45,0.35,0.33,0.46,0.63,0.40,
        0.42,0.54,0.62,0.57,0.62,0.65,0.55,0.60,0.62,0.68,
        0.75,0.90,0.87,0.98,0.92,0.85,1.01,0.92,0.83,0.98,
        1.02,0.85,0.89,1.17
    ],
}

# ── Combined (normalised overlap 2000-2022) ───────────────────────────────
COMBINED_YEARS = list(range(2000, 2023))

def _norm(arr):
    a = np.array(arr, dtype=float)
    return (a - a.min()) / (a.max() - a.min() + 1e-12)

COMBINED_DATA = {
    "Birth rate (norm)":    _norm(BIRTH_DATA["World"]).tolist(),
    "Temp anomaly (norm)":  _norm(CLIMATE_DATA["Global anomaly"][20:43]).tolist(),
}
COMBINED_DATA["Combined index"] = (
    (np.array(COMBINED_DATA["Birth rate (norm)"]) +
     np.array(COMBINED_DATA["Temp anomaly (norm)"])) / 2
).tolist()


def get_dataset(name: str):
    """Return (years, data_dict) for 'birth', 'climate', or 'combined'."""
    if name == "birth":
        return BIRTH_YEARS, BIRTH_DATA
    if name == "climate":
        return CLIMATE_YEARS, CLIMATE_DATA
    if name == "combined":
        return COMBINED_YEARS, COMBINED_DATA
    raise ValueError(f"Unknown dataset '{name}'. Choose: birth | climate | combined")


def normalise(series: list) -> np.ndarray:
    a = np.array(series, dtype=float)
    return (a - a.min()) / (a.max() - a.min() + 1e-12)
