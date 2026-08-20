"""
csv_pipeline.py
===============
Fractal-Driven Predictive System — CSV Input Version
=====================================================

Feed ANY CSV file into the full Phase 1-5 pipeline.
Works identically in:
  - Jupyter Notebook  (run cell by cell)
  - Google Colab      (upload CSV then run)
  - Terminal          (python csv_pipeline.py --csv data/birth_rate.csv --col world_avg)

Usage examples
--------------
# From terminal:
    python csv_pipeline.py --csv data/birth_rate.csv     --col world_avg
    python csv_pipeline.py --csv data/climate_anomaly.csv --col global_anomaly_c
    python csv_pipeline.py --csv data/conflict_data.csv  --col active_conflicts
    python csv_pipeline.py --csv data/template_custom.csv --col value

# From Jupyter / Colab — just call:
    results = run_csv_pipeline("data/birth_rate.csv", value_col="world_avg")
    results = run_csv_pipeline("data/conflict_data.csv", value_col="active_conflicts")
"""

import sys
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # change to "inline" inside Jupyter/Colab
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Add modules to path (works from any working directory) ─────────────────
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE / "modules"))

from neural_net import NeuralNetwork
from ifs_engine import IFSPredictor, run_chaos_game

OUTPUT = _HERE / "outputs"
OUTPUT.mkdir(exist_ok=True)

# ── Colours ────────────────────────────────────────────────────────────────
PALETTE = ["#1D9E75","#D85A30","#3B8BD4","#7F77DD",
           "#EF9F27","#C0392B","#2980B9","#E67E22"]

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "#F8F7F4",
    "axes.edgecolor":   "#BDC3C7",
    "axes.grid":        True,
    "grid.color":       "#E8E7E0",
    "grid.linewidth":   0.5,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "font.family":      "sans-serif",
    "font.size":        11,
})


# ══════════════════════════════════════════════════════════════════════════
#  STEP 1 — CSV LOADER
# ══════════════════════════════════════════════════════════════════════════
def load_csv(filepath: str,
             year_col:  str = "year",
             value_col: str = None) -> tuple[np.ndarray, np.ndarray, str, pd.DataFrame]:
    """
    Load a CSV file and return (years, values, value_column_name, full_df).

    Parameters
    ----------
    filepath  : path to your CSV file
    year_col  : name of the column containing the year/time index
                (default: 'year')
    value_col : name of the column you want to predict
                (if None: auto-picks the first numeric non-year column)

    Returns
    -------
    years     : np.ndarray of ints
    values    : np.ndarray of floats  (NaN rows dropped)
    col_name  : str  (which column was used)
    df        : full DataFrame  (for inspection)
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(
            f"CSV not found: {filepath}\n"
            f"Expected location: {path.resolve()}"
        )

    df = pd.read_csv(filepath)
    print(f"\n  Loaded: {path.name}")
    print(f"  Shape : {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"  Cols  : {list(df.columns)}")

    # ── Resolve year column ────────────────────────────────────────────────
    if year_col not in df.columns:
        # Try common alternatives
        for candidate in ["Year","YEAR","date","Date","time","Time","index"]:
            if candidate in df.columns:
                year_col = candidate
                break
        else:
            # Use integer index as year
            df["year"] = range(len(df))
            year_col   = "year"
            print("  ⚠  No year column found — using row index as time")

    # ── Resolve value column ───────────────────────────────────────────────
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                    if c != year_col]

    if value_col is None:
        value_col = numeric_cols[0]
        print(f"  ℹ  No value_col specified — auto-selected: '{value_col}'")
    elif value_col not in df.columns:
        raise ValueError(
            f"Column '{value_col}' not found.\n"
            f"Available numeric columns: {numeric_cols}"
        )

    # ── Extract and clean ──────────────────────────────────────────────────
    sub = df[[year_col, value_col]].dropna()
    years  = sub[year_col].astype(int).values
    values = sub[value_col].astype(float).values

    print(f"  Column: '{value_col}'")
    print(f"  Years : {years[0]} – {years[-1]}  ({len(years)} data points)")
    print(f"  Range : [{values.min():.3f}, {values.max():.3f}]")

    if len(values) < 6:
        raise ValueError(
            f"Only {len(values)} valid data points after dropping NaN. "
            f"Need at least 6."
        )

    return years, values, value_col, df


# ══════════════════════════════════════════════════════════════════════════
#  STEP 2 — NORMALISE
# ══════════════════════════════════════════════════════════════════════════
def normalise(arr: np.ndarray) -> np.ndarray:
    mn, mx = arr.min(), arr.max()
    return (arr - mn) / (mx - mn + 1e-12)


def denormalise(normed: np.ndarray,
                original: np.ndarray) -> np.ndarray:
    mn, mx = original.min(), original.max()
    return normed * (mx - mn) + mn


# ══════════════════════════════════════════════════════════════════════════
#  STEP 3 — TRAIN NEURAL NETWORK
# ══════════════════════════════════════════════════════════════════════════
def train_neural_network(normed: np.ndarray,
                         epochs: int = 600,
                         lr: float   = 0.01,
                         verbose: bool = True) -> tuple:
    """
    Train 2→6→1 feedforward NN on the normalised series.
    Returns (nn, loss_history, pred_series, next_pred_norm).
    """
    nn = NeuralNetwork(lr=lr, seed=42)
    print(f"\n  Training neural network (2→6→1)  "
          f"epochs={epochs}  lr={lr}")
    loss_hist   = nn.train(normed, epochs=epochs,
                           verbose=verbose, log_every=max(1, epochs//6))
    pred_series = nn.predict_series(normed)
    next_pred   = nn.predict_next(normed)

    print(f"\n  Final MSE loss : {loss_hist[-1]:.5f}"
          + (" ✓ good" if loss_hist[-1] < 0.01 else " (consider more epochs)"))
    print(f"  Next-step pred : {next_pred:.4f}  (normalised)")
    return nn, loss_hist, pred_series, next_pred


# ══════════════════════════════════════════════════════════════════════════
#  STEP 4 — RUN IFS + FEEDBACK
# ══════════════════════════════════════════════════════════════════════════
def run_ifs_feedback(normed: np.ndarray,
                     nn_pred: float,
                     n_particles:  int   = 250,
                     n_steps:      int   = 250,
                     noise:        float = 0.02,
                     data_weight:  float = 0.50,
                     nn_weight:    float = 0.35) -> IFSPredictor:
    """Run the IFS particle system anchored to real data + NN prediction."""
    ifs = IFSPredictor(n_particles=n_particles, noise=noise,
                       data_weight=data_weight, nn_weight=nn_weight, seed=7)
    ifs.set_data_anchors(normed)
    ifs.set_nn_anchor(nn_pred)
    for _ in range(n_steps):
        ifs.step()

    print(f"\n  IFS convergence : {ifs.convergence:.0%}")
    print(f"  Attractor centre: {ifs.attractor_centre.round(3)}")
    print(f"  Attractor spread: {ifs.attractor_spread:.4f}"
          + ("  (tight — confident)" if ifs.attractor_spread < 0.15
             else "  (wide — uncertain)"))
    return ifs


# ══════════════════════════════════════════════════════════════════════════
#  STEP 5 — FORWARD PROJECTION
# ══════════════════════════════════════════════════════════════════════════
def project_forward(nn: NeuralNetwork,
                    normed: np.ndarray,
                    raw: np.ndarray,
                    years: np.ndarray,
                    n_future: int = 5) -> dict:
    """
    Autoregressive forward projection for n_future steps.
    Uncertainty grows ±5% per step (compounding).
    Returns dict with arrays: years, values, lower, upper (all in raw units).
    """
    mn, mx = raw.min(), raw.max()
    window = list(normed[-2:])

    fut_years = np.array([years[-1] + i + 1 for i in range(n_future)])
    fut_norm  = []
    fut_lo    = []
    fut_hi    = []

    for step in range(n_future):
        p = nn.predict_next(np.array(window))
        u = 0.05 * (step + 1)          # ±5% per horizon step
        fut_norm.append(p)
        fut_lo.append(max(0.0, p - u))
        fut_hi.append(min(1.0, p + u))
        window = [window[-1], p]

    fut_vals = np.array(fut_norm) * (mx - mn) + mn
    fut_lo   = np.array(fut_lo)   * (mx - mn) + mn
    fut_hi   = np.array(fut_hi)   * (mx - mn) + mn

    print(f"\n  {'Year':<6} {'Projected':>12} {'Low':>10} {'High':>10}")
    print(f"  {'-'*42}")
    for yr, v, lo, hi in zip(fut_years, fut_vals, fut_lo, fut_hi):
        print(f"  {yr:<6} {v:>12.3f} {lo:>10.3f} {hi:>10.3f}")

    return {"years": fut_years, "values": fut_vals,
            "lower": fut_lo,   "upper":  fut_hi}


# ══════════════════════════════════════════════════════════════════════════
#  STEP 6 — PLOT + SAVE
# ══════════════════════════════════════════════════════════════════════════
def plot_results(years, raw, normed, value_col,
                 nn, loss_hist, pred_series,
                 ifs, projection,
                 color: str = "#1D9E75",
                 save: bool = True) -> str:
    """
    Generate 6-panel research figure.
    Returns path of saved figure (or '' if save=False).
    """
    mn, mx = raw.min(), raw.max()
    pred_denorm = np.where(
        np.isnan(pred_series),
        np.nan,
        pred_series * (mx - mn) + mn
    )

    fig = plt.figure(figsize=(18, 12))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35)
    fig.suptitle(
        f"Fractal-Driven Predictive System  —  '{value_col}'\n"
        f"Data: {years[0]}–{years[-1]}  |  Architecture: 2→6→1  |  "
        f"Final MSE: {loss_hist[-1]:.4f}",
        fontsize=13, y=1.02
    )

    # ── 1. Raw data signal ────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(years, raw, color=color, lw=1.8, label="Raw data")
    ax1.set_title(f"Input signal: '{value_col}'")
    ax1.set_xlabel("Year"); ax1.set_ylabel("Value")
    ax1.text(0.02, 0.97,
             f"n={len(years)} pts\nmin={raw.min():.2f}  max={raw.max():.2f}",
             transform=ax1.transAxes, fontsize=9,
             color="#5F5E5A", va="top")

    # ── 2. NN prediction vs actual ────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(years, raw, color=color, lw=1.8, label="Actual", zorder=3)
    valid = ~np.isnan(pred_denorm)
    ax2.plot(np.array(years)[valid], pred_denorm[valid],
             color="#EF9F27", lw=1.5, ls="--",
             label="NN one-step prediction", zorder=4)
    ax2.set_title("Neural network: prediction vs actual")
    ax2.set_xlabel("Year"); ax2.set_ylabel("Value")
    ax2.legend(fontsize=9)

    # ── 3. Forward projection ─────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(years, raw, color=color, lw=1.8, label="Historical")
    ax3.plot(np.array(years)[valid], pred_denorm[valid],
             color="#EF9F27", lw=1.2, ls="--", alpha=0.6, label="NN fit")
    ax3.plot(projection["years"], projection["values"],
             "D-", color="#2C3E50", lw=2, ms=6,
             label=f"{len(projection['years'])}-yr projection")
    ax3.fill_between(projection["years"],
                     projection["lower"], projection["upper"],
                     alpha=0.18, color="#2C3E50", label="Uncertainty band")
    ax3.axvline(years[-1], color="#888780", lw=0.8, ls=":")
    ax3.set_title("Historical + forward projection")
    ax3.set_xlabel("Year"); ax3.set_ylabel("Value")
    ax3.legend(fontsize=9)

    # ── 4. IFS attractor space ────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    pts  = ifs.particles
    conv = ifs.particle_age > 50
    ax4.scatter(pts[~conv, 0], pts[~conv, 1],
                s=1.5, alpha=0.4, c="#E67E22", label="Transient")
    ax4.scatter(pts[conv,  0], pts[conv,  1],
                s=1.5, alpha=0.55, c=color,    label="Converged")
    ax4.scatter(ifs.anchors[:, 0], ifs.anchors[:, 1],
                s=14, color="#7F77DD", alpha=0.5, zorder=5,
                label="Data anchors")
    ax4.scatter(*ifs.nn_anchor, s=160, color="#EF9F27",
                zorder=10, edgecolors="white", lw=0.8,
                label=f"NN anchor: {ifs.nn_anchor[0]:.3f}")
    ax4.set_title(f"IFS attractor space\n"
                  f"Conv={ifs.convergence:.0%}  "
                  f"spread={ifs.attractor_spread:.3f}")
    ax4.set_xlim(0, 1); ax4.set_ylim(0, 1)
    ax4.set_xlabel("IFS x"); ax4.set_ylabel("IFS y")
    ax4.legend(fontsize=8, markerscale=4)

    # ── 5. Training loss curve ────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.semilogy(loss_hist, color="#C0392B", lw=1.5)
    ax5.axhline(0.01, color="#888780", ls="--",
                lw=0.8, label="0.01 threshold")
    ax5.set_title("Training loss (MSE, log scale)")
    ax5.set_xlabel("Epoch"); ax5.set_ylabel("MSE")
    ax5.legend(fontsize=9)
    ax5.text(0.98, 0.97, f"Final: {loss_hist[-1]:.5f}",
             transform=ax5.transAxes, fontsize=9,
             color="#C0392B", va="top", ha="right")

    # ── 6. Summary stats table ────────────────────────────────────────────
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis("off")
    next_denorm = ifs.nn_anchor[0] * (mx - mn) + mn
    rows = [
        ["Metric",              "Value"],
        ["Dataset column",      value_col],
        ["Years",               f"{years[0]} – {years[-1]}"],
        ["Data points",         str(len(years))],
        ["Value range",         f"{raw.min():.3f} – {raw.max():.3f}"],
        ["NN architecture",     "2 → 6 → 1"],
        ["Final MSE",           f"{loss_hist[-1]:.5f}"],
        ["Next-step (raw)",     f"{next_denorm:.3f}"],
        ["IFS convergence",     f"{ifs.convergence:.0%}"],
        ["Attractor spread",    f"{ifs.attractor_spread:.4f}"],
        ["Attractor centre",    str(ifs.attractor_centre.round(3))],
        [f"Year {projection['years'][0]} proj.",
                                f"{projection['values'][0]:.3f}"],
        [f"Year {projection['years'][-1]} proj.",
                                f"{projection['values'][-1]:.3f}"],
        ["Uncertainty/step",    "±5% (compounding)"],
    ]
    tbl = ax6.table(cellText=rows[1:], colLabels=rows[0],
                    loc="center", cellLoc="left")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.28)
    ax6.set_title("Summary statistics", pad=10)

    plt.tight_layout()

    if save:
        safe_name = value_col.replace(" ", "_").replace("/", "_")
        path = OUTPUT / f"csv_pipeline_{safe_name}.png"
        fig.savefig(path, dpi=170, bbox_inches="tight")
        plt.close(fig)
        print(f"\n  Figure saved → {path}")
        return str(path)
    else:
        plt.show()
        return ""


# ══════════════════════════════════════════════════════════════════════════
#  ALSO: save projection to CSV
# ══════════════════════════════════════════════════════════════════════════
def save_projection_csv(projection: dict,
                        value_col: str,
                        years: np.ndarray,
                        raw: np.ndarray) -> str:
    """Save the forward projection back to a CSV file."""
    df_out = pd.DataFrame({
        "year":           projection["years"],
        "projected":      projection["values"].round(4),
        "lower_bound":    projection["lower"].round(4),
        "upper_bound":    projection["upper"].round(4),
        "source_column":  value_col,
        "last_known_year": years[-1],
        "last_known_value": raw[-1],
    })
    safe_name = value_col.replace(" ", "_").replace("/", "_")
    out_path  = OUTPUT / f"projection_{safe_name}.csv"
    df_out.to_csv(out_path, index=False)
    print(f"  Projection CSV saved → {out_path}")
    return str(out_path)


# ══════════════════════════════════════════════════════════════════════════
#  MAIN PIPELINE FUNCTION  (call this from Jupyter/Colab)
# ══════════════════════════════════════════════════════════════════════════
def run_csv_pipeline(csv_path:    str,
                     value_col:   str   = None,
                     year_col:    str   = "year",
                     epochs:      int   = 600,
                     lr:          float = 0.01,
                     n_future:    int   = 5,
                     n_particles: int   = 250,
                     n_steps:     int   = 250,
                     noise:       float = 0.02,
                     data_weight: float = 0.50,
                     nn_weight:   float = 0.35,
                     color:       str   = None,
                     save_fig:    bool  = True,
                     save_proj:   bool  = True) -> dict:
    """
    Full pipeline: CSV → Normalise → Train NN → IFS → Project → Plot.

    Parameters
    ----------
    csv_path    : path to your CSV file (relative or absolute)
    value_col   : column name to predict (auto-detects if None)
    year_col    : column name for the time axis  (default: 'year')
    epochs      : NN training epochs             (default: 600)
    lr          : learning rate                  (default: 0.01)
    n_future    : years to project forward       (default: 5)
    n_particles : IFS particle count             (default: 250)
    n_steps     : IFS evolution steps            (default: 250)
    noise       : IFS noise level                (default: 0.02)
    data_weight : IFS data anchor pull           (default: 0.50)
    nn_weight   : IFS NN anchor pull             (default: 0.35)
    color       : plot colour hex string         (auto if None)
    save_fig    : save figure to outputs/        (default: True)
    save_proj   : save projection CSV            (default: True)

    Returns
    -------
    dict with keys: years, values, lower, upper, nn, ifs, loss_history
    """
    print("\n" + "="*60)
    print("  FRACTAL-DRIVEN PREDICTIVE SYSTEM  —  CSV Pipeline")
    print("="*60)

    # Pick a colour if not specified
    if color is None:
        import hashlib
        idx   = int(hashlib.md5(csv_path.encode()).hexdigest(), 16) % len(PALETTE)
        color = PALETTE[idx]

    # ── Steps ─────────────────────────────────────────────────────────────
    years, raw, col_used, df = load_csv(csv_path, year_col, value_col)
    normed = normalise(raw)

    nn, loss_hist, pred_series, next_pred = train_neural_network(
        normed, epochs=epochs, lr=lr
    )

    ifs = run_ifs_feedback(
        normed, next_pred,
        n_particles=n_particles, n_steps=n_steps,
        noise=noise, data_weight=data_weight, nn_weight=nn_weight
    )

    projection = project_forward(nn, normed, raw, years, n_future=n_future)

    fig_path  = plot_results(
        years, raw, normed, col_used,
        nn, loss_hist, pred_series,
        ifs, projection,
        color=color, save=save_fig
    )

    proj_path = ""
    if save_proj:
        proj_path = save_projection_csv(projection, col_used, years, raw)

    print("\n  DONE.")
    print(f"  Figure     : {fig_path or 'shown inline'}")
    print(f"  Projection : {proj_path or 'not saved'}")
    print("="*60)

    return {
        "projection":   projection,
        "nn":           nn,
        "ifs":          ifs,
        "loss_history": loss_hist,
        "years":        years,
        "raw":          raw,
        "column":       col_used,
    }


# ══════════════════════════════════════════════════════════════════════════
#  TERMINAL ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fractal-Driven Predictive System — CSV Input Pipeline"
    )
    parser.add_argument("--csv",         required=True,
                        help="Path to input CSV file")
    parser.add_argument("--col",         default=None,
                        help="Column name to predict (auto-detects if omitted)")
    parser.add_argument("--year-col",    default="year",
                        help="Column name for the time axis (default: year)")
    parser.add_argument("--epochs",      type=int,   default=600)
    parser.add_argument("--lr",          type=float, default=0.01)
    parser.add_argument("--future",      type=int,   default=5,
                        help="Years to project forward")
    parser.add_argument("--particles",   type=int,   default=250)
    parser.add_argument("--color",       default=None,
                        help="Hex colour for plots e.g. #1D9E75")
    args = parser.parse_args()

    run_csv_pipeline(
        csv_path    = args.csv,
        value_col   = args.col,
        year_col    = args.year_col,
        epochs      = args.epochs,
        lr          = args.lr,
        n_future    = args.future,
        n_particles = args.particles,
        color       = args.color,
    )
