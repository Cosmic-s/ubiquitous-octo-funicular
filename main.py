"""
main.py
-------
Fractal-Driven Predictive Systems — Full Research Pipeline
==========================================================

Runs all five phases in sequence and saves publication-quality figures
to outputs/. Each phase corresponds to a section of the research proposal.

Usage
-----
    python main.py                    # run all phases
    python main.py --phase 5          # run a single phase
    python main.py --dataset climate  # choose dataset for phases 4 & 5
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

# ── Project imports ────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent / "modules"))
from datasets        import get_dataset, normalise
from ifs_engine      import run_chaos_game, IFSPredictor, lorenz_divergence
from cellular_automata import evolve, active_cell_counts, fractal_dimension_estimate
from neural_net      import NeuralNetwork

OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Colour palette (matches browser prototype) ────────────────────────────
C_TEAL   = "#1D9E75"
C_CORAL  = "#D85A30"
C_BLUE   = "#3B8BD4"
C_PURPLE = "#7F77DD"
C_AMBER  = "#EF9F27"
C_GRAY   = "#888780"

plt.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "#F8F7F3",
    "axes.edgecolor":    "#C8C7C0",
    "axes.grid":         True,
    "grid.color":        "#E8E7E0",
    "grid.linewidth":    0.5,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "font.family":       "sans-serif",
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.titleweight":  "normal",
    "axes.labelsize":    11,
    "legend.fontsize":   9,
    "legend.frameon":    False,
})


def section(title: str):
    print(f"\n{'═'*60}")
    print(f"  {title}")
    print(f"{'═'*60}")


# ══════════════════════════════════════════════════════════════════════════
# PHASE 1 — Chaos Game (Sierpinski Triangle)
# ══════════════════════════════════════════════════════════════════════════
def phase1_chaos_game():
    section("Phase 1 — Chaos Game: Sierpinski Triangle")
    t0 = time.time()

    xs, ys = run_chaos_game(n_points=80_000, seed=42)
    elapsed = time.time() - t0

    print(f"  Points generated : 80,000")
    print(f"  Time             : {elapsed:.2f}s")
    print(f"  Fractal dimension: 1.585  (log3/log2, theoretical)")
    print(f"  X range          : [{xs.min():.3f}, {xs.max():.3f}]")
    print(f"  Y range          : [{ys.min():.3f}, {ys.max():.3f}]")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Phase 1 — Chaos Game: Emergent Order from Randomness", y=1.01)

    # Full attractor
    ax = axes[0]
    hue = xs
    cmap = LinearSegmentedColormap.from_list("fp", [C_BLUE, C_TEAL, C_PURPLE])
    ax.scatter(xs, ys, s=0.05, c=hue, cmap=cmap, alpha=0.4, rasterized=True)
    ax.set_title("Sierpinski Triangle (80,000 random iterations)")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_aspect("equal")
    ax.text(0.02, 0.97, f"Fractal dim. = 1.585", transform=ax.transAxes,
            fontsize=9, color=C_GRAY, va="top")

    # Iteration convergence
    ax2 = axes[1]
    milestones = [100, 500, 1000, 5000, 10000, 50000, 80000]
    for n in milestones:
        xi, yi = xs[:n], ys[:n]
        ax2.scatter(xi, yi, s=0.2, alpha=0.3,
                    label=f"n={n:,}", rasterized=True)
    ax2.set_title("Convergence by iteration count")
    ax2.set_xlabel("x"); ax2.set_ylabel("y")
    ax2.set_aspect("equal")
    ax2.legend(markerscale=8, loc="upper right")

    plt.tight_layout()
    path = OUTPUT_DIR / "phase1_chaos_game.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


# ══════════════════════════════════════════════════════════════════════════
# PHASE 2 — Cellular Automata Emergence (Rule 90)
# ══════════════════════════════════════════════════════════════════════════
def phase2_cellular_automata():
    section("Phase 2 — Cellular Automata: Emergence")

    rules_to_show = [90, 110, 30, 18]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Phase 2 — Emergence from Local Rules (Wolfram Cellular Automata)", y=1.01)

    for ax, rule in zip(axes.flat, rules_to_show):
        grid = evolve(rule_number=rule, width=201, generations=100)
        fd   = fractal_dimension_estimate(grid)
        counts = active_cell_counts(grid)

        cmap = LinearSegmentedColormap.from_list("ca", ["white", C_TEAL if rule==90 else C_BLUE])
        ax.imshow(grid, cmap=cmap, aspect="auto", interpolation="nearest")
        ax.set_title(f"Rule {rule}  |  fractal dim ≈ {fd:.3f}"
                     + (" ← Sierpinski" if rule == 90 else ""))
        ax.set_xlabel("Cell"); ax.set_ylabel("Generation")
        ax.set_facecolor("white")

    plt.tight_layout()
    path = OUTPUT_DIR / "phase2_cellular_automata.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Rule 90 stats
    grid90  = evolve(rule_number=90, width=201, generations=100)
    counts  = active_cell_counts(grid90)
    fd90    = fractal_dimension_estimate(grid90)
    print(f"  Rule 90 fractal dim estimate : {fd90:.4f}  (theoretical: 1.585)")
    print(f"  Active cells gen 1 / gen 99  : {counts[1]} / {counts[99]}")
    print(f"  Saved → {path}")


# ══════════════════════════════════════════════════════════════════════════
# PHASE 3 — IFS Particle Predictor
# ══════════════════════════════════════════════════════════════════════════
def phase3_ifs_predictor():
    section("Phase 3 — IFS Particle Predictor")

    noise_levels = [0.0, 0.02, 0.05, 0.10]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle("Phase 3 — IFS Convergence Under Varying Noise", y=1.01)

    for ax, noise in zip(axes, noise_levels):
        ifs = IFSPredictor(n_particles=300, noise=noise,
                           data_weight=0.0, nn_weight=0.0, seed=7)
        for _ in range(200):
            ifs.step()
        pts = ifs.particles
        ax.scatter(pts[:, 0], pts[:, 1], s=2, alpha=0.5, c=C_TEAL)
        ax.set_title(f"Noise = {noise:.0%}\nConv = {ifs.convergence:.0%}")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("x"); ax.set_ylabel("y" if noise == 0.0 else "")

    plt.tight_layout()
    path = OUTPUT_DIR / "phase3_ifs_predictor.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Convergence curve
    ifs2 = IFSPredictor(n_particles=200, noise=0.02,
                        data_weight=0.0, nn_weight=0.0, seed=42)
    conv_curve = []
    for _ in range(150):
        ifs2.step()
        conv_curve.append(ifs2.convergence)

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.plot(conv_curve, color=C_TEAL, lw=1.8)
    ax2.axhline(1.0, color=C_GRAY, ls="--", lw=0.8)
    ax2.set_title("IFS Particle Convergence Over Time")
    ax2.set_xlabel("Step"); ax2.set_ylabel("Fraction converged")
    ax2.set_ylim(0, 1.05)
    plt.tight_layout()
    path2 = OUTPUT_DIR / "phase3_convergence_curve.png"
    fig2.savefig(path2, dpi=180, bbox_inches="tight")
    plt.close(fig2)

    print(f"  Final convergence (noise=2%) : {conv_curve[-1]:.2%}")
    print(f"  Saved → {path}")
    print(f"  Saved → {path2}")


# ══════════════════════════════════════════════════════════════════════════
# PHASE 4 — Real-World Data Integration
# ══════════════════════════════════════════════════════════════════════════
def phase4_real_data(dataset_name: str = "birth"):
    section(f"Phase 4 — Real-World Data Integration ({dataset_name})")

    years, data_dict = get_dataset(dataset_name)
    primary_key = list(data_dict.keys())[0]
    raw    = np.array(data_dict[primary_key])
    normed = normalise(raw)

    fig = plt.figure(figsize=(14, 10))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)
    fig.suptitle(
        f"Phase 4 — IFS Attractor with Real-World Data\n"
        f"Dataset: {primary_key}  ({years[0]}–{years[-1]})",
        y=1.02
    )

    # 1. Raw data signal
    ax1 = fig.add_subplot(gs[0, 0])
    colors = [C_TEAL, C_BLUE, C_CORAL, C_PURPLE]
    for (label, series), col in zip(data_dict.items(), colors):
        ax1.plot(years, series, label=label, color=col, lw=1.5)
    ax1.set_title("Raw data signal")
    ax1.set_xlabel("Year"); ax1.legend()

    # 2. Normalised signal with anchor points
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(years, normed, color=C_TEAL, lw=1.5, label="Normalised")
    ax2.scatter(years, normed, s=20, color=C_PURPLE, zorder=5,
                label="IFS anchors")
    ax2.set_title("Normalised → IFS anchor points")
    ax2.set_xlabel("Year")
    ax2.set_ylabel("[0, 1]")
    ax2.legend()

    # 3. IFS attractor — no data (pure)
    ax3 = fig.add_subplot(gs[1, 0])
    ifs_pure = IFSPredictor(n_particles=300, noise=0.02,
                            data_weight=0.0, nn_weight=0.0, seed=1)
    for _ in range(200):
        ifs_pure.step()
    pts = ifs_pure.particles
    ax3.scatter(pts[:, 0], pts[:, 1], s=2, alpha=0.5, c=C_GRAY)
    ax3.set_title("Pure IFS (no data influence)")
    ax3.set_xlim(0, 1); ax3.set_ylim(0, 1)

    # 4. IFS attractor — data anchored
    ax4 = fig.add_subplot(gs[1, 1])
    ifs_data = IFSPredictor(n_particles=300, noise=0.02,
                            data_weight=0.55, nn_weight=0.0, seed=1)
    ifs_data.set_data_anchors(normed)
    for _ in range(200):
        ifs_data.step()
    pts2 = ifs_data.particles
    ax4.scatter(pts2[:, 0], pts2[:, 1], s=2, alpha=0.5, c=C_TEAL)
    anchors = ifs_data.anchors
    ax4.scatter(anchors[:, 0], anchors[:, 1], s=12, color=C_PURPLE,
                alpha=0.6, label="Anchors", zorder=5)
    ax4.set_title(f"Data-anchored IFS (weight=0.55)\nConv = {ifs_data.convergence:.0%}")
    ax4.set_xlim(0, 1); ax4.set_ylim(0, 1)
    ax4.legend()

    path = OUTPUT_DIR / f"phase4_realdata_{dataset_name}.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Dataset          : {primary_key}")
    print(f"  Years            : {years[0]}–{years[-1]}")
    print(f"  Value range      : [{raw.min():.2f}, {raw.max():.2f}]")
    print(f"  IFS convergence  : {ifs_data.convergence:.2%}")
    print(f"  Attractor centre : {ifs_data.attractor_centre.round(3)}")
    print(f"  Attractor spread : {ifs_data.attractor_spread:.4f}")
    print(f"  Saved → {path}")


# ══════════════════════════════════════════════════════════════════════════
# PHASE 5 — Neural Network Feedback Layer
# ══════════════════════════════════════════════════════════════════════════
def phase5_neural_feedback(dataset_name: str = "birth"):
    section(f"Phase 5 — Neural Network Feedback Layer ({dataset_name})")

    years, data_dict = get_dataset(dataset_name)
    primary_key = list(data_dict.keys())[0]
    raw    = np.array(data_dict[primary_key])
    normed = normalise(raw)

    # ── Train neural network ──────────────────────────────────────────────
    print(f"\n  Dataset  : {primary_key}")
    nn = NeuralNetwork(lr=0.01, seed=7)
    loss_history = nn.train(normed, epochs=600, verbose=True, log_every=100)
    predictions  = nn.predict_series(normed)
    next_pred    = nn.predict_next(normed)

    final_loss = loss_history[-1]
    pred_error = abs(predictions[~np.isnan(predictions)][-1] -
                     normed[~np.isnan(predictions)][-1])

    print(f"\n  Final MSE loss   : {final_loss:.5f}")
    print(f"  Next-step pred   : {next_pred:.4f}  (normalised)")
    print(f"  Prediction error : {pred_error:.4f}")

    # ── Run IFS with NN feedback ──────────────────────────────────────────
    ifs = IFSPredictor(n_particles=250, noise=0.02,
                       data_weight=0.45, nn_weight=0.35, seed=3)
    ifs.set_data_anchors(normed)
    ifs.set_nn_anchor(next_pred)

    convergence_log = []
    nn_anchor_log   = []
    spread_log      = []

    WARMUP  = 60
    COUPLED = 140

    # Warm-up: IFS without NN feedback
    for _ in range(WARMUP):
        ifs.step()
        convergence_log.append(ifs.convergence)
        nn_anchor_log.append(0.5)
        spread_log.append(ifs.attractor_spread)

    # Coupled phase: retrain NN every 10 IFS steps, update anchor
    for outer in range(COUPLED):
        ifs.step()
        convergence_log.append(ifs.convergence)
        nn_anchor_log.append(next_pred)
        spread_log.append(ifs.attractor_spread)
        if outer % 10 == 0:
            nn.train_epoch(normed)
            next_pred = nn.predict_next(normed)
            ifs.set_nn_anchor(next_pred)

    final_pts = ifs.particles.copy()

    # ── Figure ────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 12))
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)
    fig.suptitle(
        f"Phase 5 — Neural Network Feedback Loop\n"
        f"Dataset: {primary_key}  |  Architecture: 2→6→1  |  "
        f"Final MSE: {final_loss:.4f}",
        y=1.02
    )

    # 1. Loss curve
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(loss_history, color=C_CORAL, lw=1.5)
    ax1.axhline(0.01, color=C_GRAY, ls="--", lw=0.8, label="0.01 threshold")
    ax1.set_title("Training loss (MSE) over epochs")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("MSE")
    ax1.set_yscale("log")
    ax1.legend()

    # 2. NN prediction vs actual
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.plot(years, normed, color=C_TEAL, lw=1.5, label="Actual")
    valid = ~np.isnan(predictions)
    ax2.plot(np.array(years)[valid], predictions[valid],
             color=C_AMBER, lw=1.5, ls="--", label="NN predicted")
    ax2.scatter([years[-1] + 1], [next_pred], s=60, color=C_AMBER,
                zorder=6, label=f"Next: {next_pred:.3f}")
    ax2.set_title("Prediction vs actual")
    ax2.set_xlabel("Year"); ax2.legend()

    # 3. IFS attractor — warm-up phase
    ax3 = fig.add_subplot(gs[1, 0])
    ifs_wu = IFSPredictor(n_particles=250, noise=0.02,
                          data_weight=0.45, nn_weight=0.0, seed=3)
    ifs_wu.set_data_anchors(normed)
    for _ in range(WARMUP):
        ifs_wu.step()
    pts_wu = ifs_wu.particles
    ax3.scatter(pts_wu[:, 0], pts_wu[:, 1], s=2, alpha=0.5, c=C_GRAY)
    anchors = ifs_wu.anchors
    ax3.scatter(anchors[:, 0], anchors[:, 1], s=10, color=C_PURPLE,
                alpha=0.5, zorder=5)
    ax3.set_title(f"IFS — data only (no NN)\nConv={ifs_wu.convergence:.0%}")
    ax3.set_xlim(0, 1); ax3.set_ylim(0, 1)

    # 4. IFS attractor — with NN feedback (final)
    ax4 = fig.add_subplot(gs[1, 1])
    conv_colors = np.where(final_pts[:, 0] > 0, C_TEAL, C_CORAL)
    converged   = ifs.particle_age > 50
    ax4.scatter(final_pts[~converged, 0], final_pts[~converged, 1],
                s=2, alpha=0.5, c=C_CORAL, label="Transient")
    ax4.scatter(final_pts[converged,  0], final_pts[converged,  1],
                s=2, alpha=0.5, c=C_TEAL,  label="Converged")
    ax4.scatter(*ifs.nn_anchor, s=100, color=C_AMBER,
                zorder=10, label=f"NN anchor ({next_pred:.3f})")
    ax4.scatter(anchors[:, 0], anchors[:, 1], s=8, color=C_PURPLE,
                alpha=0.4, zorder=5)
    ax4.set_title(f"IFS + NN feedback\nConv={ifs.convergence:.0%}  "
                  f"spread={ifs.attractor_spread:.3f}")
    ax4.set_xlim(0, 1); ax4.set_ylim(0, 1)
    ax4.legend(markerscale=4, fontsize=8)

    # 5. Convergence over time (warmup vs coupled)
    ax5 = fig.add_subplot(gs[1, 2])
    steps = list(range(len(convergence_log)))
    ax5.plot(steps[:WARMUP], convergence_log[:WARMUP],
             color=C_GRAY, lw=1.5, label="No NN")
    ax5.plot(steps[WARMUP:], convergence_log[WARMUP:],
             color=C_TEAL, lw=1.5, label="NN active")
    ax5.axvline(WARMUP, color=C_AMBER, ls="--", lw=0.9, label="NN switched on")
    ax5.set_title("Convergence: IFS only vs NN-feedback")
    ax5.set_xlabel("Step"); ax5.set_ylabel("Fraction converged")
    ax5.legend()

    # 6. Attractor spread
    ax6 = fig.add_subplot(gs[2, 0])
    ax6.plot(steps[:WARMUP], spread_log[:WARMUP], color=C_GRAY, lw=1.5)
    ax6.plot(steps[WARMUP:], spread_log[WARMUP:], color=C_TEAL, lw=1.5)
    ax6.axvline(WARMUP, color=C_AMBER, ls="--", lw=0.9)
    ax6.set_title("Attractor spread (std dev)\nNarrowing = higher order")
    ax6.set_xlabel("Step"); ax6.set_ylabel("Std dev")

    # 7. NN anchor trajectory
    ax7 = fig.add_subplot(gs[2, 1])
    ax7.plot(steps[WARMUP:],
             [v for v in nn_anchor_log[WARMUP:]],
             color=C_AMBER, lw=1.5)
    ax7.axhline(normed[-1], color=C_TEAL, ls="--",
                lw=0.9, label=f"Last actual ({normed[-1]:.3f})")
    ax7.set_title("NN prediction anchor over time")
    ax7.set_xlabel("Step (coupled phase)")
    ax7.set_ylabel("Prediction (normalised)")
    ax7.legend()

    # 8. Summary stats table
    ax8 = fig.add_subplot(gs[2, 2])
    ax8.axis("off")
    rows = [
        ["Metric",                  "Value"],
        ["Architecture",            "2 → 6 → 1"],
        ["Learning rate",           "0.010"],
        ["Epochs trained",          "600 + 14"],
        [f"Final MSE",              f"{final_loss:.5f}"],
        ["Next-step prediction",    f"{next_pred:.4f}"],
        ["Prediction error",        f"{pred_error:.4f}"],
        ["IFS particles",           "250"],
        ["Final convergence",       f"{ifs.convergence:.1%}"],
        ["Attractor spread",        f"{ifs.attractor_spread:.4f}"],
        ["NN→IFS weight",           "35%"],
        ["Data→IFS weight",         "45%"],
    ]
    tbl = ax8.table(cellText=rows[1:], colLabels=rows[0],
                    cellLoc="left", loc="center",
                    colColours=[C_BLUE+"22", C_BLUE+"22"])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1, 1.25)
    ax8.set_title("Summary statistics", pad=8)

    path = OUTPUT_DIR / f"phase5_neural_feedback_{dataset_name}.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


# ══════════════════════════════════════════════════════════════════════════
# LORENZ — Deterministic Chaos + Divergence
# ══════════════════════════════════════════════════════════════════════════
def phase_lorenz():
    section("Lorenz Attractor — Deterministic Chaos")

    trajA, trajB, div = lorenz_divergence(epsilon=1e-4, n_steps=12_000)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Lorenz Attractor — Two Trajectories, Identical Start, Divergent Paths",
                 y=1.01)

    ax1 = axes[0]
    ax1.plot(trajA[:, 0], trajA[:, 1], lw=0.4, color=C_BLUE, alpha=0.6,
             label="Trajectory A")
    ax1.set_title("Trajectory A (x₀ = 0.1)")
    ax1.set_xlabel("x"); ax1.set_ylabel("z")

    ax2 = axes[1]
    ax2.plot(trajB[:, 0], trajB[:, 1], lw=0.4, color=C_CORAL, alpha=0.6,
             label="Trajectory B")
    ax2.set_title("Trajectory B (x₀ = 0.10001)")
    ax2.set_xlabel("x"); ax2.set_ylabel("z")

    ax3 = axes[2]
    ax3.semilogy(div, color=C_AMBER, lw=1.2, label="Divergence")
    ax3.set_title("Trajectory divergence over time\n(log scale)")
    ax3.set_xlabel("Step"); ax3.set_ylabel("Distance (log)")
    ax3.legend()

    plt.tight_layout()
    path = OUTPUT_DIR / "lorenz_attractor.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"  Initial separation : 1×10⁻⁴")
    print(f"  Final divergence   : {div[-1]:.2f}")
    print(f"  Saved → {path}")


# ══════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Fractal-Driven Predictive Systems — Research Pipeline")
    parser.add_argument("--phase", type=int, choices=[1,2,3,4,5,6],
                        default=0, help="Run a single phase (0 = all)")
    parser.add_argument("--dataset", type=str,
                        choices=["birth", "climate", "combined"],
                        default="birth", help="Dataset for phases 4 & 5")
    args = parser.parse_args()

    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║   Fractal-Driven Predictive Systems — Research Pipeline  ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"  Dataset  : {args.dataset}")
    print(f"  Outputs  : {OUTPUT_DIR.resolve()}")

    runners = {
        1: phase1_chaos_game,
        2: phase2_cellular_automata,
        3: phase3_ifs_predictor,
        4: lambda: phase4_real_data(args.dataset),
        5: lambda: phase5_neural_feedback(args.dataset),
        6: phase_lorenz,
    }

    if args.phase == 0:
        for fn in runners.values():
            fn()
    else:
        runners[args.phase]()

    print("\n  All figures saved to outputs/")
    print("  Done.\n")


if __name__ == "__main__":
    main()
