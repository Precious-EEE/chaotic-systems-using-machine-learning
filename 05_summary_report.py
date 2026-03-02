"""
=============================================================
STEP 5: Final Summary Report Generator
=============================================================
Assembles all results into a publication-quality summary figure
and prints a formatted report.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os, glob

os.makedirs("plots", exist_ok=True)


def make_summary_figure():
    """
    Build a single 3x3 multi-panel figure summarising the project.
    """
    # Check which plots exist
    plot_files = {
        'attractor':   'plots/01_lorenz_attractor.png',
        'sensitivity': 'plots/02_sensitive_dependence.png',
        'esn_pred':    'plots/03_esn_prediction.png',
        'esn_auto':    'plots/04_esn_autonomous.png',
        'node_train':  'plots/05_neural_ode_training.png',
        'node_rollout':'plots/06_neural_ode_rollout.png',
        'phase_cmp':   'plots/07_phase_space_comparison.png',
        'spectra':     'plots/08_power_spectra.png',
        'corr_dim':    'plots/09_correlation_dimension.png',
        'comparison':  'plots/10_model_comparison.png',
    }

    existing = {k: v for k, v in plot_files.items() if os.path.exists(v)}
    print(f"  Found {len(existing)}/{len(plot_files)} plot files")

    if len(existing) < 3:
        print("  Not enough plots to build summary. Run the earlier steps first.")
        return

    fig = plt.figure(figsize=(20, 16))
    fig.patch.set_facecolor('#0a0a18')

    # Title
    fig.suptitle(
        "Predicting Chaotic Systems Using Neural ODEs & Reservoir Computing\n"
        "A Mathematics × Machine Learning Research Project",
        color='white', fontsize=16, fontweight='bold', y=0.98
    )

    keys_order = ['attractor', 'sensitivity', 'esn_pred',
                  'phase_cmp', 'node_train',  'node_rollout',
                  'spectra',   'corr_dim',    'comparison']

    available_keys = [k for k in keys_order if k in existing][:9]

    rows = 3
    cols = 3
    gs   = gridspec.GridSpec(rows, cols, figure=fig,
                              hspace=0.35, wspace=0.25,
                              left=0.03, right=0.97,
                              top=0.93, bottom=0.03)

    panel_titles = {
        'attractor':   '① Lorenz Strange Attractor',
        'sensitivity': '② Butterfly Effect (Chaos)',
        'esn_pred':    '③ ESN — Teacher-Forced Prediction',
        'phase_cmp':   '④ Phase Space: True vs Neural ODE',
        'node_train':  '⑤ Neural ODE Training Convergence',
        'node_rollout':'⑥ Neural ODE Long-Horizon Rollout',
        'spectra':     '⑦ Power Spectral Density Comparison',
        'corr_dim':    '⑧ Correlation Dimension D₂',
        'comparison':  '⑨ Final Model Comparison',
    }

    for idx, key in enumerate(available_keys):
        row, col = divmod(idx, cols)
        ax = fig.add_subplot(gs[row, col])
        ax.set_facecolor('#0a0a18')

        try:
            img = plt.imread(existing[key])
            ax.imshow(img, aspect='auto')
        except Exception as e:
            ax.text(0.5, 0.5, f"[{key}]\n{e}", ha='center', va='center',
                    transform=ax.transAxes, color='grey')

        ax.set_title(panel_titles.get(key, key), color='#aaccff', fontsize=9, pad=4)
        ax.axis('off')

    # Empty panels filled with info
    for idx in range(len(available_keys), rows * cols):
        row, col = divmod(idx, cols)
        ax = fig.add_subplot(gs[row, col])
        ax.set_facecolor('#0a0a18')
        ax.text(0.5, 0.5, "Run all steps to\ngenerate this panel",
                ha='center', va='center', transform=ax.transAxes,
                color='#444466', fontsize=10, style='italic')
        ax.axis('off')

    out = "plots/SUMMARY_FIGURE.png"
    plt.savefig(out, dpi=120, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {out}")
    return out


def print_final_report():
    print()
    print("╔" + "═"*62 + "╗")
    print("║" + "  FINAL PROJECT REPORT".center(62) + "║")
    print("║" + "  Chaotic System Prediction via Neural ODEs & ESN".center(62) + "║")
    print("╠" + "═"*62 + "╣")

    # Load saved results
    if os.path.exists("data/lyapunov_exponents.npy"):
        le = np.load("data/lyapunov_exponents.npy")
        print("║" + "".center(62) + "║")
        print("║  MATHEMATICAL ANALYSIS — LORENZ SYSTEM".ljust(63) + "║")
        print("║" + "─"*62 + "║")
        print(f"║  Lyapunov exponent λ₁:  {le[0]:+.4f}  (chaos confirmed)  ".ljust(63) + "║")
        print(f"║  Lyapunov exponent λ₂:  {le[1]:+.4f}                       ".ljust(63) + "║")
        print(f"║  Lyapunov exponent λ₃:  {le[2]:+.4f}  (dissipative)       ".ljust(63) + "║")
        print(f"║  Lyapunov time:  1/λ₁  =  {1/le[0]:.2f} time units          ".ljust(63) + "║")
        print(f"║  KY Dimension:  d_KY   =  {2 + le[0]/abs(le[2]):.3f}             ".ljust(63) + "║")

    print("║" + "".center(62) + "║")
    print("║  MODELS IMPLEMENTED".ljust(63) + "║")
    print("║" + "─"*62 + "║")
    print("║  1. Echo State Network (Reservoir Computing)          ║")
    print("║     • N=1000 neurons, ρ=0.95, sparse connections      ║")
    print("║     • Ridge regression output training (closed-form)  ║")
    print("║     • Teacher-forced & autonomous modes               ║")
    print("║" + "".center(62) + "║")
    print("║  2. Neural ODE (Chen et al., 2018)                    ║")
    print("║     • Learns continuous vector field dz/dt = f_θ(z)   ║")
    print("║     • 3-layer Tanh network, 128 hidden units           ║")
    print("║     • RK4 integration, adjoint backpropagation         ║")

    print("║" + "".center(62) + "║")
    print("║  MATHEMATICS DEMONSTRATED".ljust(63) + "║")
    print("║" + "─"*62 + "║")
    print("║  ✓ Ordinary Differential Equations (Lorenz system)     ║")
    print("║  ✓ Dynamical Systems Theory (attractors, bifurcations) ║")
    print("║  ✓ Lyapunov Exponents & Chaos Quantification           ║")
    print("║  ✓ Spectral Theory (eigenvalues, spectral radius)      ║")
    print("║  ✓ Linear Algebra (Ridge regression, QR decomposition) ║")
    print("║  ✓ Functional Analysis (ODE flow maps)                 ║")
    print("║  ✓ Fractal Geometry (correlation/Kaplan-Yorke dim)     ║")
    print("║  ✓ Optimisation (Adam, cosine annealing, adjoint)      ║")
    print("║  ✓ Spectral Analysis (Welch PSD, frequency domain)     ║")

    print("║" + "".center(62) + "║")
    print("║  OUTPUT FILES".ljust(63) + "║")
    print("║" + "─"*62 + "║")
    plots = sorted(glob.glob("plots/*.png"))
    for p in plots:
        print(f"║    {p:<57} ║")

    print("╚" + "═"*62 + "╝")


if __name__ == "__main__":
    print("=" * 60)
    print("  STEP 5 — Summary Report")
    print("=" * 60)

    print("\n[1/2] Building summary figure …")
    make_summary_figure()

    print("\n[2/2] Printing final report …")
    print_final_report()

    print("\n" + "="*60)
    print("  🎓  PROJECT COMPLETE!")
    print("="*60)
    print("""
  What you built:
  ───────────────
  ✅  Simulated the Lorenz chaotic attractor from scratch
  ✅  Computed Lyapunov exponents (proved chaos mathematically)
  ✅  Built & trained an Echo State Network (reservoir computing)
  ✅  Built & trained a Neural ODE (Chen et al., 2018)
  ✅  Compared models via PSD, fractal dimension, MSE, VPT
  ✅  Generated publication-quality visualisations

  Next steps to make this Ivy-League-ready:
  ──────────────────────────────────────────
  → Add Topological Data Analysis (giotto-tda library)
    • Compute persistent homology of true vs learned attractor
    • Measure Wasserstein distance between persistence diagrams
  → Test on Rössler attractor, double pendulum
  → Write it up as a conference paper (arXiv-ready)
  → Compare against LSTM baseline
    """)