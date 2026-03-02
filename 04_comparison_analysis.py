"""
=============================================================
STEP 4: Model Comparison & Attractor Analysis
=============================================================
Compare all methods:
  1. Echo State Network (ESN)
  2. Neural ODE

Metrics:
  - Short-term: MSE on test trajectories
  - Long-term: Valid Prediction Time (in Lyapunov units)
  - Structural: Attractor statistics (mean, std, power spectrum)
  - Geometric: Correlation dimension (fractal dimension estimate)

Mathematical Background — Correlation Dimension:
  C(r) = (2/N²) Σ_{i<j} Θ(r - ||x_i - x_j||)
  D₂   = lim_{r→0} log C(r) / log r

For the Lorenz attractor: D₂ ≈ 2.06
A good model should produce D₂ ≈ 2.06 autonomously.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import welch
from sklearn.metrics import mean_squared_error
import warnings, os

os.makedirs("plots",   exist_ok=True)
os.makedirs("results", exist_ok=True)

# ─────────────────────────────────────────────
#  Correlation Dimension (Grassberger-Procaccia)
# ─────────────────────────────────────────────
def correlation_dimension(data, n_samples=3000, r_range=None):
    """
    Estimate the correlation dimension D₂ of an attractor
    using the Grassberger-Procaccia algorithm.

    data : (N, d) array of attractor points
    Returns: (r_vals, log_C, D2_estimate)
    """
    # Subsample for speed
    idx  = np.random.choice(len(data), min(n_samples, len(data)), replace=False)
    pts  = data[idx]

    # All pairwise distances
    from scipy.spatial.distance import pdist
    dists = pdist(pts)

    if r_range is None:
        r_min = np.percentile(dists, 1)
        r_max = np.percentile(dists, 50)
        r_vals = np.logspace(np.log10(r_min), np.log10(r_max), 30)
    else:
        r_vals = r_range

    # C(r) = fraction of pairs with distance < r
    N     = len(pts)
    total = N * (N - 1) / 2
    C_r   = np.array([(dists < r).sum() / total for r in r_vals])

    # Slope in log-log = D₂
    valid    = C_r > 0
    log_r    = np.log(r_vals[valid])
    log_C    = np.log(C_r[valid])

    # Linear fit in the scaling region (middle 50%)
    mid      = slice(len(log_r)//4, 3*len(log_r)//4)
    coeffs   = np.polyfit(log_r[mid], log_C[mid], 1)
    D2       = coeffs[0]

    return r_vals[valid], log_C, D2


# ─────────────────────────────────────────────
#  Power Spectral Density
# ─────────────────────────────────────────────
def power_spectrum(signal, fs=200.0):
    """Return frequencies and PSD using Welch's method."""
    f, Pxx = welch(signal, fs=fs, nperseg=512)
    return f, Pxx


# ─────────────────────────────────────────────
#  Attractor Statistics
# ─────────────────────────────────────────────
def attractor_stats(data):
    return {
        'mean':   data.mean(0),
        'std':    data.std(0),
        'min':    data.min(0),
        'max':    data.max(0),
        'range':  data.max(0) - data.min(0),
    }


# ─────────────────────────────────────────────
#  Valid Prediction Time
# ─────────────────────────────────────────────
def vpt(true, pred, threshold=0.4):
    std   = true.std(axis=0) + 1e-8
    err   = np.sqrt(((true - pred) / std) ** 2).mean(axis=1)
    idx   = np.where(err > threshold)[0]
    return idx[0] if len(idx) > 0 else len(true)


# ─────────────────────────────────────────────
#  Load All Data
# ─────────────────────────────────────────────
def load_data():
    print("  Loading saved data …")
    sol  = np.load("data/lorenz_sol.npy")
    mean_ = sol.mean(0); std_ = sol.std(0)
    true_norm = (sol - mean_) / std_

    results = {'true': true_norm}

    # ESN results
    if os.path.exists("results/esn_generated.npy"):
        results['esn'] = np.load("results/esn_generated.npy")
        results['esn_true'] = np.load("results/esn_true.npy")
        results['esn_pred_tf'] = np.load("results/esn_pred_tf.npy")
        print("    ✓ ESN results loaded")
    else:
        print("    ✗ ESN results not found — run 02_reservoir_computing.py first")

    # Neural ODE results
    if os.path.exists("results/neural_ode_pred.npy"):
        results['node'] = np.load("results/neural_ode_pred.npy")
        results['node_true'] = np.load("results/neural_ode_true.npy")
        print("    ✓ Neural ODE results loaded")
    else:
        print("    ✗ Neural ODE results not found — run 03_neural_ode.py first")

    return results


# ─────────────────────────────────────────────
#  Print Summary Table
# ─────────────────────────────────────────────
def print_summary_table(metrics):
    print("\n" + "="*65)
    print("  RESULTS SUMMARY TABLE")
    print("="*65)
    print(f"  {'Metric':<30}  {'ESN':>12}  {'Neural ODE':>12}")
    print("-"*65)
    for key, row in metrics.items():
        esn_val  = f"{row.get('esn', 'N/A'):>12.4f}" if isinstance(row.get('esn'), float) else f"{'N/A':>12}"
        node_val = f"{row.get('node', 'N/A'):>12.4f}" if isinstance(row.get('node'), float) else f"{'N/A':>12}"
        print(f"  {key:<30}  {esn_val}  {node_val}")
    print("="*65)


# ─────────────────────────────────────────────
#  Plots
# ─────────────────────────────────────────────
def plot_power_spectra(results):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    fig.patch.set_facecolor('#0f0f1a')
    labels = ['x', 'y', 'z']
    colors = {'true': '#00d4ff', 'esn': '#ffd700', 'node': '#ff6b6b'}

    for i, ax in enumerate(axes):
        ax.set_facecolor('#0f0f1a')
        f_t, P_t = power_spectrum(results['true'][:, i])
        ax.semilogy(f_t, P_t, color=colors['true'], lw=1.5, label='True Lorenz', alpha=0.9)

        if 'esn' in results:
            f_e, P_e = power_spectrum(results['esn'][:, i])
            ax.semilogy(f_e, P_e, color=colors['esn'], lw=1.2, linestyle='--',
                        label='ESN', alpha=0.85)

        if 'node' in results:
            data = results['node']
            if len(data) > 50:
                f_n, P_n = power_spectrum(data[:, i])
                ax.semilogy(f_n, P_n, color=colors['node'], lw=1.2, linestyle=':',
                            label='Neural ODE', alpha=0.85)

        ax.set_xlabel("Frequency", color='white')
        ax.set_ylabel("PSD" if i == 0 else "", color='white')
        ax.set_title(f"Dimension {labels[i]}", color='white')
        ax.legend(facecolor='#1a1a2e', labelcolor='white', fontsize=8)
        ax.tick_params(colors='grey')
        ax.spines[:].set_color('#333355')

    plt.suptitle("Power Spectral Density — Model Comparison", color='white', fontsize=12)
    plt.tight_layout()
    plt.savefig("plots/08_power_spectra.png", dpi=150, bbox_inches='tight',
                facecolor='#0f0f1a')
    plt.close()
    print("  Saved: plots/08_power_spectra.png")


def plot_correlation_dimension(results):
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor('#0f0f1a')
    ax.set_facecolor('#0f0f1a')

    print("  Computing correlation dimensions …")
    datasets = {'True Lorenz (#00d4ff)': (results['true'][1000:6000], '#00d4ff')}
    if 'esn' in results:
        datasets['ESN (#ffd700)'] = (results['esn'][:5000], '#ffd700')
    if 'node' in results and len(results['node']) > 100:
        datasets['Neural ODE (#ff6b6b)'] = (results['node'][:5000], '#ff6b6b')

    dim_results = {}
    for name, (data, color) in datasets.items():
        label_name = name.split(' (')[0]
        try:
            r, logC, D2 = correlation_dimension(data)
            ax.plot(np.log(r), logC, color=color, lw=2, label=f'{label_name}  D₂≈{D2:.2f}')
            dim_results[label_name] = D2
            print(f"    {label_name}: D₂ = {D2:.3f}  (true ≈ 2.06)")
        except Exception as e:
            print(f"    {label_name}: failed ({e})")

    ax.set_xlabel("log(r)", color='white')
    ax.set_ylabel("log C(r)", color='white')
    ax.set_title("Correlation Dimension  —  Grassberger-Procaccia Algorithm\n"
                 "Slope = D₂  (Lorenz true value ≈ 2.06)", color='white')
    ax.legend(facecolor='#1a1a2e', labelcolor='white', fontsize=9)
    ax.tick_params(colors='grey')
    ax.spines[:].set_color('#333355')

    plt.tight_layout()
    plt.savefig("plots/09_correlation_dimension.png", dpi=150, bbox_inches='tight',
                facecolor='#0f0f1a')
    plt.close()
    print("  Saved: plots/09_correlation_dimension.png")
    return dim_results


def plot_model_comparison_bar(metrics):
    """Bar chart comparing key metrics across models."""
    if not metrics:
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fig.patch.set_facecolor('#0f0f1a')
    colors_ = ['#ffd700', '#ff6b6b']
    models  = ['ESN', 'Neural ODE']

    # Short-term MSE
    ax = axes[0]
    ax.set_facecolor('#0f0f1a')
    mse_vals = [metrics.get('Short-term MSE', {}).get('esn', 0),
                metrics.get('Short-term MSE', {}).get('node', 0)]
    bars = ax.bar(models, mse_vals, color=colors_, alpha=0.85, edgecolor='white', linewidth=0.5)
    ax.set_title("Short-term MSE (lower=better)", color='white')
    ax.set_ylabel("MSE", color='white')
    ax.tick_params(colors='grey')
    ax.spines[:].set_color('#333355')
    for b, v in zip(bars, mse_vals):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.001,
                f'{v:.4f}', ha='center', color='white', fontsize=9)

    # Correlation dimension error
    ax2 = axes[1]
    ax2.set_facecolor('#0f0f1a')
    true_d2 = metrics.get('Correlation Dim D2', {}).get('true', 2.06)
    d2_err  = [abs(metrics.get('Correlation Dim D2', {}).get('esn',  2.06) - true_d2),
               abs(metrics.get('Correlation Dim D2', {}).get('node', 2.06) - true_d2)]
    bars2 = ax2.bar(models, d2_err, color=colors_, alpha=0.85, edgecolor='white', linewidth=0.5)
    ax2.set_title("|D₂ error|  vs  true D₂≈2.06  (lower=better)", color='white')
    ax2.set_ylabel("|ΔD₂|", color='white')
    ax2.tick_params(colors='grey')
    ax2.spines[:].set_color('#333355')
    for b, v in zip(bars2, d2_err):
        ax2.text(b.get_x() + b.get_width()/2, b.get_height() + 0.002,
                 f'{v:.3f}', ha='center', color='white', fontsize=9)

    plt.suptitle("Model Comparison Dashboard", color='white', fontsize=12)
    plt.tight_layout()
    plt.savefig("plots/10_model_comparison.png", dpi=150, bbox_inches='tight',
                facecolor='#0f0f1a')
    plt.close()
    print("  Saved: plots/10_model_comparison.png")


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  STEP 4 — Comparison Analysis & Attractor Topology")
    print("=" * 60)

    results = load_data()
    metrics = {}

    # ── Short-term MSE ─────────────────────────
    print("\n[1/4] Short-term prediction accuracy …")
    mse_row = {}
    if 'esn_pred_tf' in results and 'esn_true' in results:
        m = mean_squared_error(results['esn_true'], results['esn_pred_tf'])
        mse_row['esn'] = float(m)
        print(f"  ESN  teacher-forced MSE: {m:.6f}")
    if 'node' in results and 'node_true' in results:
        L = min(len(results['node']), len(results['node_true']))
        m = mean_squared_error(results['node_true'][:L], results['node'][:L])
        mse_row['node'] = float(m)
        print(f"  Neural ODE  MSE: {m:.6f}")
    metrics['Short-term MSE'] = mse_row

    # ── Valid prediction time ──────────────────
    print("\n[2/4] Valid prediction time …")
    lyap_dt = 0.005 / (1 / 0.9056)   # steps per Lyapunov time
    vpt_row  = {}
    if 'esn_pred_tf' in results and 'esn_true' in results:
        v = vpt(results['esn_true'], results['esn_pred_tf'])
        vpt_row['esn'] = float(v)
        print(f"  ESN  VPT: {v} steps  ({v * 0.005 / (1/0.9056):.1f} Lyapunov times)")
    metrics['Valid Pred Time (steps)'] = vpt_row

    # ── Power spectra plot ─────────────────────
    print("\n[3/4] Plotting power spectra …")
    plot_power_spectra(results)

    # ── Correlation dimensions ─────────────────
    print("\n[4/4] Correlation dimension analysis …")
    dim_results = plot_correlation_dimension(results)

    d2_row = {'true': 2.06}
    if 'ESN' in dim_results:
        d2_row['esn']  = dim_results['ESN']
    if 'Neural ODE' in dim_results:
        d2_row['node'] = dim_results['Neural ODE']
    metrics['Correlation Dim D2'] = d2_row

    # ── Summary ───────────────────────────────
    plot_model_comparison_bar(metrics)
    print_summary_table(metrics)

    # Save metrics
    np.save("results/metrics.npy", metrics)

    print("\n✅  Step 4 Complete!")
    print("   All plots saved in: plots/")
    print("\nNext → Run: python 05_summary_report.py")