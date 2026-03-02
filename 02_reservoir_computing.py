"""
=============================================================
STEP 2: Echo State Network (Reservoir Computing)
=============================================================
An Echo State Network (ESN) is a type of Recurrent Neural
Network where only the output weights are trained.

Architecture:
  u(t) → [Win] → Reservoir (W, fixed) → [Wout, trained] → ŷ(t)

Key Mathematics:
  x(t+1) = tanh(W·x(t) + Win·u(t) + b)
  y(t)   = Wout · x(t)

  Training: Wout = Y · X^T · (X·X^T + λI)^{-1}
  (Ridge Regression — closed-form solution)

The "echo state property" requires the spectral radius ρ(W) < 1
so past inputs fade (the network has "fading memory").
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import os

os.makedirs("plots",   exist_ok=True)
os.makedirs("results", exist_ok=True)


# ─────────────────────────────────────────────
#  Echo State Network Class
# ─────────────────────────────────────────────
class EchoStateNetwork:
    """
    Echo State Network for time-series prediction.

    Parameters
    ----------
    n_reservoir : int     — number of reservoir neurons
    spectral_radius : float — desired spectral radius of W (< 1 for ESP)
    sparsity : float      — fraction of zero connections in W
    input_scaling : float — scale of Win
    alpha : float         — leaking rate (α=1 → no leaking)
    ridge_alpha : float   — regularisation for Ridge regression
    """

    def __init__(self,
                 n_reservoir=500,
                 spectral_radius=0.95,
                 sparsity=0.9,
                 input_scaling=1.0,
                 alpha=1.0,
                 ridge_alpha=1e-6,
                 random_state=42):

        self.N   = n_reservoir
        self.rho = spectral_radius
        self.sp  = sparsity
        self.is_ = input_scaling
        self.alpha = alpha
        self.ridge_alpha = ridge_alpha
        self.rng = np.random.RandomState(random_state)

        self.W    = None   # reservoir weights
        self.Win  = None   # input weights
        self.Wout = None   # output weights (trained)

    # ──── Initialise reservoir ─────────────────
    def _init_reservoir(self, n_inputs, n_outputs):
        # Sparse random reservoir
        W = self.rng.randn(self.N, self.N)
        W[self.rng.rand(self.N, self.N) < self.sp] = 0.0

        # Scale to desired spectral radius
        eigvals    = np.linalg.eigvals(W)
        current_sr = np.max(np.abs(eigvals))
        self.W     = W * (self.rho / current_sr)

        # Random input weights
        self.Win = self.rng.randn(self.N, n_inputs) * self.is_

    # ──── Run reservoir dynamics ───────────────
    def _run_reservoir(self, inputs):
        """
        inputs : (T, n_inputs)
        Returns: (T, N) reservoir states
        """
        T   = inputs.shape[0]
        X   = np.zeros((T, self.N))
        x   = np.zeros(self.N)

        for t in range(T):
            x = (1 - self.alpha) * x + self.alpha * np.tanh(
                self.W @ x + self.Win @ inputs[t]
            )
            X[t] = x

        return X

    # ──── Train ───────────────────────────────
    def fit(self, inputs, targets, warmup=200):
        """
        inputs  : (T, n_inputs)   — input time series
        targets : (T, n_outputs)  — target outputs
        warmup  : int             — discard initial transient states
        """
        n_in  = inputs.shape[1]
        n_out = targets.shape[1]
        self._init_reservoir(n_in, n_out)

        print(f"  Running reservoir (N={self.N}, ρ={self.rho}) …")
        X = self._run_reservoir(inputs)

        # Discard warmup
        X_train = X[warmup:]
        Y_train = targets[warmup:]

        # Ridge regression: Wout minimises ||Y - Wout·X||² + λ||Wout||²
        reg = Ridge(alpha=self.ridge_alpha, fit_intercept=False)
        reg.fit(X_train, Y_train)
        self.Wout = reg.coef_    # shape: (n_out, N)

        train_pred = X_train @ self.Wout.T
        train_mse  = mean_squared_error(Y_train, train_pred)
        print(f"  Training MSE (after warmup): {train_mse:.6f}")
        return self

    # ──── Predict: teacher-forced ─────────────
    def predict(self, inputs):
        """Predict using teacher-forced inputs (test mode)."""
        X = self._run_reservoir(inputs)
        return X @ self.Wout.T

    # ──── Predict: autonomous (closed-loop) ───
    def generate(self, seed_input, n_steps):
        """
        Closed-loop / autonomous prediction:
        Feed the network's own output back as input.

        seed_input : (warmup, n_inputs) — initial driver
        n_steps    : int                — steps to generate freely
        """
        N_in  = seed_input.shape[1]
        x     = np.zeros(self.N)

        # Warm up with seed
        for t in range(len(seed_input)):
            x = (1 - self.alpha) * x + self.alpha * np.tanh(
                self.W @ x + self.Win @ seed_input[t]
            )

        # Free generation
        generated = []
        u = seed_input[-1]
        for _ in range(n_steps):
            x = (1 - self.alpha) * x + self.alpha * np.tanh(
                self.W @ x + self.Win @ u
            )
            y = self.Wout @ x
            generated.append(y)
            u = y   # feed output back as next input

        return np.array(generated)


# ─────────────────────────────────────────────
#  Utility: Prediction Horizon
# ─────────────────────────────────────────────
def valid_prediction_time(true, pred, threshold=0.4):
    """
    Time until normalised prediction error exceeds threshold.
    Often measured in Lyapunov times.
    """
    std   = np.std(true, axis=0)
    error = np.sqrt(np.mean(((true - pred) / (std + 1e-8)) ** 2, axis=1))
    idx   = np.where(error > threshold)[0]
    return idx[0] if len(idx) > 0 else len(true)


# ─────────────────────────────────────────────
#  Main Training and Evaluation
# ─────────────────────────────────────────────
def run_esn():
    # ── Load Lorenz data ──────────────────────
    print("\n[1/5] Loading Lorenz data …")
    t   = np.load("data/lorenz_t.npy")
    sol = np.load("data/lorenz_sol.npy")   # (N, 3)

    # Normalise to [-1, 1] for ESN stability
    mean = sol.mean(axis=0)
    std  = sol.std(axis=0)
    data = (sol - mean) / std

    # ── Train / Test split ────────────────────
    split = int(0.8 * len(data))
    train_data = data[:split]
    test_data  = data[split:]

    # Use x(t) to predict x(t+1): shift by 1 step
    X_train = train_data[:-1]
    Y_train = train_data[1:]
    X_test  = test_data[:-1]
    Y_test  = test_data[1:]

    print(f"  Train: {len(X_train):,} steps  |  Test: {len(X_test):,} steps")

    # ── Build & train ESN ─────────────────────
    print("\n[2/5] Building Echo State Network …")
    esn = EchoStateNetwork(
        n_reservoir=1000,
        spectral_radius=0.95,
        sparsity=0.85,
        input_scaling=1.0,
        alpha=0.3,
        ridge_alpha=1e-6
    )
    esn.fit(X_train, Y_train, warmup=200)

    # ── Teacher-forced test ───────────────────
    print("\n[3/5] Teacher-forced prediction on test set …")
    Y_pred_tf = esn.predict(X_test)
    mse_tf    = mean_squared_error(Y_test, Y_pred_tf)
    print(f"  Test MSE (teacher-forced): {mse_tf:.6f}")

    # ── Autonomous prediction ─────────────────
    print("\n[4/5] Autonomous (closed-loop) prediction …")
    seed = test_data[:500]
    n_gen = 3000
    generated = esn.generate(seed, n_steps=n_gen)

    # Compare with true continuation
    true_cont = data[split + 500: split + 500 + n_gen]
    if len(true_cont) >= n_gen:
        vpt = valid_prediction_time(true_cont[:n_gen], generated[:len(true_cont)])
        lyapunov_time = 1 / 0.9056   # λ₁ ≈ 0.9056 for standard Lorenz
        print(f"  Valid Prediction Time: {vpt} steps ({vpt * 0.005 / lyapunov_time:.1f} Lyapunov times)")
        np.save("results/esn_vpt.npy", np.array([vpt]))

    # Save results
    np.save("results/esn_pred_tf.npy", Y_pred_tf)
    np.save("results/esn_generated.npy", generated)
    np.save("results/esn_true.npy", Y_test)

    # ── Plots ─────────────────────────────────
    print("\n[5/5] Plotting results …")
    dt = t[1] - t[0]
    t_test = np.arange(len(Y_test)) * dt

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    fig.patch.set_facecolor('#0f0f1a')

    labels  = ['x', 'y', 'z']
    colors  = ['#00d4ff', '#ff6b6b', '#69ff47']

    for i, ax in enumerate(axes):
        ax.set_facecolor('#0f0f1a')
        ax.plot(t_test, Y_test[:, i],
                color=colors[i], lw=1.2, label=f'True {labels[i]}(t)', alpha=0.9)
        ax.plot(t_test, Y_pred_tf[:, i],
                color='white', lw=0.8, linestyle='--',
                label='ESN (teacher-forced)', alpha=0.7)
        ax.set_ylabel(labels[i], color='white')
        ax.legend(facecolor='#1a1a2e', labelcolor='white', fontsize=8, loc='upper right')
        ax.tick_params(colors='grey')
        ax.spines[:].set_color('#333355')

    axes[-1].set_xlabel("Time", color='white')
    plt.suptitle("Echo State Network — Teacher-Forced Prediction on Lorenz Attractor",
                 color='white', fontsize=12)
    plt.tight_layout()
    plt.savefig("plots/03_esn_prediction.png", dpi=150, bbox_inches='tight',
                facecolor='#0f0f1a')
    plt.close()
    print("  Saved: plots/03_esn_prediction.png")

    # Autonomous prediction plot
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    fig.patch.set_facecolor('#0f0f1a')
    t_gen = np.arange(n_gen) * dt

    for i, ax in enumerate(axes):
        ax.set_facecolor('#0f0f1a')
        if len(true_cont) >= n_gen:
            ax.plot(t_gen, true_cont[:n_gen, i],
                    color=colors[i], lw=1.2, label='True', alpha=0.9)
        ax.plot(t_gen, generated[:, i],
                color='#ffd700', lw=0.9, linestyle='--',
                label='ESN (autonomous)', alpha=0.85)
        ax.set_ylabel(labels[i], color='white')
        ax.legend(facecolor='#1a1a2e', labelcolor='white', fontsize=8)
        ax.tick_params(colors='grey')
        ax.spines[:].set_color('#333355')

    axes[-1].set_xlabel("Time", color='white')
    plt.suptitle("Echo State Network — Autonomous (Closed-loop) Generation",
                 color='white', fontsize=12)
    plt.tight_layout()
    plt.savefig("plots/04_esn_autonomous.png", dpi=150, bbox_inches='tight',
                facecolor='#0f0f1a')
    plt.close()
    print("  Saved: plots/04_esn_autonomous.png")

    print("\n✅  Step 2 Complete!")
    print("   Results saved in: results/")
    print("\nNext → Run: python 03_neural_ode.py")

    return esn


if __name__ == "__main__":
    print("=" * 60)
    print("  STEP 2 — Echo State Network (Reservoir Computing)")
    print("=" * 60)
    run_esn()