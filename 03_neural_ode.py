"""
=============================================================
STEP 3: Neural ODE for Chaotic System Learning
=============================================================
A Neural ODE learns the vector field f such that:
    dz/dt = f_θ(z, t)

Instead of learning discrete updates (like RNNs), it learns
the CONTINUOUS dynamics. The ODE is solved numerically at
inference time using an adaptive step-size solver.

Reference: Chen et al. (2018) "Neural Ordinary Differential Equations"

Key Mathematical Insight:
  Standard ResNet:  z_{t+1} = z_t + f(z_t)
  Neural ODE:       dz/dt  = f_θ(z, t)   — the continuous limit

Training uses the ADJOINT METHOD to backpropagate through the
ODE solver without storing intermediate states:
  da/dt = -a^T ∂f/∂z   where a = ∂L/∂z (adjoint state)

This is memory-efficient and mathematically elegant.

NOTE: If torchdiffeq is not installed, we fall back to a
      manual Euler/RK4 solver that is self-contained.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os, time

os.makedirs("plots",   exist_ok=True)
os.makedirs("results", exist_ok=True)

# ── Try importing torchdiffeq; fall back to manual RK4 ──
try:
    from torchdiffeq import odeint
    TORCHDIFFEQ_AVAILABLE = True
    print("  [info] torchdiffeq found — using adjoint-based Neural ODE")
except ImportError:
    TORCHDIFFEQ_AVAILABLE = False
    print("  [info] torchdiffeq not installed — using manual RK4 ODE solver")


# ─────────────────────────────────────────────
#  Manual RK4 ODE solver (fallback)
# ─────────────────────────────────────────────
def rk4_step(f, t, y, dt):
    """Single RK4 step."""
    k1 = f(t,        y)
    k2 = f(t + dt/2, y + dt/2 * k1)
    k3 = f(t + dt/2, y + dt/2 * k2)
    k4 = f(t + dt,   y + dt    * k3)
    return y + dt/6 * (k1 + 2*k2 + 2*k3 + k4)


def manual_odeint(f, y0, t):
    """Integrate f from t[0] to t[-1], returning states at each t."""
    states = [y0]
    y = y0
    for i in range(len(t) - 1):
        dt = t[i+1] - t[i]
        y  = rk4_step(f, t[i], y, dt)
        states.append(y)
    return torch.stack(states, dim=0)   # (T, batch, dim)


# ─────────────────────────────────────────────
#  Neural ODE Vector Field  f_θ(z)
# ─────────────────────────────────────────────
class ODEFunc(nn.Module):
    """
    Learnable vector field f_θ: R^d → R^d.

    Architecture: 3 → 128 → 128 → 128 → 3
    Activation: Tanh (smooth, suitable for ODEs)

    The network learns to approximate the true Lorenz vector field:
        dz/dt = f_θ(z)  ≈  [σ(y-x), x(ρ-z)-y, xy-βz]
    """

    def __init__(self, latent_dim=3, hidden_dim=128, n_layers=3):
        super().__init__()

        layers = [nn.Linear(latent_dim, hidden_dim), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers.append(nn.Linear(hidden_dim, latent_dim))

        self.net = nn.Sequential(*layers)
        self.nfe = 0   # number of function evaluations (diagnostics)

        # Initialise weights small to start near identity dynamics
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.zeros_(m.bias)

    def forward(self, t, z):
        """
        t : scalar time (not used — autonomous system)
        z : (batch, 3) state
        """
        self.nfe += 1
        return self.net(z)


# ─────────────────────────────────────────────
#  Neural ODE Wrapper
# ─────────────────────────────────────────────
class NeuralODE(nn.Module):
    def __init__(self, ode_func):
        super().__init__()
        self.func = ode_func

    def forward(self, z0, t):
        """
        z0 : (batch, 3) — initial state
        t  : (T,)       — time points to evaluate at
        Returns: (T, batch, 3)
        """
        if TORCHDIFFEQ_AVAILABLE:
            return odeint(self.func, z0, t, method='rk4',
                          options=dict(step_size=t[1]-t[0]))
        else:
            return manual_odeint(self.func, z0, t)


# ─────────────────────────────────────────────
#  Dataset Preparation
# ─────────────────────────────────────────────
def prepare_data(seq_len=50, stride=5, train_frac=0.8):
    """
    Create sliding-window sequences from Lorenz data.

    Each sample: (initial_state, trajectory of length seq_len)
    """
    sol  = np.load("data/lorenz_sol.npy")   # (N, 3)
    t_np = np.load("data/lorenz_t.npy")     # (N,)

    # Normalise
    mean = sol.mean(0)
    std  = sol.std(0)
    data = (sol - mean) / std

    # Sliding windows
    seqs   = []
    starts = []
    for i in range(0, len(data) - seq_len, stride):
        seqs.append(data[i: i+seq_len])
        starts.append(i)

    seqs = np.array(seqs, dtype=np.float32)   # (n_seqs, seq_len, 3)
    dt   = float(t_np[1] - t_np[0])
    t_   = torch.linspace(0, dt * (seq_len - 1), seq_len)

    split    = int(train_frac * len(seqs))
    train_ds = TensorDataset(torch.tensor(seqs[:split]))
    test_ds  = TensorDataset(torch.tensor(seqs[split:]))

    return train_ds, test_ds, t_, mean, std


# ─────────────────────────────────────────────
#  Training Loop
# ─────────────────────────────────────────────
def train_neural_ode(n_epochs=200, batch_size=64, lr=3e-3,
                     seq_len=50, hidden_dim=128):

    device = torch.device('cpu')   # laptop-friendly

    print("\n[1/4] Preparing data …")
    train_ds, test_ds, t_seq, mean, std = prepare_data(seq_len=seq_len)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size)
    print(f"  Train sequences: {len(train_ds):,}  |  Test: {len(test_ds):,}")
    print(f"  Sequence length: {seq_len} steps  |  dt = {t_seq[1]:.4f}")

    print("\n[2/4] Building Neural ODE …")
    ode_func = ODEFunc(latent_dim=3, hidden_dim=hidden_dim, n_layers=3).to(device)
    model    = NeuralODE(ode_func).to(device)
    t_seq    = t_seq.to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    print(f"\n[3/4] Training for {n_epochs} epochs …")
    train_losses, test_losses = [], []

    for epoch in range(n_epochs):
        # ─ Train ─
        model.train()
        epoch_loss = 0.0
        for (batch,) in train_loader:
            batch = batch.to(device)           # (B, T, 3)
            z0    = batch[:, 0, :]             # (B, 3) initial states

            z_pred = model(z0, t_seq)          # (T, B, 3)
            z_pred = z_pred.permute(1, 0, 2)   # (B, T, 3)

            loss = torch.mean((z_pred - batch) ** 2)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()

        scheduler.step()
        avg_train = epoch_loss / len(train_loader)

        # ─ Validate ─
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for (batch,) in test_loader:
                batch  = batch.to(device)
                z0     = batch[:, 0, :]
                z_pred = model(z0, t_seq).permute(1, 0, 2)
                val_loss += torch.mean((z_pred - batch) ** 2).item()

        avg_val = val_loss / len(test_loader)
        train_losses.append(avg_train)
        test_losses.append(avg_val)

        if (epoch + 1) % 20 == 0 or epoch == 0:
            lr_now = optimizer.param_groups[0]['lr']
            print(f"  Epoch {epoch+1:>4}/{n_epochs}  "
                  f"Train MSE: {avg_train:.5f}  "
                  f"Val MSE: {avg_val:.5f}  "
                  f"LR: {lr_now:.2e}  "
                  f"NFE: {ode_func.nfe}")
            ode_func.nfe = 0

    # Save model
    torch.save({
        'model_state': model.state_dict(),
        'ode_func_state': ode_func.state_dict(),
        'mean': mean, 'std': std, 't_seq': t_seq.cpu().numpy()
    }, "results/neural_ode_checkpoint.pt")
    print("  Model saved: results/neural_ode_checkpoint.pt")

    return model, ode_func, train_losses, test_losses, t_seq, mean, std


# ─────────────────────────────────────────────
#  Long-horizon Rollout
# ─────────────────────────────────────────────
def long_horizon_rollout(model, t_seq, mean, std, horizon_steps=2000):
    """
    Iteratively predict seq_len steps at a time to build a
    long autonomous trajectory.
    """
    sol = np.load("data/lorenz_sol.npy")
    data = (sol - mean) / std

    seq_len = len(t_seq)
    model.eval()

    # Seed with first 50 true steps
    seed  = torch.tensor(data[:seq_len], dtype=torch.float32)
    preds = [seed.numpy()]

    z0 = seed[0].unsqueeze(0)   # (1, 3)

    n_chunks = horizon_steps // seq_len
    with torch.no_grad():
        for _ in range(n_chunks):
            z_pred = model(z0, t_seq)        # (T, 1, 3)
            chunk  = z_pred[:, 0, :].numpy()  # (T, 3)
            preds.append(chunk)
            z0 = z_pred[-1, :, :]             # use last state as new IC

    return np.concatenate(preds, axis=0), data[:horizon_steps + seq_len]


# ─────────────────────────────────────────────
#  Plots
# ─────────────────────────────────────────────
def plot_training(train_losses, test_losses):
    fig, ax = plt.subplots(figsize=(9, 4))
    fig.patch.set_facecolor('#0f0f1a')
    ax.set_facecolor('#0f0f1a')

    epochs = range(1, len(train_losses) + 1)
    ax.semilogy(epochs, train_losses, color='#00d4ff', lw=1.5, label='Train MSE')
    ax.semilogy(epochs, test_losses,  color='#ff6b6b', lw=1.5, label='Val MSE', linestyle='--')
    ax.set_xlabel("Epoch", color='white')
    ax.set_ylabel("MSE (log scale)", color='white')
    ax.set_title("Neural ODE Training Curve", color='white', fontsize=12)
    ax.legend(facecolor='#1a1a2e', labelcolor='white')
    ax.tick_params(colors='grey')
    ax.spines[:].set_color('#333355')

    plt.tight_layout()
    plt.savefig("plots/05_neural_ode_training.png", dpi=150, bbox_inches='tight',
                facecolor='#0f0f1a')
    plt.close()
    print("  Saved: plots/05_neural_ode_training.png")


def plot_rollout(pred, true):
    T   = min(len(pred), len(true), 3000)
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    fig.patch.set_facecolor('#0f0f1a')
    labels = ['x', 'y', 'z']
    colors = ['#00d4ff', '#ff6b6b', '#69ff47']

    for i, ax in enumerate(axes):
        ax.set_facecolor('#0f0f1a')
        ax.plot(true[:T, i],  color=colors[i],  lw=1.2, label='True Lorenz', alpha=0.9)
        ax.plot(pred[:T, i],  color='#ffd700',   lw=0.9, linestyle='--',
                label='Neural ODE', alpha=0.85)
        ax.set_ylabel(labels[i], color='white')
        ax.legend(facecolor='#1a1a2e', labelcolor='white', fontsize=8)
        ax.tick_params(colors='grey')
        ax.spines[:].set_color('#333355')

    axes[-1].set_xlabel("Time steps", color='white')
    plt.suptitle("Neural ODE — Long-Horizon Autonomous Rollout vs True Lorenz",
                 color='white', fontsize=12)
    plt.tight_layout()
    plt.savefig("plots/06_neural_ode_rollout.png", dpi=150, bbox_inches='tight',
                facecolor='#0f0f1a')
    plt.close()
    print("  Saved: plots/06_neural_ode_rollout.png")


def plot_phase_space(pred, true):
    fig = plt.figure(figsize=(12, 5))
    fig.patch.set_facecolor('#0f0f1a')

    ax1 = fig.add_subplot(121, projection='3d', facecolor='#0f0f1a')
    T = min(len(true), 5000)
    ax1.plot(true[:T,0], true[:T,1], true[:T,2],
             lw=0.5, color='#00d4ff', alpha=0.8)
    ax1.set_title("True Lorenz Attractor", color='white')
    ax1.tick_params(colors='grey')
    for pane in [ax1.xaxis.pane, ax1.yaxis.pane, ax1.zaxis.pane]:
        pane.fill = False

    ax2 = fig.add_subplot(122, projection='3d', facecolor='#0f0f1a')
    P = min(len(pred), 5000)
    ax2.plot(pred[:P,0], pred[:P,1], pred[:P,2],
             lw=0.5, color='#ffd700', alpha=0.8)
    ax2.set_title("Neural ODE Learned Attractor", color='white')
    ax2.tick_params(colors='grey')
    for pane in [ax2.xaxis.pane, ax2.yaxis.pane, ax2.zaxis.pane]:
        pane.fill = False

    plt.suptitle("Attractor Geometry: True vs Neural ODE", color='white', fontsize=12)
    plt.tight_layout()
    plt.savefig("plots/07_phase_space_comparison.png", dpi=150, bbox_inches='tight',
                facecolor='#0f0f1a')
    plt.close()
    print("  Saved: plots/07_phase_space_comparison.png")


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  STEP 3 — Neural ODE")
    print("=" * 60)

    t0 = time.time()
    model, ode_func, tr_loss, te_loss, t_seq, mean, std = train_neural_ode(
        n_epochs=150,
        batch_size=64,
        lr=3e-3,
        seq_len=40,
        hidden_dim=128
    )
    print(f"\n  Training time: {(time.time()-t0)/60:.1f} min")

    print("\n[4/4] Generating long-horizon rollout …")
    pred, true = long_horizon_rollout(model, t_seq, mean, std, horizon_steps=2000)

    np.save("results/neural_ode_pred.npy", pred)
    np.save("results/neural_ode_true.npy", true)

    plot_training(tr_loss, te_loss)
    plot_rollout(pred, true)
    plot_phase_space(pred, true)

    print("\n✅  Step 3 Complete!")
    print("\nNext → Run: python 04_comparison_analysis.py")