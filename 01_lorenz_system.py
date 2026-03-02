"""
=============================================================
STEP 1: Lorenz Chaotic System - Mathematical Foundation
=============================================================
This module simulates the Lorenz attractor and computes key
mathematical properties:
  - Lyapunov exponents (measure of chaos)
  - Phase space trajectories
  - Attractor visualization

The Lorenz system:
    dx/dt = σ(y - x)
    dy/dt = x(ρ - z) - y
    dz/dt = xy - βz

Classic parameters: σ=10, ρ=28, β=8/3 → chaotic regime
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import os

os.makedirs("plots", exist_ok=True)
os.makedirs("data",  exist_ok=True)


# ─────────────────────────────────────────────
#  Lorenz ODE Definition
# ─────────────────────────────────────────────
def lorenz(t, state, sigma=10.0, rho=28.0, beta=8/3):
    """Lorenz system ODE — returns time derivatives."""
    x, y, z = state
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return [dxdt, dydt, dzdt]


def lorenz_jacobian(state, sigma=10.0, rho=28.0, beta=8/3):
    """
    Jacobian matrix J of the Lorenz system at a given point.
    Used for Lyapunov exponent computation.
    J_ij = ∂f_i / ∂x_j
    """
    x, y, z = state
    J = np.array([
        [-sigma,  sigma,   0  ],
        [rho-z,   -1,     -x  ],
        [y,        x,    -beta]
    ])
    return J


# ─────────────────────────────────────────────
#  Simulate Lorenz Trajectory
# ─────────────────────────────────────────────
def simulate_lorenz(t_span=(0, 60), dt=0.01,
                    ic=None, sigma=10.0, rho=28.0, beta=8/3):
    """
    Integrate the Lorenz system using RK45.

    Parameters
    ----------
    t_span : (float, float)  — start and end time
    dt     : float           — output step size
    ic     : array-like      — initial condition [x0, y0, z0]

    Returns
    -------
    t   : (N,) time array
    sol : (N, 3) state array
    """
    if ic is None:
        ic = [0.1, 0.0, 0.0]

    t_eval = np.arange(t_span[0], t_span[1], dt)

    sol = solve_ivp(
        lorenz, t_span, ic,
        method='RK45',
        t_eval=t_eval,
        args=(sigma, rho, beta),
        rtol=1e-10, atol=1e-12
    )
    return sol.t, sol.y.T   # shapes: (N,), (N, 3)


# ─────────────────────────────────────────────
#  Lyapunov Exponents — QR Method
# ─────────────────────────────────────────────
def compute_lyapunov_exponents(sigma=10.0, rho=28.0, beta=8/3,
                                T=50.0, dt=0.01,
                                ic=None):
    """
    Compute all three Lyapunov exponents using the QR decomposition method.

    Theory:
      We evolve the tangent space along the trajectory.
      At each step: Q, R = QR(J @ Q)
      Lyapunov exponents: λ_i = (1/T) * Σ log|R_ii|

    Returns
    -------
    exponents : (3,) array — sorted descending
    """
    if ic is None:
        ic = [0.1, 0.0, 0.0]

    n = 3
    t_eval = np.arange(0, T, dt)
    N      = len(t_eval)

    # Integrate trajectory
    sol = solve_ivp(
        lorenz, (0, T), ic,
        method='RK45', t_eval=t_eval,
        args=(sigma, rho, beta),
        rtol=1e-10, atol=1e-12
    )
    trajectory = sol.y.T   # (N, 3)

    # Initialise orthonormal basis
    Q = np.eye(n)
    log_sum = np.zeros(n)

    for i in range(N - 1):
        J  = lorenz_jacobian(trajectory[i], sigma, rho, beta)
        M  = np.eye(n) + dt * J     # first-order approximation of exp(J*dt)
        Z  = M @ Q
        Q, R = np.linalg.qr(Z)
        log_sum += np.log(np.abs(np.diag(R)))

    exponents = log_sum / T
    return np.sort(exponents)[::-1]


# ─────────────────────────────────────────────
#  Sensitive Dependence Demonstration
# ─────────────────────────────────────────────
def demonstrate_sensitivity(epsilon=1e-8):
    """
    Show how two trajectories with nearly identical initial conditions
    diverge exponentially — the hallmark of chaos.
    """
    ic1 = [0.1, 0.0, 0.0]
    ic2 = [0.1 + epsilon, 0.0, 0.0]

    t, sol1 = simulate_lorenz(t_span=(0, 40), ic=ic1)
    _, sol2 = simulate_lorenz(t_span=(0, 40), ic=ic2)

    distance = np.linalg.norm(sol1 - sol2, axis=1)
    return t, sol1, sol2, distance


# ─────────────────────────────────────────────
#  Plots
# ─────────────────────────────────────────────
def plot_attractor(t, sol, filename="plots/01_lorenz_attractor.png"):
    fig = plt.figure(figsize=(14, 5))
    fig.patch.set_facecolor('#0f0f1a')

    ax1 = fig.add_subplot(131, projection='3d', facecolor='#0f0f1a')
    ax1.plot(sol[:,0], sol[:,1], sol[:,2],
             lw=0.4, alpha=0.9, color='#00d4ff')
    ax1.set_title("Lorenz Attractor (3D)", color='white', pad=10)
    ax1.tick_params(colors='grey')
    for pane in [ax1.xaxis.pane, ax1.yaxis.pane, ax1.zaxis.pane]:
        pane.fill = False

    for ax_flat, (i, j, labels) in zip(
        [fig.add_subplot(132, facecolor='#0f0f1a'),
         fig.add_subplot(133, facecolor='#0f0f1a')],
        [(0, 2, ('x', 'z')), (1, 2, ('y', 'z'))]
    ):
        scatter = ax_flat.scatter(sol[:, i], sol[:, j],
                                   c=t, cmap='plasma', s=0.3, alpha=0.8)
        ax_flat.set_xlabel(labels[0], color='white')
        ax_flat.set_ylabel(labels[1], color='white')
        ax_flat.set_title(f"Projection: {labels[0]}-{labels[1]}", color='white')
        ax_flat.tick_params(colors='grey')
        ax_flat.spines[:].set_color('#333355')

    plt.suptitle("Lorenz Strange Attractor  |  σ=10, ρ=28, β=8/3",
                 color='white', fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {filename}")


def plot_sensitivity(t, sol1, sol2, distance,
                     filename="plots/02_sensitive_dependence.png"):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.patch.set_facecolor('#0f0f1a')

    ax = axes[0]
    ax.set_facecolor('#0f0f1a')
    ax.plot(t, sol1[:, 0], lw=1.0, color='#00d4ff', label='Trajectory 1')
    ax.plot(t, sol2[:, 0], lw=1.0, color='#ff6b6b', label='Trajectory 2  (Δx₀=1e-8)',
            linestyle='--')
    ax.set_xlabel("Time", color='white')
    ax.set_ylabel("x(t)", color='white')
    ax.set_title("Divergence of x(t)", color='white')
    ax.legend(facecolor='#1a1a2e', labelcolor='white', fontsize=9)
    ax.tick_params(colors='grey')
    ax.spines[:].set_color('#333355')

    ax2 = axes[1]
    ax2.set_facecolor('#0f0f1a')
    ax2.semilogy(t, distance + 1e-16, color='#ffd700', lw=1.5)
    ax2.set_xlabel("Time", color='white')
    ax2.set_ylabel("||Δstate||  (log scale)", color='white')
    ax2.set_title("Exponential Divergence — Butterfly Effect", color='white')
    ax2.tick_params(colors='grey')
    ax2.spines[:].set_color('#333355')

    plt.suptitle("Sensitive Dependence on Initial Conditions",
                 color='white', fontsize=13)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {filename}")


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  STEP 1 — Lorenz Chaotic System")
    print("=" * 60)

    # 1. Simulate
    print("\n[1/4] Simulating Lorenz system …")
    t, sol = simulate_lorenz(t_span=(0, 60), dt=0.005)
    np.save("data/lorenz_t.npy",   t)
    np.save("data/lorenz_sol.npy", sol)
    print(f"  Generated {len(t):,} time steps  |  t ∈ [0, 60]")

    # 2. Lyapunov exponents
    print("\n[2/4] Computing Lyapunov exponents (this takes ~30 s) …")
    exponents = compute_lyapunov_exponents(T=50.0, dt=0.005)
    print(f"  λ₁ = {exponents[0]:+.4f}  ← positive → CHAOS")
    print(f"  λ₂ = {exponents[1]:+.4f}  ← near zero → limit cycle direction")
    print(f"  λ₃ = {exponents[2]:+.4f}  ← negative → dissipation")
    print(f"  Lyapunov time = 1/λ₁ = {1/exponents[0]:.2f} time units")
    np.save("data/lyapunov_exponents.npy", exponents)

    # 3. Plots
    print("\n[3/4] Plotting attractor …")
    plot_attractor(t, sol)

    print("\n[4/4] Demonstrating sensitive dependence …")
    t2, s1, s2, dist = demonstrate_sensitivity()
    plot_sensitivity(t2, s1, s2, dist)

    print("\n✅  Step 1 Complete!")
    print("   Data saved in: data/")
    print("   Plots saved in: plots/")
    print("\nNext → Run: python 02_reservoir_computing.py")