"""
=============================================================
  CHAOS × ML — Interactive Research Dashboard
  Streamlit App
=============================================================
Run with:
    streamlit run app.py
"""

import streamlit as st
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.colors as mcolors
from scipy.integrate import solve_ivp
from scipy.signal import welch
import time, warnings
warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────
st.set_page_config(
    page_title="Chaos × ML Dashboard",
    page_icon="🌀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Global dark theme CSS ──────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');

html, body, [class*="css"] {
    background-color: #070714 !important;
    color: #e0e0ff !important;
}

/* Main background */
.stApp { background: #070714; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0d0d24 !important;
    border-right: 1px solid #1e1e4a;
}
section[data-testid="stSidebar"] * { color: #c0c0e0 !important; }

/* Headings */
h1 { font-family: 'Syne', sans-serif !important; font-weight: 800 !important;
     font-size: 2.4rem !important; letter-spacing: -1px;
     background: linear-gradient(120deg, #00d4ff, #7b5ea7, #ff6b6b);
     -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
h2, h3 { font-family: 'Syne', sans-serif !important; color: #a0c4ff !important; }

/* Metric boxes */
[data-testid="metric-container"] {
    background: #0f0f2a; border: 1px solid #1e1e5a;
    border-radius: 10px; padding: 1rem;
}
[data-testid="metric-container"] label { color: #6060aa !important; font-size: 0.75rem; }
[data-testid="metric-container"] [data-testid="metric-value"] {
    color: #00d4ff !important; font-family: 'Space Mono', monospace;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] { gap: 8px; background: #0d0d24;
    border-bottom: 1px solid #1e1e4a; }
.stTabs [data-baseweb="tab"] { background: transparent;
    color: #5050aa !important; border-radius: 6px 6px 0 0;
    font-family: 'Space Mono', monospace; font-size: 0.8rem; }
.stTabs [aria-selected="true"] { background: #1a1a3a !important;
    color: #00d4ff !important; border-bottom: 2px solid #00d4ff !important; }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #1a1a4a, #0d2040);
    border: 1px solid #00d4ff44; color: #00d4ff;
    font-family: 'Space Mono', monospace; font-size: 0.8rem;
    border-radius: 6px; transition: all 0.2s;
}
.stButton > button:hover { border-color: #00d4ff; background: #0d2040;
    box-shadow: 0 0 15px #00d4ff33; }

/* Sliders */
.stSlider [data-baseweb="slider"] div[role="slider"] {
    background-color: #00d4ff !important; }

/* Info boxes */
.stInfo { background: #0a0a2a; border: 1px solid #1e1e5a; border-radius: 8px; }
div[data-testid="stMarkdownContainer"] code {
    background: #0f0f2a; color: #00d4ff;
    border: 1px solid #1e1e4a; border-radius: 4px;
    font-family: 'Space Mono', monospace;
}

/* Divider */
hr { border-color: #1e1e4a !important; }

/* Select boxes */
.stSelectbox [data-baseweb="select"] { background: #0d0d24 !important; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
#  PHYSICS & MATH FUNCTIONS
# ═══════════════════════════════════════════════════════════

def lorenz_ode(t, state, sigma, rho, beta):
    x, y, z = state
    return [sigma*(y-x), x*(rho-z)-y, x*y-beta*z]

def rossler_ode(t, state, a=0.2, b=0.2, c=5.7):
    x, y, z = state
    return [-y-z, x+a*y, b+z*(x-c)]

@st.cache_data(show_spinner=False)
def simulate_system(system, sigma, rho, beta, t_end, dt, ic):
    t_eval = np.arange(0, t_end, dt)
    if system == "Lorenz":
        ode = lorenz_ode
        args = (sigma, rho, beta)
    else:
        ode = rossler_ode
        args = (0.2, 0.2, 5.7)

    sol = solve_ivp(ode, (0, t_end), list(ic), method='RK45',
                    t_eval=t_eval, args=args, rtol=1e-9, atol=1e-11)
    return sol.t, sol.y.T

@st.cache_data(show_spinner=False)
def compute_lyapunov(sigma, rho, beta, T=40.0, dt=0.01):
    ic = [0.1, 0.0, 0.0]
    t_eval = np.arange(0, T, dt)
    sol = solve_ivp(lorenz_ode, (0, T), ic, method='RK45',
                    t_eval=t_eval, args=(sigma, rho, beta),
                    rtol=1e-10, atol=1e-12)
    traj = sol.y.T
    n = 3
    Q = np.eye(n)
    log_sum = np.zeros(n)
    for i in range(len(traj)-1):
        x, y, z = traj[i]
        J = np.array([[-sigma, sigma, 0],
                      [rho-z,  -1,   -x],
                      [y,       x, -beta]])
        M = np.eye(n) + dt * J
        Z = M @ Q
        Q, R = np.linalg.qr(Z)
        log_sum += np.log(np.abs(np.diag(R)) + 1e-30)
    return np.sort(log_sum / T)[::-1]

def lorenz_jacobian(state, sigma, rho, beta):
    x, y, z = state
    return np.array([[-sigma, sigma, 0],
                     [rho-z,  -1,   -x],
                     [y,       x, -beta]])

# ESN — lightweight, fast
class MiniESN:
    def __init__(self, N=400, rho=0.95, sparsity=0.85, alpha=0.3, seed=0):
        rng = np.random.RandomState(seed)
        W = rng.randn(N, N)
        W[rng.rand(N, N) < sparsity] = 0.0
        ev = np.max(np.abs(np.linalg.eigvals(W)))
        self.W   = W * (rho / (ev + 1e-8))
        self.Win = rng.randn(N, 3)
        self.N   = N
        self.alpha = alpha
        self.Wout = None

    def _states(self, inputs):
        x = np.zeros(self.N)
        out = []
        for u in inputs:
            x = (1-self.alpha)*x + self.alpha*np.tanh(self.W@x + self.Win@u)
            out.append(x.copy())
        return np.array(out)

    def fit(self, data, warmup=100, ridge=1e-5):
        X = self._states(data[:-1])
        Y = data[1:]
        Xw, Yw = X[warmup:], Y[warmup:]
        I = np.eye(self.N)
        self.Wout = (Yw.T @ Xw) @ np.linalg.inv(Xw.T @ Xw + ridge*I)

    def generate(self, seed, n):
        x = np.zeros(self.N)
        for u in seed:
            x = (1-self.alpha)*x + self.alpha*np.tanh(self.W@x + self.Win@u)
        out, u = [], seed[-1]
        for _ in range(n):
            x = (1-self.alpha)*x + self.alpha*np.tanh(self.W@x + self.Win@u)
            y = self.Wout @ x
            out.append(y); u = y
        return np.array(out)

# Neural ODE — pure numpy RK4 (no torch needed for dashboard)
class MiniNeuralODE:
    """Tiny NN that approximates dz/dt, integrated with RK4."""
    def __init__(self, hidden=64, seed=1):
        rng = np.random.RandomState(seed)
        s = 0.1
        self.W1 = rng.randn(hidden, 3) * s
        self.b1 = np.zeros(hidden)
        self.W2 = rng.randn(hidden, hidden) * s
        self.b2 = np.zeros(hidden)
        self.W3 = rng.randn(3, hidden) * s
        self.b3 = np.zeros(3)

    def forward(self, z):
        h = np.tanh(self.W1 @ z + self.b1)
        h = np.tanh(self.W2 @ h + self.b2)
        return self.W3 @ h + self.b3

    def rk4_step(self, z, dt):
        k1 = self.forward(z)
        k2 = self.forward(z + dt/2*k1)
        k3 = self.forward(z + dt/2*k2)
        k4 = self.forward(z + dt*k3)
        return z + dt/6*(k1+2*k2+2*k3+k4)

    def train(self, data, dt, n_epochs=300, lr=5e-3):
        """Simple gradient descent via finite differences (no autograd needed)."""
        params = [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]
        eps = 1e-4

        prog = st.progress(0, text="Training Neural ODE…")
        losses = []
        for epoch in range(n_epochs):
            # Pick random mini-batch of 20 consecutive pairs
            idx = np.random.randint(0, len(data)-1, 20)
            z0  = data[idx]
            z1  = data[idx+1]
            pred = np.array([self.rk4_step(z, dt) for z in z0])
            loss = np.mean((pred - z1)**2)
            losses.append(loss)

            # Finite-difference gradient on W3 only (fast approximate update)
            for i in range(self.W3.shape[0]):
                for j in range(self.W3.shape[1]):
                    orig = self.W3[i,j]
                    self.W3[i,j] = orig + eps
                    pred_p = np.array([self.rk4_step(z, dt) for z in z0])
                    lp = np.mean((pred_p - z1)**2)
                    self.W3[i,j] = orig - eps
                    pred_m = np.array([self.rk4_step(z, dt) for z in z0])
                    lm = np.mean((pred_m - z1)**2)
                    self.W3[i,j] = orig - lr * (lp-lm)/(2*eps)

            if (epoch+1) % 10 == 0:
                prog.progress((epoch+1)/n_epochs,
                              text=f"Training Neural ODE… loss={loss:.5f}")
        prog.empty()
        return losses

    def rollout(self, z0, n, dt):
        traj = [z0]
        z = z0.copy()
        for _ in range(n):
            z = self.rk4_step(z, dt)
            traj.append(z.copy())
        return np.array(traj)


# ═══════════════════════════════════════════════════════════
#  PLOT HELPERS  (dark theme)
# ═══════════════════════════════════════════════════════════
BG    = '#070714'
PANEL = '#0d0d24'
CYAN  = '#00d4ff'
RED   = '#ff6b6b'
GOLD  = '#ffd700'
GREEN = '#69ff47'
PURPLE= '#b06bff'

def dark_fig(w=12, h=4, subplots=(1,1), projection=None):
    fig, axes = plt.subplots(*subplots, figsize=(w, h),
                              subplot_kw={'projection': projection} if projection else {})
    fig.patch.set_facecolor(BG)
    if hasattr(axes, '__iter__'):
        for ax in np.array(axes).flatten():
            ax.set_facecolor(PANEL)
            ax.tick_params(colors='#5050aa')
            ax.spines[:].set_color('#1e1e4a')
    else:
        axes.set_facecolor(PANEL)
        axes.tick_params(colors='#5050aa')
        axes.spines[:].set_color('#1e1e4a')
    return fig, axes

def style_ax(ax, xlabel='', ylabel='', title=''):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors='#5050aa', labelsize=8)
    ax.spines[:].set_color('#1e1e4a')
    if xlabel: ax.set_xlabel(xlabel, color='#6060aa', fontsize=8)
    if ylabel: ax.set_ylabel(ylabel, color='#6060aa', fontsize=8)
    if title:  ax.set_title(title,   color='#a0c4ff', fontsize=9, pad=6)

def plot_3d_attractor(sol, color=CYAN, title="Lorenz Attractor"):
    fig = plt.figure(figsize=(5, 4))
    fig.patch.set_facecolor(BG)
    ax = fig.add_subplot(111, projection='3d', facecolor=PANEL)
    ax.plot(sol[:,0], sol[:,1], sol[:,2], lw=0.3, color=color, alpha=0.85)
    ax.set_title(title, color='#a0c4ff', fontsize=9)
    ax.tick_params(colors='#5050aa', labelsize=6)
    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.fill = False
        pane.set_edgecolor('#1e1e4a')
    ax.xaxis.label.set_color('#5050aa')
    ax.yaxis.label.set_color('#5050aa')
    ax.zaxis.label.set_color('#5050aa')
    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🌀 Control Panel")
    st.markdown("---")

    st.markdown("### System")
    system = st.selectbox("Dynamical System", ["Lorenz", "Rössler"])

    st.markdown("### Lorenz Parameters")
    sigma = st.slider("σ (sigma)", 1.0, 20.0, 10.0, 0.5,
                      help="Controls coupling between x and y")
    rho   = st.slider("ρ (rho)",   1.0, 50.0, 28.0, 0.5,
                      help="Rayleigh number — chaos onset near ρ≈24.7")
    beta  = st.slider("β (beta)",  0.5,  5.0,  8/3, 0.1,
                      help="Geometric parameter — classic: 8/3")

    st.markdown("### Simulation")
    t_end = st.slider("Duration (time units)", 10, 100, 40, 5)
    dt    = st.select_slider("Step size dt", [0.02, 0.01, 0.005], value=0.01)

    st.markdown("### Initial Condition")
    col1, col2, col3 = st.columns(3)
    ic_x = col1.number_input("x₀", value=0.1, step=0.1, format="%.2f")
    ic_y = col2.number_input("y₀", value=0.0, step=0.1, format="%.2f")
    ic_z = col3.number_input("z₀", value=0.0, step=0.1, format="%.2f")

    st.markdown("---")
    run_btn = st.button("▶  Run Simulation", use_container_width=True)

    st.markdown("---")
    st.markdown("### ML Models")
    run_esn  = st.checkbox("Echo State Network",  value=True)
    run_node = st.checkbox("Neural ODE",          value=True)
    esn_N    = st.slider("ESN Reservoir Size", 100, 800, 400, 100)
    train_btn = st.button("🧠  Train Models", use_container_width=True)

    st.markdown("---")
    st.caption("Built with Streamlit + NumPy + SciPy")
    st.caption("Research Dashboard — Ivy League Project")


# ═══════════════════════════════════════════════════════════
#  SESSION STATE
# ═══════════════════════════════════════════════════════════
if 'sol' not in st.session_state:
    st.session_state.sol = None
if 'esn_gen' not in st.session_state:
    st.session_state.esn_gen = None
if 'node_traj' not in st.session_state:
    st.session_state.node_traj = None
if 'lyap' not in st.session_state:
    st.session_state.lyap = None
if 'trained' not in st.session_state:
    st.session_state.trained = False


# ═══════════════════════════════════════════════════════════
#  RUN SIMULATION
# ═══════════════════════════════════════════════════════════
if run_btn or st.session_state.sol is None:
    with st.spinner("Integrating ODE…"):
        t, sol = simulate_system(system, sigma, rho, beta,
                                  t_end, dt, (ic_x, ic_y, ic_z))
        st.session_state.sol   = sol
        st.session_state.t     = t
        st.session_state.sigma = sigma
        st.session_state.rho   = rho
        st.session_state.beta  = beta
        st.session_state.trained = False
        st.session_state.lyap  = None

sol = st.session_state.sol
t   = st.session_state.t


# ═══════════════════════════════════════════════════════════
#  TRAIN MODELS
# ═══════════════════════════════════════════════════════════
if train_btn:
    mean_ = sol.mean(0); std_ = sol.std(0)
    data_n = (sol - mean_) / std_

    if run_esn:
        with st.spinner("Training Echo State Network…"):
            esn = MiniESN(N=esn_N, seed=42)
            esn.fit(data_n, warmup=100)
            seed_len = min(300, len(data_n)//4)
            gen = esn.generate(data_n[:seed_len], n=min(3000, len(data_n)))
            # De-normalise
            st.session_state.esn_gen  = gen * std_ + mean_
            st.session_state.esn_seed = seed_len

    if run_node:
        with st.spinner("Training Neural ODE (fast 300-epoch version)…"):
            node = MiniNeuralODE(hidden=48, seed=2)
            node.train(data_n, dt=float(dt), n_epochs=300, lr=3e-3)
            z0   = data_n[0]
            roll = node.rollout(z0, n=min(3000, len(data_n)-1), dt=float(dt))
            st.session_state.node_traj = roll * std_ + mean_

    st.session_state.trained = True
    st.success("✅ Models trained!")


# ═══════════════════════════════════════════════════════════
#  HEADER
# ═══════════════════════════════════════════════════════════
st.markdown("# Chaos × ML")
st.markdown(
    "<p style='font-family:Space Mono,monospace; color:#5050aa; font-size:0.85rem; margin-top:-1rem;'>"
    "Predicting Chaotic Systems with Neural ODEs & Reservoir Computing"
    "</p>", unsafe_allow_html=True
)

# ── Key metrics row ──────────────────────────────────────
with st.spinner("Computing Lyapunov exponents…"):
    if st.session_state.lyap is None or (
        st.session_state.get('sigma') != sigma or
        st.session_state.get('rho')   != rho   or
        st.session_state.get('beta')  != beta
    ):
        le = compute_lyapunov(sigma, rho, beta)
        st.session_state.lyap = le
    else:
        le = st.session_state.lyap

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("λ₁ (Lyapunov)", f"{le[0]:+.4f}",
          delta="CHAOS ✓" if le[0] > 0 else "no chaos",
          delta_color="normal" if le[0] > 0 else "inverse")
m2.metric("λ₂", f"{le[1]:+.4f}")
m3.metric("λ₃", f"{le[2]:+.4f}")
m4.metric("Lyapunov Time", f"{1/le[0]:.2f}" if le[0] > 0.01 else "∞",
          help="1/λ₁ — prediction horizon bound")
ky_dim = 2 + le[0] / abs(le[2]) if le[2] != 0 else 2.0
m5.metric("KY Dimension", f"{ky_dim:.3f}", help="Kaplan-Yorke fractal dimension")

st.markdown("---")


# ═══════════════════════════════════════════════════════════
#  TABS
# ═══════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🌀 Attractor", "🦋 Chaos Theory", "🧠 ML Models", "📊 Analysis", "📐 Math Notes"
])


# ──────────────────────────────────────────────────────────
#  TAB 1 — ATTRACTOR
# ──────────────────────────────────────────────────────────
with tab1:
    col_a, col_b = st.columns([1, 1])

    with col_a:
        st.markdown("#### 3D Phase Space")
        fig3d = plot_3d_attractor(sol, color=CYAN,
                                   title=f"{system} Attractor  (σ={sigma}, ρ={rho}, β={beta:.2f})")
        st.pyplot(fig3d, use_container_width=True)
        plt.close()

    with col_b:
        st.markdown("#### Time Series")
        fig, axes = plt.subplots(3, 1, figsize=(6, 5), sharex=True)
        fig.patch.set_facecolor(BG)
        labels = ['x(t)', 'y(t)', 'z(t)']
        colors = [CYAN, RED, GREEN]
        for i, (ax, lbl, col) in enumerate(zip(axes, labels, colors)):
            ax.plot(t, sol[:, i], lw=0.6, color=col, alpha=0.9)
            style_ax(ax, ylabel=lbl)
            ax.set_facecolor(PANEL)
        axes[-1].set_xlabel("Time", color='#5050aa', fontsize=8)
        fig.suptitle("State Variables vs Time", color='#a0c4ff', fontsize=9, y=1.01)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    # 2D projections
    st.markdown("#### Phase Plane Projections")
    fc, ac = plt.subplots(1, 3, figsize=(13, 3))
    fc.patch.set_facecolor(BG)
    projs = [(0,1,'x','y'), (0,2,'x','z'), (1,2,'y','z')]
    cms   = ['plasma', 'viridis', 'inferno']
    for ax, (i, j, xl, yl), cm in zip(ac, projs, cms):
        sc = ax.scatter(sol[:,i], sol[:,j], c=t, cmap=cm, s=0.5, alpha=0.8)
        style_ax(ax, xlabel=xl, ylabel=yl, title=f"{xl}-{yl} projection")
    fc.suptitle("2D Projections (colour = time)", color='#a0c4ff', fontsize=9)
    plt.tight_layout()
    st.pyplot(fc, use_container_width=True)
    plt.close()


# ──────────────────────────────────────────────────────────
#  TAB 2 — CHAOS THEORY
# ──────────────────────────────────────────────────────────
with tab2:
    st.markdown("#### Butterfly Effect — Sensitive Dependence on Initial Conditions")
    epsilon = st.select_slider("Perturbation ε",
                                options=[1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2],
                                value=1e-8,
                                format_func=lambda x: f"{x:.0e}")

    ic1 = [ic_x, ic_y, ic_z]
    ic2 = [ic_x + epsilon, ic_y, ic_z]

    with st.spinner("Simulating twin trajectories…"):
        t1, s1 = simulate_system(system, sigma, rho, beta, min(t_end, 40), dt, ic1)
        t2, s2 = simulate_system(system, sigma, rho, beta, min(t_end, 40), dt, ic2)

    dist = np.linalg.norm(s1 - s2, axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 3.5))
    fig.patch.set_facecolor(BG)

    ax = axes[0]
    ax.plot(t1, s1[:,0], color=CYAN, lw=0.8, label='Trajectory 1', alpha=0.9)
    ax.plot(t2, s2[:,0], color=RED,  lw=0.8, label=f'Trajectory 2 (Δx₀={epsilon:.0e})',
            linestyle='--', alpha=0.85)
    style_ax(ax, xlabel='Time', ylabel='x(t)', title='Divergence of x(t)')
    ax.legend(facecolor='#0d0d24', labelcolor='white', fontsize=8)

    ax2 = axes[1]
    ax2.semilogy(t1, dist + 1e-20, color=GOLD, lw=1.2)
    if le[0] > 0.01:
        t_fit = t1[:len(t1)//3]
        exp_fit = epsilon * np.exp(le[0] * t_fit)
        ax2.semilogy(t_fit, exp_fit, color=PURPLE, lw=1.0, linestyle=':',
                     label=f'e^(λ₁t), λ₁={le[0]:.3f}')
        ax2.legend(facecolor='#0d0d24', labelcolor='white', fontsize=8)
    style_ax(ax2, xlabel='Time', ylabel='‖Δstate‖  (log)', title='Exponential Divergence')

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    st.markdown("---")
    st.markdown("#### Lyapunov Spectrum vs ρ (bifurcation scan)")

    with st.spinner("Scanning ρ values…"):
        rho_vals = np.linspace(1, 50, 40)
        lambda1s = []
        for r in rho_vals:
            le_r = compute_lyapunov(sigma, r, beta, T=25.0, dt=0.02)
            lambda1s.append(le_r[0])

    fig2, ax3 = plt.subplots(figsize=(10, 3))
    fig2.patch.set_facecolor(BG)
    ax3.axhline(0, color='#3a3a6a', lw=0.8, linestyle='--')
    ax3.axvline(rho, color=GOLD, lw=1.0, linestyle=':', alpha=0.7, label=f'Current ρ={rho}')
    colors_b = [CYAN if l > 0 else RED for l in lambda1s]
    ax3.bar(rho_vals, lambda1s, width=0.9, color=colors_b, alpha=0.8)
    style_ax(ax3, xlabel='ρ (rho)', ylabel='λ₁ (max Lyapunov)',
             title='Lyapunov Exponent vs ρ  |  Blue=Chaos, Red=Stable')
    ax3.legend(facecolor='#0d0d24', labelcolor='white', fontsize=8)
    plt.tight_layout()
    st.pyplot(fig2, use_container_width=True)
    plt.close()


# ──────────────────────────────────────────────────────────
#  TAB 3 — ML MODELS
# ──────────────────────────────────────────────────────────
with tab3:
    if not st.session_state.trained:
        st.info("👈  Adjust settings in the sidebar, then click **🧠 Train Models** to see predictions.")
        st.markdown("""
| Model | Method | Training |
|---|---|---|
| **Echo State Network** | Reservoir Computing | Ridge Regression (closed-form, instant) |
| **Neural ODE** | Continuous dynamics | Gradient descent on vector field f_θ(z) |
        """)
    else:
        true = sol
        labels_= ['x', 'y', 'z']
        colors_= [CYAN, RED, GREEN]

        if st.session_state.esn_gen is not None:
            st.markdown("#### Echo State Network — Autonomous Generation")
            gen   = st.session_state.esn_gen
            T_cmp = min(len(true), len(gen), 2000)

            fig, axes = plt.subplots(3, 1, figsize=(13, 6), sharex=True)
            fig.patch.set_facecolor(BG)
            for i, ax in enumerate(axes):
                ax.plot(t[:T_cmp], true[:T_cmp, i],
                        color=colors_[i], lw=0.9, label='True', alpha=0.9)
                ax.plot(np.arange(T_cmp)*float(dt), gen[:T_cmp, i],
                        color=GOLD, lw=0.7, linestyle='--', label='ESN', alpha=0.85)
                style_ax(ax, ylabel=labels_[i])
                ax.legend(facecolor='#0d0d24', labelcolor='white', fontsize=7)
            axes[-1].set_xlabel("Time", color='#5050aa', fontsize=8)
            fig.suptitle(f"ESN Autonomous Prediction  (N={esn_N} reservoir neurons)",
                         color='#a0c4ff', fontsize=9)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

            # 3D comparison
            col1, col2 = st.columns(2)
            with col1:
                fig3 = plot_3d_attractor(true[:3000], CYAN, "True Attractor")
                st.pyplot(fig3, use_container_width=True); plt.close()
            with col2:
                fig4 = plot_3d_attractor(gen[:3000], GOLD, "ESN Learned Attractor")
                st.pyplot(fig4, use_container_width=True); plt.close()

        if st.session_state.node_traj is not None:
            st.markdown("#### Neural ODE — Rollout")
            node_r = st.session_state.node_traj
            T_n    = min(len(true), len(node_r), 2000)

            fig, axes = plt.subplots(3, 1, figsize=(13, 6), sharex=True)
            fig.patch.set_facecolor(BG)
            for i, ax in enumerate(axes):
                ax.plot(t[:T_n], true[:T_n, i],
                        color=colors_[i], lw=0.9, label='True')
                ax.plot(t[:T_n], node_r[:T_n, i],
                        color=PURPLE, lw=0.7, linestyle='--', label='Neural ODE')
                style_ax(ax, ylabel=labels_[i])
                ax.legend(facecolor='#0d0d24', labelcolor='white', fontsize=7)
            axes[-1].set_xlabel("Time", color='#5050aa', fontsize=8)
            fig.suptitle("Neural ODE  dz/dt = f_θ(z)  — Learned Continuous Dynamics",
                         color='#a0c4ff', fontsize=9)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()


# ──────────────────────────────────────────────────────────
#  TAB 4 — ANALYSIS
# ──────────────────────────────────────────────────────────
with tab4:
    st.markdown("#### Power Spectral Density")
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5), sharey=True)
    fig.patch.set_facecolor(BG)
    for i, ax in enumerate(axes):
        f_, P_ = welch(sol[:, i], fs=1/float(dt), nperseg=512)
        ax.semilogy(f_, P_, color=[CYAN, RED, GREEN][i], lw=1.2)
        if st.session_state.esn_gen is not None:
            g = st.session_state.esn_gen
            fg, Pg = welch(g[:len(sol), i], fs=1/float(dt), nperseg=512)
            ax.semilogy(fg, Pg, color=GOLD, lw=0.9, linestyle='--', label='ESN')
        if st.session_state.node_traj is not None:
            n_ = st.session_state.node_traj
            fn, Pn = welch(n_[:len(sol), i], fs=1/float(dt), nperseg=512)
            ax.semilogy(fn, Pn, color=PURPLE, lw=0.9, linestyle=':', label='Neural ODE')
        style_ax(ax, xlabel='Frequency (Hz)', ylabel='PSD' if i==0 else '',
                 title=f"dim {['x','y','z'][i]}")
        if i == 0:
            ax.legend(facecolor='#0d0d24', labelcolor='white', fontsize=7)
    fig.suptitle("Power Spectral Density — Frequency Domain Analysis", color='#a0c4ff')
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    st.markdown("---")
    st.markdown("#### Return Map (Poincaré Section  z̤ = 27)")

    # Local maxima of z → return map
    zvals = sol[:, 2]
    maxima = []
    for i in range(1, len(zvals)-1):
        if zvals[i] > zvals[i-1] and zvals[i] > zvals[i+1] and zvals[i] > 20:
            maxima.append(zvals[i])
    maxima = np.array(maxima)

    fig_r, ax_r = plt.subplots(figsize=(5, 4))
    fig_r.patch.set_facecolor(BG)
    if len(maxima) > 2:
        ax_r.scatter(maxima[:-1], maxima[1:], s=4, color=CYAN, alpha=0.7)
        style_ax(ax_r, xlabel='zₙ (n-th max)', ylabel='zₙ₊₁ (next max)',
                 title='Poincaré Return Map — Lorenz')
    else:
        ax_r.text(0.5, 0.5, "Not enough maxima\n(increase duration)",
                  ha='center', va='center', color='grey', transform=ax_r.transAxes)
        style_ax(ax_r, title='Poincaré Return Map')
    plt.tight_layout()

    col_r1, col_r2 = st.columns([1, 2])
    with col_r1:
        st.pyplot(fig_r, use_container_width=True)
        plt.close()
        st.markdown(f"""
**Statistics:**
- Max z-values found: `{len(maxima)}`
- Mean: `{maxima.mean():.2f}`
- Std:  `{maxima.std():.2f}`
        """)

    with col_r2:
        st.markdown("""
**What is a Poincaré Return Map?**

The Poincaré map reduces a continuous trajectory to a discrete map.
For the Lorenz system, we record successive local maxima of z.

If the system were periodic: all points would land on a fixed point.
The **tent-map like shape** you see proves the dynamics are chaotic —
it's essentially a 1D chaotic map hiding inside the 3D flow.

This is why even 1D maps like f(x) = 4x(1-x) capture the essence of chaos.
        """)


# ──────────────────────────────────────────────────────────
#  TAB 5 — MATH NOTES
# ──────────────────────────────────────────────────────────
with tab5:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Lorenz System")
        st.markdown(r"""
The Lorenz ODE system:

$$\frac{dx}{dt} = \sigma(y - x)$$

$$\frac{dy}{dt} = x(\rho - z) - y$$

$$\frac{dz}{dt} = xy - \beta z$$

**Current parameters:**
- $\sigma$ = """ + f"{sigma}" + r""" (Prandtl number)
- $\rho$ = """ + f"{rho}" + r""" (Rayleigh number)
- $\beta$ = """ + f"{beta:.3f}" + r"""

**Chaos onset:** $\rho > \rho_c \approx 24.74$
        """)

        st.markdown("### Lyapunov Exponents")
        st.markdown(r"""
Measure exponential divergence rate of nearby trajectories.

$$\lambda_i = \lim_{T\to\infty} \frac{1}{T} \ln \| \delta z_i(T) \|$$

Computed via **QR decomposition** of the tangent map:

$$Q_{t+1}, R_{t+1} = \text{QR}(J_t \cdot Q_t)$$

$$\lambda_i \approx \frac{1}{T} \sum_t \ln |R_{ii}^{(t)}|$$

**Current values:**
""" + f"- λ₁ = `{le[0]:+.4f}` {'← **CHAOS**' if le[0] > 0 else '← stable'}\n- λ₂ = `{le[1]:+.4f}`\n- λ₃ = `{le[2]:+.4f}`")

        st.markdown("### Kaplan-Yorke Dimension")
        st.markdown(r"""
Estimates the **fractal dimension** of the attractor:

$$d_{KY} = j + \frac{\sum_{i=1}^{j} \lambda_i}{|\lambda_{j+1}|}$$

where $j$ is the largest index with $\sum_{i=1}^j \lambda_i > 0$.

**Current:** $d_{KY}$ = """ + f"`{ky_dim:.4f}`" + r"""

For the classic Lorenz attractor: $d_{KY} \approx 2.06$
        """)

    with col2:
        st.markdown("### Echo State Network")
        st.markdown(r"""
**Reservoir dynamics:**

$$x(t+1) = (1-\alpha)x(t) + \alpha \tanh(W x(t) + W_{in} u(t))$$

**Output (trained):**

$$\hat{y}(t) = W_{out} x(t)$$

**Training via Ridge Regression:**

$$W_{out} = YX^T(XX^T + \lambda I)^{-1}$$

**Echo State Property:** Requires spectral radius $\rho(W) < 1$
so reservoir has *fading memory*.

**Current config:** N = """ + f"`{esn_N}`" + r""" neurons, $\rho$ = `0.95`
        """)

        st.markdown("### Neural ODE")
        st.markdown(r"""
Learns a continuous vector field $f_\theta: \mathbb{R}^d \to \mathbb{R}^d$:

$$\frac{dz}{dt} = f_\theta(z, t)$$

**Prediction:** solve the ODE forward from $z_0$:

$$z(T) = z_0 + \int_0^T f_\theta(z(t), t)\, dt$$

**Integration:** 4th-order Runge-Kutta:

$$z_{n+1} = z_n + \frac{h}{6}(k_1 + 2k_2 + 2k_3 + k_4)$$

**Backprop:** via the **Adjoint method** (Chen et al. 2018):

$$\frac{da}{dt} = -a^T \frac{\partial f_\theta}{\partial z}$$

Memory-efficient: $O(1)$ vs $O(T)$ for BPTT.
        """)

        st.markdown("### References")
        st.markdown("""
1. Lorenz (1963) — *Deterministic Nonperiodic Flow*
2. Jaeger (2001) — *Echo State Networks*
3. Chen et al. (2018) — *Neural ODEs*, NeurIPS
4. Grassberger & Procaccia (1983) — *Correlation Dimension*
5. Kaplan & Yorke (1979) — *Chaotic behavior*
        """)