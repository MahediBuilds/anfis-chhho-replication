import os, sys, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); import core
INK, MUTED, BLUE, RED, GRID = "#1f2328", "#59636e", "#2f6fb3", "#c0392b", "#d8dee4"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})
A = np.linspace(1.30, 2.0, 1400)
fig, ax = plt.subplots(1, 3, figsize=(10.2, 3.0), gridspec_kw={"width_ratios": [1.25, 1.25, 1]})
# (a) orbit diagram
for a in A:
    x = 0.6
    for _ in range(600): x = a - x * x
    pts = []
    for _ in range(120): x = a - x * x; pts.append(x)
    ax[0].plot([a] * len(pts), pts, ",", color=INK, alpha=0.35)
ax[0].axvline(1.4, color=RED, lw=1); ax[0].text(1.41, 1.85, "a = 1.4", color=RED, fontsize=8)
ax[0].set_xlabel("map parameter a"); ax[0].set_ylabel("x (after transient)"); ax[0].set_title("(a) Orbit diagram of x' = a − x²", fontsize=9, loc="left")
# (b) Lyapunov exponent
lam = [core.lyapunov_quadratic(a, n=4000, burn=500) for a in A[::4]]
ax[1].plot(A[::4], lam, color=BLUE, lw=1); ax[1].axhline(0, color=MUTED, lw=0.8)
ax[1].axvline(1.4, color=RED, lw=1)
l14 = core.lyapunov_quadratic(1.4)
ax[1].annotate(f"λ(1.4) = {l14:.3f} < 0", (1.4, l14), (1.5, -1.25), color=RED, fontsize=8, arrowprops=dict(arrowstyle="->", color=RED, lw=0.8))
ax[1].set_ylim(-1.5, 0.8); ax[1].set_xlabel("map parameter a"); ax[1].set_ylabel("Lyapunov exponent λ"); ax[1].set_title("(b) λ > 0 indicates chaos", fontsize=9, loc="left")
# (c) values actually produced (rescaled to [0,1]) vs uniform
seq = core.QuadraticChaos(0.6)(50000)
ax[2].hist(seq[1000:], bins=40, range=(0, 1), color=INK, density=True)
ax[2].axhline(1, color=BLUE, lw=1, ls="--"); ax[2].text(0.45, 1.3, "uniform U(0,1)", color=BLUE, fontsize=8, bbox=dict(fc="white", ec="none", pad=1))
ax[2].set_xlabel("rescaled value"); ax[2].set_ylabel("density"); ax[2].set_title("(c) Values ChHHO draws (x0 = 0.6)", fontsize=9, loc="left")
for a_ in ax: a_.grid(axis="y", color=GRID, lw=0.5)
os.makedirs(os.path.join(HERE, "figs"), exist_ok=True)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "figs", "fig7_chaos.png"), dpi=220)
print("lyap", l14)
