import os, sys, json, glob
import numpy as np, matplotlib, matplotlib.ticker
matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import core
from analyze import DATASETS, DSNAME, ALGOS

S = json.load(open(os.path.join(HERE, "results", "summary.json")))
INK, MUTED, GRID = "#1f2328", "#59636e", "#d8dee4"
# categorical order (fixed): HHO, ChHHO, ChHHO-init, PSO, GWO
COL = {"HHO": "#2f6fb3", "ChHHO": "#c0392b", "ChHHO-init": "#e08e45", "PSO": "#3a9e6f", "GWO": "#7a5cad"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})
OUT = os.path.join(HERE, "figs")
os.makedirs(OUT, exist_ok=True)


def u2_runs(p, ds, sch, a):
    d = core.load_dataset(os.path.join(HERE, "data", ds + ".csv"), "train" if p == "P1" else "full")
    te = slice(d["n_tr"], None); y = d["y_raw"][te]; cp = d["c_prev"][te]
    nv = np.sqrt(((y - cp) ** 2).mean())
    return np.array(S["cells"][f"{p}|{ds}|{sch}|{a}"]["runs"]["rmse"]) / nv


# ---------------- Fig 8: U2 distributions, P1, both schemes
fig, axes = plt.subplots(2, 1, figsize=(10.2, 6.4), sharex=True)
for ax, sch, ttl in zip(axes, "AB", ["(a) Scheme A: metaheuristic searches premise parameters, consequents by least squares",
                                     "(b) Scheme B: metaheuristic searches all 44 parameters in [−25, 25] (base-paper setting)"]):
    w = 0.15
    for j, ds in enumerate(DATASETS):
        for k, a in enumerate(ALGOS):
            v = u2_runs("P1", ds, sch, a); x = j + (k - 2) * w
            bp = ax.boxplot(v, positions=[x], widths=w * 0.8, patch_artist=True, showfliers=True,
                            flierprops=dict(marker=".", ms=2, mec=COL[a]), medianprops=dict(color="white", lw=1),
                            boxprops=dict(fc=COL[a], ec=COL[a]), whiskerprops=dict(color=COL[a]), capprops=dict(color=COL[a]))
        for m, mk in [("ANFIS", "D"), ("OLS", "s"), ("LSTM", "^")]:
            v = S["baselines"][f"P1|{ds}|{m}"].get("U2_med", S["baselines"][f"P1|{ds}|{m}"]["U2"])
            ax.plot([j - 0.42, j + 0.42], [v, v], color=INK if m == "ANFIS" else MUTED, lw=1,
                    ls={"ANFIS": "-", "OLS": "--", "LSTM": ":"}[m], label=f"classical {m}" if m == "ANFIS" else m)
    ax.axhline(1, color=INK, lw=0.6, alpha=0.6)
    ax.set_yscale("log"); ax.set_ylabel("Theil U2 = RMSE / RMSE(naive)")
    ax.set_title(ttl, fontsize=9, loc="left"); ax.grid(axis="y", color=GRID, lw=0.5, which="both")
axes[1].set_xticks(range(len(DATASETS))); axes[1].set_xticklabels([DSNAME[d] for d in DATASETS])
h = [plt.Rectangle((0, 0), 1, 1, fc=COL[a]) for a in ALGOS]
hl, ll = axes[0].get_legend_handles_labels()
axes[0].legend(h + hl[:3], ["ANFIS-" + a for a in ALGOS] + ll[:3], ncol=8, fontsize=7.5, frameon=False,
               loc="lower left", bbox_to_anchor=(0, 1.08))
fig.text(0.01, 0.005, "U2 = 1 (thin line): no better than the naive forecast Ĉ(t+1) = C(t). Boxes: 30 independent runs.", fontsize=7.5, color=MUTED)
fig.tight_layout(rect=(0, 0.02, 1, 1)); fig.savefig(os.path.join(OUT, "fig8_u2_boxes.png"), dpi=220); plt.close(fig)

# ---------------- Fig 9: convergence (median of traced runs), scheme B, two datasets
fig, axes = plt.subplots(1, 3, figsize=(10.2, 2.9))
for ax, ds in zip(axes, ["GSPC", "NSEI", "N225"]):
    for a in ALGOS:
        T = [np.load(f)["trace"] for f in sorted(glob.glob(os.path.join(HERE, "results", "mh", f"P1_{ds}_B_{a}_0[0-4].npz")))]
        T = [t for t in T if len(t)]
        if not T: continue
        grid = np.linspace(100, 50000, 400)
        Ys = np.array([np.interp(grid, t[:, 0], t[:, 1]) for t in T])
        ax.plot(grid, np.median(Ys, 0), color=COL[a], lw=1.4 if a == "ChHHO" else 1, label="ANFIS-" + a)
    ref = S["baselines"][f"P1|{ds}|ANFIS"]["train_mse"]
    ax.axhline(ref, color=INK, lw=0.8, ls="-"); ax.text(50000, ref, " classical\n ANFIS", va="center", fontsize=7, color=INK)
    ax.set_yscale("log"); ax.set_title(DSNAME[ds], fontsize=9, loc="left"); ax.set_xlabel("objective evaluations")
    ax.grid(axis="y", color=GRID, lw=0.5, which="both")
axes[0].set_ylabel("best training MSE (scaled)")
axes[0].legend(fontsize=7, frameon=False)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig9_convergence.png"), dpi=220); plt.close(fig)

# ---------------- Fig 10: training MSE vs test U2, scheme B, all runs pooled per dataset (rank-normalised)
fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.1))
for ax, sch in zip(axes, "AB"):
    for a in ALGOS:
        for ds in DATASETS:
            c = S["cells"][f"P1|{ds}|{sch}|{a}"]
            tr = np.array(c["runs"]["train"]); u = u2_runs("P1", ds, sch, a)
            ax.scatter(tr / S["baselines"][f"P1|{ds}|ANFIS"]["train_mse"], u / S["baselines"][f"P1|{ds}|ANFIS"]["U2"],
                       s=6, color=COL[a], alpha=0.55, lw=0, label="ANFIS-" + a if ds == DATASETS[0] else None)
    ax.axhline(1, color=INK, lw=0.6); ax.axvline(1, color=INK, lw=0.6)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g")); ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("training MSE / classical-ANFIS training MSE"); ax.set_ylabel("test RMSE / classical-ANFIS test RMSE")
    ax.set_title(f"({'ab'[sch == 'B']}) Scheme {sch}", fontsize=9, loc="left"); ax.grid(color=GRID, lw=0.4, which="both")
axes[1].legend(fontsize=7, frameon=False, markerscale=2.5)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig10_train_vs_test.png"), dpi=220); plt.close(fig)

# ---------------- Fig 11: leakage effect (P1 vs P1L) on U2 for ANFIS, OLS, LSTM, HHO-B, ChHHO-B
models = [("ANFIS", None), ("LSTM", None), ("HHO", "A"), ("ChHHO", "A"), ("HHO", "B"), ("ChHHO", "B")]
fig, ax = plt.subplots(figsize=(10.2, 3.0))
w = 0.13
for k, (m, sch) in enumerate(models):
    for j, ds in enumerate(DATASETS):
        if sch is None:
            g = lambda p: S["baselines"][f"{p}|{ds}|{m}"].get("U2_med", S["baselines"][f"{p}|{ds}|{m}"]["U2"])
            v1, v2 = g("P1"), g("P1L")
        else:
            v1 = np.median(u2_runs("P1", ds, sch, m)); v2 = np.median(u2_runs("P1L", ds, sch, m))
        x = j + (k - 2.5) * w
        ax.plot([x, x], [v1, v2], color=MUTED, lw=0.8)
        ax.plot(x, v1, "o", ms=4, mfc="white", mec=INK, mew=0.9)
        ax.plot(x, v2, "o", ms=4, color=INK)
ax.axhline(1, color=INK, lw=0.6); ax.set_yscale("log")
ax.set_xticks(range(len(DATASETS))); ax.set_xticklabels([DSNAME[d] for d in DATASETS])
ax.set_ylabel("Theil U2 (median run)"); ax.grid(axis="y", color=GRID, lw=0.5, which="both")
fig.text(0.01, 0.01, "Within each index, left to right: classical ANFIS, LSTM, ANFIS-HHO (A), ANFIS-ChHHO (A), ANFIS-HHO (B), ANFIS-ChHHO (B).",
         fontsize=7.5, color=MUTED)
ax.plot([], [], "o", mfc="white", mec=INK, label="leak-free scaling (train only)")
ax.plot([], [], "o", color=INK, label="look-ahead scaling (full series)")
ax.legend(fontsize=7.5, frameon=False, ncol=2, loc="upper right")
fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig(os.path.join(OUT, "fig11_leakage.png"), dpi=220); plt.close(fig)
print("figures written")
