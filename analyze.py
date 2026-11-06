import os, sys, glob, json
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import core

DATASETS = ["GSPC", "IXIC", "DJI", "NSEI", "BVSP", "N225"]
DSNAME = {"GSPC": "S&P 500", "IXIC": "NASDAQ Comp.", "DJI": "DJIA", "NSEI": "NIFTY 50",
          "BVSP": "Bovespa", "N225": "Nikkei 225"}
ALGOS = ["HHO", "ChHHO", "ChHHO-init", "PSO", "GWO"]
PROTS = {"P1": "train", "P1L": "full", "P2": "ret"}
NEMENYI_Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164}


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for r, i in enumerate(o):
        run = max(run, (m - r) * p[i]); adj[i] = min(1, run)
    return adj


def dm_test(y, f1, f2):
    """Harvey-Leybourne-Newbold corrected DM, squared loss, h = 1. Negative stat => f1 better."""
    d = (y - f1) ** 2 - (y - f2) ** 2
    T = len(d); s = d.mean() / np.sqrt(d.var(ddof=0) / T)
    s *= np.sqrt((T - 1) / T)
    return float(s), float(2 * stats.t.sf(abs(s), T - 1))


def main():
    out = {"cells": {}, "baselines": {}, "tests": {}, "friedman": {}, "corr": {}}
    data = {p: {ds: core.load_dataset(os.path.join(HERE, "data", ds + ".csv"), sc) for ds in DATASETS}
            for p, sc in PROTS.items()}
    preds = {}
    for p in PROTS:
        for ds in DATASETS:
            d = data[p][ds]; te = slice(d["n_tr"], None); y = d["y_raw"][te]; cp = d["c_prev"][te]
            bf = os.path.join(HERE, "results", "base", f"{p}_{ds}.npz")
            if os.path.exists(bf):
                B = np.load(bf)
                for m in ["naive", "OLS", "ANFIS"]:
                    out["baselines"][f"{p}|{ds}|{m}"] = core.metrics(y, B[m][te], cp)
                    preds[(p, ds, m)] = B[m][te]
                L = [core.metrics(y, B["LSTM"][s][te], cp) for s in range(B["LSTM"].shape[0])]
                out["baselines"][f"{p}|{ds}|LSTM"] = {k: float(np.mean([l[k] for l in L])) for k in L[0]}
                out["baselines"][f"{p}|{ds}|LSTM"]["RMSE_sd"] = float(np.std([l["RMSE"] for l in L], ddof=1))
                out["baselines"][f"{p}|{ds}|LSTM"]["RMSE_med"] = float(np.median([l["RMSE"] for l in L]))
                out["baselines"][f"{p}|{ds}|LSTM"]["U2_med"] = float(np.median([l["U2"] for l in L]))
                med = int(np.argsort([l["RMSE"] for l in L])[len(L) // 2])
                preds[(p, ds, "LSTM")] = B["LSTM"][med][te]
                out["baselines"][f"{p}|{ds}|ANFIS"]["train_mse"] = float(B["ANFIS_train_mse"])
            for sch in "AB":
                for a in ALGOS:
                    fs = sorted(glob.glob(os.path.join(HERE, "results", "mh", f"{p}_{ds}_{sch}_{a}_*.npz")))
                    if not fs:
                        continue
                    R = [np.load(f) for f in fs]
                    M = [core.metrics(y, r["pred"][te], cp) for r in R]
                    rm = np.array([m["RMSE"] for m in M]); tr = np.array([float(r["train_mse"]) for r in R])
                    cell = {k: float(np.mean([m[k] for m in M])) for k in M[0]}
                    cell.update(RMSE_sd=float(rm.std(ddof=1)) if len(rm) > 1 else 0.0, RMSE_best=float(rm.min()),
                                RMSE_med=float(np.median(rm)), n=len(rm), train_mse=float(tr.mean()),
                                train_mse_sd=float(tr.std(ddof=1)) if len(tr) > 1 else 0.0,
                                fe=float(np.mean([int(r["fe"]) for r in R])),
                                secs=float(np.mean([float(r["secs"]) for r in R])),
                                runs={"rmse": rm.tolist(), "train": tr.tolist()})
                    out["cells"][f"{p}|{ds}|{sch}|{a}"] = cell
                    med = int(np.argsort(rm)[len(rm) // 2])
                    preds[(p, ds, sch, a)] = R[med]["pred"][te]
                    # train-vs-test: does a better optimum give a better forecast?
                    if len(rm) > 5:
                        rho, pv = stats.spearmanr(tr, rm)
                        out["corr"][f"{p}|{ds}|{sch}|{a}"] = [float(rho), float(pv)]
    # ---- pairwise tests: ChHHO vs others (paired by seed), and vs classical ANFIS
    for p in PROTS:
        for sch in "AB":
            for ds in DATASETS:
                c = out["cells"]; key = lambda a: f"{p}|{ds}|{sch}|{a}"
                if key("ChHHO") not in c:
                    continue
                ch = np.array(c[key("ChHHO")]["runs"]["rmse"]); rows = {}
                pv = []; names = []
                for a in ["HHO", "ChHHO-init", "PSO", "GWO"]:
                    if key(a) not in c: continue
                    o = np.array(c[key(a)]["runs"]["rmse"]); n = min(len(o), len(ch))
                    if n < 6: continue
                    w = stats.wilcoxon(ch[:n], o[:n]); pv.append(w.pvalue); names.append(a)
                    rows[a] = {"median_diff": float(np.median(ch[:n] - o[:n])), "p": float(w.pvalue),
                               "ChHHO_better_runs": int((ch[:n] < o[:n]).sum()), "n": n}
                for a, ph in zip(names, holm(pv)):
                    rows[a]["p_holm"] = float(ph)
                bk = f"{p}|{ds}|ANFIS"
                if bk in out["baselines"] and len(ch) >= 6:
                    ref = out["baselines"][bk]["RMSE"]
                    w = stats.wilcoxon(ch - ref)
                    rows["ANFIS"] = {"median_diff": float(np.median(ch - ref)), "p": float(w.pvalue),
                                     "ChHHO_better_runs": int((ch < ref).sum()), "n": len(ch)}
                d = data[p][ds]; y = d["y_raw"][d["n_tr"]:]
                for ref in ["naive", "ANFIS", "OLS"]:
                    if (p, ds, ref) in preds and (p, ds, sch, "ChHHO") in preds:
                        rows["DM_vs_" + ref] = dm_test(y, preds[(p, ds, sch, "ChHHO")], preds[(p, ds, ref)])
                out["tests"][f"{p}|{ds}|{sch}"] = rows
            # ---- Friedman over datasets
            models = ["naive", "OLS", "ANFIS", "LSTM"] + [a for a in ALGOS if f"{p}|{DATASETS[0]}|{sch}|{a}" in out["cells"]]
            mat = []
            for ds in DATASETS:
                row = []
                for m in models:
                    k1, k2 = f"{p}|{ds}|{sch}|{m}", f"{p}|{ds}|{m}"
                    row.append(out["cells"][k1]["RMSE_med"] if k1 in out["cells"] else out["baselines"].get(k2, {}).get("RMSE_med", out["baselines"].get(k2, {}).get("RMSE", np.nan)))
                mat.append(row)
            mat = np.array(mat)
            if not np.isnan(mat).any():
                ranks = np.array([stats.rankdata(r) for r in mat]).mean(0)
                fr = stats.friedmanchisquare(*mat.T)
                k, N = len(models), len(DATASETS)
                cd = NEMENYI_Q05[k] * np.sqrt(k * (k + 1) / (6 * N))
                R = np.array([stats.rankdata(r) for r in mat])
                out["friedman"][f"{p}|{sch}"] = {"models": models, "avg_rank": ranks.tolist(), "rank_table": R.tolist(),
                                                  "chi2": float(fr.statistic), "p": float(fr.pvalue), "CD": float(cd)}
    # ---- extrapolation: share of test targets outside the training range (P1 level data)
    out["range"] = {}
    for ds in DATASETS:
        d = data["P1"][ds]; n = d["n_tr"]; y = d["y_raw"]
        lo, hi = y[:n].min(), y[:n].max(); t = y[n:]
        out["range"][ds] = {"above": float((t > hi).mean()), "below": float((t < lo).mean()), "test_max_over_train_max": float(t.max() / hi),
                            "train_start": str(d["dates"][0])[:10], "test_start": str(d["dates"][n])[:10], "test_end": str(d["dates"][-1])[:10],
                            "n_train": int(n), "n_test": int(len(t))}
    # ---- chaos diagnostics
    ch = core.QuadraticChaos(0.6); seq = ch(20000)[2000:]
    per = next((q for q in range(1, 1025) if np.abs(seq[q:] - seq[:-q]).max() < 1e-9), None)
    out["chaos"] = {"lyap_1.4": core.lyapunov_quadratic(1.4), "lyap_2.0": core.lyapunov_quadratic(2.0),
                    "period_1.4": per, "distinct_values": int(len(np.unique(np.round(seq, 9))))}
    json.dump(out, open(os.path.join(HERE, "results", "summary.json"), "w"), indent=1)
    print(json.dumps({k: out[k] for k in ["friedman", "chaos"]}, indent=1))


if __name__ == "__main__":
    main()
