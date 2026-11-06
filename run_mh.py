import ctypes as _ct
try:  # run at idle priority so the laptop stays responsive (inherited by worker processes)
    _ct.windll.kernel32.SetPriorityClass(_ct.windll.kernel32.GetCurrentProcess(), 0x40)
except Exception:
    pass
import os
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[v] = "1"
import sys, json, time, itertools
import numpy as np
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import core

DATASETS = ["GSPC", "IXIC", "DJI", "NSEI", "BVSP", "N225"]
ALGOS = ["HHO", "ChHHO", "ChHHO-init", "PSO", "GWO"]
SCHEMES = ["A", "B"]
PROTS = {"P1": "train", "P1L": "full", "P2": "ret"}
RUNS = int(os.environ.get("RUNS", 30))
BUDGET = int(os.environ.get("BUDGET", 50000))
OUT = os.path.join(HERE, "results", "mh")


def task(args):
    prot, ds, sch, algo, run = args
    fn = os.path.join(OUT, f"{prot}_{ds}_{sch}_{algo}_{run:02d}.npz")
    if os.path.exists(fn):
        return fn
    d = core.load_dataset(os.path.join(HERE, "data", ds + ".csv"), PROTS[prot])
    n = d["n_tr"]
    obj = core.Objective(d["X"][:n], d["y"][:n], sch)
    rng = np.random.default_rng(1000 + run)          # common random numbers across optimisers
    tr = [] if run < 5 else None
    t0 = time.time()
    if algo == "HHO":
        best, f = core.hho(obj, rng, budget=BUDGET, trace=tr)
    elif algo == "ChHHO":
        best, f = core.hho(obj, rng, budget=BUDGET, chaos_init=True, chaos_rand=True, trace=tr)
    elif algo == "ChHHO-init":
        best, f = core.hho(obj, rng, budget=BUDGET, chaos_init=True, chaos_rand=False, trace=tr)
    elif algo == "PSO":
        best, f = core.pso(obj, rng, budget=BUDGET, trace=tr)
    else:
        best, f = core.gwo(obj, rng, budget=BUDGET, trace=tr)
    prem, th = obj.params(best[None])
    pred_s = core.predict(prem[0], th[0], d["X"])
    pred = core.to_price(d, pred_s)
    np.savez_compressed(fn, best=best, train_mse=f, pred=pred.astype(np.float64), fe=obj.fe,
                        secs=time.time() - t0, trace=np.array(tr if tr else []))
    return fn


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    if os.environ.get("STAGE") == "P2":
        tasks = list(itertools.product(["P2"], DATASETS, SCHEMES, ["HHO", "ChHHO", "PSO", "GWO"], range(20)))
    else:
        tasks = list(itertools.product(["P1"], DATASETS, SCHEMES, ALGOS, range(RUNS)))
        # leakage check: look-ahead scaling, HHO vs ChHHO only, 10 runs
        tasks += list(itertools.product(["P1L"], DATASETS, SCHEMES, ["HHO", "ChHHO"], range(10)))
    # P1 first; within it interleave by run so partial results cover every cell early
    tasks.sort(key=lambda t: (t[0], t[4], t[1], t[2], t[3]))
    t0 = time.time()
    with Pool(int(os.environ.get("WORKERS", 15))) as p:
        for i, fn in enumerate(p.imap_unordered(task, tasks, chunksize=1), 1):
            if i % 50 == 0 or i == len(tasks):
                print(f"{i}/{len(tasks)} done, {time.time() - t0:.0f}s", flush=True)
