import ctypes as _ct
try:  # run at idle priority so the laptop stays responsive (inherited by worker processes)
    _ct.windll.kernel32.SetPriorityClass(_ct.windll.kernel32.GetCurrentProcess(), 0x4000)
except Exception:
    pass
"""Non-metaheuristic baselines: naive random walk, OLS, classical hybrid-learning ANFIS, LSTM."""
import os, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import core

DATASETS = ["GSPC", "IXIC", "DJI", "NSEI", "BVSP", "N225"]
PROTS = {"P1": "train", "P1L": "full", "P2": "ret"}
OUT = os.path.join(HERE, "results", "base")
LSTM_SEEDS = 30
WIN = 5
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_num_threads(1)


class LSTMNet(torch.nn.Module):
    def __init__(self, nin=3, h=32):
        super().__init__()
        self.lstm = torch.nn.LSTM(nin, h, batch_first=True)
        self.fc = torch.nn.Linear(h, 1)

    def forward(self, x):
        o, _ = self.lstm(x)
        return self.fc(o[:, -1]).squeeze(-1)


def windows(X):
    W = np.stack([X[i - WIN + 1:i + 1] for i in range(WIN - 1, len(X))])
    return W  # aligned with targets WIN-1 .. end


def lstm_run(d, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    W = windows(d["X"]); ys = d["y"][WIN - 1:]
    n_tr = d["n_tr"] - (WIN - 1)
    n_fit = int(0.85 * n_tr)                       # last 15 % of training block = validation
    Wt = torch.tensor(W, dtype=torch.float32, device=dev)
    yt = torch.tensor(ys, dtype=torch.float32, device=dev)
    net = LSTMNet().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    best, best_state, bad = 1e9, None, 0
    g = torch.Generator(device="cpu").manual_seed(seed)
    for ep in range(300):
        net.train()
        perm = torch.randperm(n_fit, generator=g)
        for b in range(0, n_fit, 64):
            idx = perm[b:b + 64].to(dev)
            loss = ((net(Wt[idx]) - yt[idx]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            v = ((net(Wt[n_fit:n_tr]) - yt[n_fit:n_tr]) ** 2).mean().item()
        if v < best - 1e-9:
            best, bad = v, 0
            best_state = {k: t.clone() for k, t in net.state_dict().items()}
        else:
            bad += 1
            if bad >= 25:
                break
    net.load_state_dict(best_state)
    with torch.no_grad():
        p = net(Wt).cpu().numpy().astype(np.float64)
    full = np.full(len(d["y"]), np.nan); full[WIN - 1:] = p
    return full


def main():
    os.makedirs(OUT, exist_ok=True)
    for prot, sc in PROTS.items():
        for ds in DATASETS:
            if os.path.exists(os.path.join(OUT, f"{prot}_{ds}.npz")):
                continue
            d = core.load_dataset(os.path.join(HERE, "data", ds + ".csv"), sc)
            n = d["n_tr"]; inv = lambda p: core.to_price(d, p)
            res = {"naive": d["c_prev"].copy()}
            Xa = np.c_[d["X"], np.ones(len(d["X"]))]
            beta = np.linalg.lstsq(Xa[:n], d["y"][:n], rcond=None)[0]
            res["OLS"] = inv(Xa @ beta)
            t = time.time()
            prem, th = core.classical_anfis(d["X"][:n], d["y"][:n])
            res["ANFIS"] = inv(core.predict(prem, th, d["X"]))
            res["ANFIS_train_mse"] = np.array(((core.predict(prem, th, d["X"][:n]) - d["y"][:n]) ** 2).mean())
            L = np.stack([inv(lstm_run(d, 2000 + s)) for s in range(LSTM_SEEDS)])
            res["LSTM"] = L
            np.savez_compressed(os.path.join(OUT, f"{prot}_{ds}.npz"), **res)
            print(prot, ds, "done", round(time.time() - t, 1), flush=True)


if __name__ == "__main__":
    main()
