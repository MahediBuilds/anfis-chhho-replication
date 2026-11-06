"""ANFIS + metaheuristic training (HHO, ChHHO, PSO, GWO) -- replication core.

Design (documented in the manuscript):
  * Inputs  : Open_t, High_t, Low_t   ->  target Close_{t+1}  (base-paper protocol)
  * Split   : chronological 70 % train / 30 % test; min-max scaling fitted on train only
  * ANFIS   : first-order Sugeno, 3 inputs x 2 Gaussian MFs (grid) -> 8 rules,
              12 premise + 32 consequent parameters
  * Scheme A ("hybrid MH+LSE"): metaheuristic searches the 12 premise parameters,
              consequents solved by least squares for every candidate (Jang's hybrid idea)
  * Scheme B ("all-parameter MH"): metaheuristic searches all 44 parameters in
              [-25, 25] as in Table 2 of the base paper
  * Budget  : 50 000 objective evaluations for every optimiser (pop 50 x 1000 it.)
"""
import math
import numpy as np

N_IN, N_MF = 3, 2
N_RULE = N_MF ** N_IN
N_PREM = N_IN * N_MF * 2          # centre + sigma per MF
N_CONS = N_RULE * (N_IN + 1)
SIG_MIN = 1e-3
RULE_IDX = np.array(np.meshgrid(*[range(N_MF)] * N_IN, indexing="ij")).reshape(N_IN, -1).T  # (8,3)


# ----------------------------------------------------------------------------- data
def load_dataset(path, scaling="train"):
    """scaling: "train" (leak-free min-max on levels), "full" (look-ahead min-max on levels),
    "ret" (scale-free: H/O-1, L/O-1, O/O_prev-1 -> C_{t+1}/O_t - 1, min-max fitted on train)."""
    raw = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    o, h, l, c = (raw[k].astype(float) for k in ("open", "high", "low", "close"))
    n_tr_lvl = int(round(0.7 * (len(c) - 1)))
    if scaling == "ret":
        t = np.arange(1, len(c) - 1)                     # needs O_{t-1}
        X = np.stack([h[t] / o[t] - 1, l[t] / o[t] - 1, o[t] / o[t - 1] - 1], 1)
        y = c[t + 1] / o[t] - 1
        y_raw, c_prev, o_t = c[t + 1], c[t], o[t]
        n_tr = n_tr_lvl - 1                              # identical test days to the level protocols
        dates = raw["date"][t + 1]
    else:
        X = np.stack([o, h, l], 1)[:-1]
        y = c[1:]
        y_raw, c_prev, o_t = c[1:], c[:-1], o[:-1]
        n_tr = n_tr_lvl
        dates = raw["date"][1:]
    ref = slice(None) if scaling == "full" else slice(0, n_tr)   # "full" = look-ahead scaling
    xmin, xmax = X[ref].min(0), X[ref].max(0)
    ymin, ymax = y[ref].min(), y[ref].max()
    Xs = (X - xmin) / (xmax - xmin)
    ys = (y - ymin) / (ymax - ymin)
    return dict(X=Xs, y=ys, y_raw=y_raw, c_prev=c_prev, o_t=o_t, n_tr=n_tr, ymin=ymin, ymax=ymax,
                dates=dates, ret=(scaling == "ret"))


def to_price(d, pred_s):
    p = pred_s * (d["ymax"] - d["ymin"]) + d["ymin"]
    return d["o_t"] * (1 + p) if d["ret"] else p


# ----------------------------------------------------------------------------- ANFIS
SEL = np.zeros((N_IN * N_MF, N_RULE))
for _r in range(N_RULE):
    for _i in range(N_IN):
        SEL[_i * N_MF + RULE_IDX[_r, _i], _r] = 1.0


def firing(prem, X):
    """prem (k, 12) -> normalised firing strengths (k, n, 8) (log-space product, softmax-normalised)."""
    k = prem.shape[0]
    c = prem[:, : N_IN * N_MF].reshape(k, N_IN, N_MF)
    s = np.abs(prem[:, N_IN * N_MF:].reshape(k, N_IN, N_MF)) + SIG_MIN
    d = (X[None, :, :, None] - c[:, None, :, :]) / s[:, None]      # (k,n,3,2)
    logw = (-0.5 * d * d).reshape(k, X.shape[0], -1) @ SEL          # (k,n,8)
    logw -= logw.max(-1, keepdims=True)
    w = np.exp(logw)
    return w / w.sum(-1, keepdims=True)


def design(wn, X):
    Xa = np.concatenate([X, np.ones((X.shape[0], 1))], 1)            # (n,4)
    return (wn[..., None] * Xa[None, :, None, :]).reshape(wn.shape[0], X.shape[0], -1)


def lse(Phi, y, lam=1e-6):
    PhiT = Phi.transpose(0, 2, 1)
    A = PhiT @ Phi + lam * np.eye(Phi.shape[2])[None]
    b = PhiT @ y
    return np.linalg.solve(A, b[..., None])[..., 0]


class Objective:
    """Training-MSE objective for a population of candidate vectors."""

    def __init__(self, X, y, scheme):
        self.X, self.y, self.scheme = X, y, scheme
        self.fe = 0
        if scheme == "A":
            self.dim = N_PREM
            lo = np.r_[np.full(N_IN * N_MF, -0.25), np.full(N_IN * N_MF, 0.05)]
            hi = np.r_[np.full(N_IN * N_MF, 1.25), np.full(N_IN * N_MF, 1.50)]
        else:
            self.dim = N_PREM + N_CONS
            lo, hi = np.full(self.dim, -25.0), np.full(self.dim, 25.0)
        self.lo, self.hi = lo, hi

    def params(self, P):
        """Return (premise, consequent) for population P."""
        P = np.atleast_2d(P)
        if self.scheme == "A":
            wn = firing(P, self.X)
            Phi = design(wn, self.X)
            th = lse(Phi, self.y)
            return P, th
        return P[:, :N_PREM], P[:, N_PREM:]

    def __call__(self, P):
        P = np.atleast_2d(P)
        self.fe += P.shape[0]
        if self.scheme == "A":
            Phi = design(firing(P, self.X), self.X)
            th = lse(Phi, self.y)
        else:
            th = P[:, N_PREM:]
            Phi = design(firing(P[:, :N_PREM], self.X), self.X)
        pred = (Phi @ th[..., None])[..., 0]
        f = ((pred - self.y[None]) ** 2).mean(1)
        return np.where(np.isfinite(f), f, 1e10)


def predict(prem, th, X):
    wn = firing(prem[None], X)
    return design(wn, X)[0] @ th


# ----------------------------------------------------------------------------- chaos
A_QUAD = 1.4
LO_Q, HI_Q = A_QUAD - A_QUAD ** 2, A_QUAD            # invariant interval [-0.56, 1.4]


class QuadraticChaos:
    """x_{i+1} = a - x_i^2 (a = 1.4, x0 = 0.6), rescaled to [0,1]."""

    def __init__(self, x0=0.6):
        self.x = x0

    def __call__(self, size=None):
        n = 1 if size is None else int(np.prod(size))
        out = np.empty(n)
        x = self.x
        for i in range(n):
            x = A_QUAD - x * x
            out[i] = (x - LO_Q) / (HI_Q - LO_Q)
        self.x = x
        return out[0] if size is None else out.reshape(size)


def lyapunov_quadratic(a, x0=0.6, n=200000, burn=1000):
    x, acc = x0, 0.0
    for i in range(n + burn):
        if i >= burn:
            acc += math.log(abs(-2 * x) + 1e-300)
        x = a - x * x
    return acc / n


# ----------------------------------------------------------------------------- optimisers
def levy(rng, d, beta=1.5):
    sig = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2) /
           (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = rng.normal(size=d) * sig
    v = rng.normal(size=d)
    return 0.01 * u / np.abs(v) ** (1 / beta)


def hho(obj, rng, pop=50, budget=50000, chaos_init=False, chaos_rand=False, trace=None):
    lo, hi, D = obj.lo, obj.hi, obj.dim
    ch = QuadraticChaos(0.6)
    U = (lambda *s: ch(s if s else None)) if chaos_rand else (lambda *s: rng.random(s if s else None))
    Z = lo + (hi - lo) * (ch((pop, D)) if chaos_init else rng.random((pop, D)))
    F = obj(Z)
    best_i = F.argmin(); rab, rabf = Z[best_i].copy(), F[best_i]
    while obj.fe < budget:
        frac = min(obj.fe / budget, 1.0)
        zm = Z.mean(0)
        newZ = Z.copy()
        dive = []
        for i in range(pop):
            E0 = 2 * U() - 1
            E = 2 * E0 * (1 - frac)
            if abs(E) >= 1:
                q = U()
                if q >= 0.5:
                    zr = Z[rng.integers(pop)]
                    newZ[i] = zr - U() * np.abs(zr - 2 * U() * Z[i])
                else:
                    newZ[i] = (rab - zm) - U() * (lo + U() * (hi - lo))
            else:
                r = U(); J = 2 * (1 - U())
                if r >= 0.5 and abs(E) >= 0.5:
                    newZ[i] = (rab - Z[i]) - E * np.abs(J * rab - Z[i])
                elif r >= 0.5 and abs(E) < 0.5:
                    newZ[i] = rab - E * np.abs(rab - Z[i])
                else:
                    base = Z[i] if abs(E) >= 0.5 else zm
                    Y = rab - E * np.abs(J * rab - base)
                    Zc = Y + rng.random(D) * levy(rng, D)
                    dive.append((i, Y, Zc))
        plain = [i for i in range(pop) if i not in {d[0] for d in dive}]
        if plain:
            newZ[plain] = np.clip(newZ[plain], lo, hi)
            F[plain] = obj(newZ[plain])
            Z[plain] = newZ[plain]
        if dive:
            idx = np.array([d[0] for d in dive])
            Ys = np.clip(np.array([d[1] for d in dive]), lo, hi)
            Zs = np.clip(np.array([d[2] for d in dive]), lo, hi)
            fY, fZ = obj(Ys), obj(Zs)
            for j, i in enumerate(idx):
                if fY[j] < F[i]:
                    Z[i], F[i] = Ys[j], fY[j]
                elif fZ[j] < F[i]:
                    Z[i], F[i] = Zs[j], fZ[j]
        b = F.argmin()
        if F[b] < rabf:
            rab, rabf = Z[b].copy(), F[b]
        if trace is not None:
            trace.append((obj.fe, rabf))
    return rab, rabf


def pso(obj, rng, pop=50, budget=50000, w=1.0, wdamp=0.99, c1=1.0, c2=2.0, trace=None):
    lo, hi, D = obj.lo, obj.hi, obj.dim
    vmax = 0.1 * (hi - lo)
    X = lo + (hi - lo) * rng.random((pop, D)); V = np.zeros((pop, D))
    F = obj(X); PB, PBf = X.copy(), F.copy()
    g = PBf.argmin(); G, Gf = PB[g].copy(), PBf[g]
    while obj.fe < budget:
        V = w * V + c1 * rng.random((pop, D)) * (PB - X) + c2 * rng.random((pop, D)) * (G - X)
        V = np.clip(V, -vmax, vmax)
        X = np.clip(X + V, lo, hi)
        F = obj(X)
        imp = F < PBf; PB[imp], PBf[imp] = X[imp], F[imp]
        g = PBf.argmin()
        if PBf[g] < Gf: G, Gf = PB[g].copy(), PBf[g]
        w *= wdamp
        if trace is not None: trace.append((obj.fe, Gf))
    return G, Gf


def gwo(obj, rng, pop=50, budget=50000, trace=None):
    lo, hi, D = obj.lo, obj.hi, obj.dim
    X = lo + (hi - lo) * rng.random((pop, D)); F = obj(X)
    order = F.argsort(); L = [(X[j].copy(), F[j]) for j in order[:3]]
    T = budget // pop
    t = 0
    while obj.fe < budget:
        a = 2 - 2 * t / T
        Xn = np.zeros_like(X)
        for (Lp, _) in L:
            A = 2 * a * rng.random((pop, D)) - a
            C = 2 * rng.random((pop, D))
            Xn += Lp - A * np.abs(C * Lp - X)
        X = np.clip(Xn / 3, lo, hi); F = obj(X)
        cand = [(X[j], F[j]) for j in range(pop)] + L
        cand.sort(key=lambda z: z[1]); L = [(c[0].copy(), c[1]) for c in cand[:3]]
        t += 1
        if trace is not None: trace.append((obj.fe, L[0][1]))
    return L[0]


# ----------------------------------------------------------------------------- classical ANFIS
def classical_anfis(X, y, epochs=300, lr=0.01):
    """Jang's hybrid learning: LSE for consequents + gradient descent (Adam) on premises."""
    import torch
    torch.manual_seed(0)
    s0 = 1 / (2 * math.sqrt(2 * math.log(2)))
    c = torch.tensor(np.tile([0.0, 1.0], (N_IN, 1)), dtype=torch.float64, requires_grad=True)
    s = torch.full((N_IN, N_MF), s0, dtype=torch.float64, requires_grad=True)
    Xt, yt = torch.tensor(X), torch.tensor(y)
    Xa = torch.cat([Xt, torch.ones(len(Xt), 1, dtype=torch.float64)], 1)
    ridx = torch.tensor(RULE_IDX)
    opt = torch.optim.Adam([c, s], lr=lr)

    def phi():
        mu = torch.exp(-0.5 * ((Xt[:, :, None] - c[None]) / (s.abs()[None] + SIG_MIN)) ** 2)
        w = torch.ones(len(Xt), N_RULE, dtype=torch.float64)
        for i in range(N_IN):
            w = w * mu[:, i, ridx[:, i]]
        wn = w / (w.sum(1, keepdim=True) + 1e-12)
        return (wn[:, :, None] * Xa[:, None, :]).reshape(len(Xt), -1)

    for ep in range(epochs + 1):
        P = phi()
        with torch.no_grad():
            A = P.T @ P + 1e-6 * torch.eye(P.shape[1], dtype=torch.float64)
            th = torch.linalg.solve(A, P.T @ yt)
        if ep == epochs:
            break
        loss = ((P @ th - yt) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    prem = np.r_[c.detach().numpy().ravel(), s.detach().abs().numpy().ravel()]
    return prem, th.numpy()


# ----------------------------------------------------------------------------- metrics
def metrics(y, yhat, c_prev):
    e = y - yhat
    rmse = float(np.sqrt((e ** 2).mean()))
    naive = float(np.sqrt(((y - c_prev) ** 2).mean()))
    return dict(
        RMSE=rmse, MAE=float(np.abs(e).mean()), MAPE=float(100 * np.abs(e / y).mean()),
        TheilU1=float(rmse / (np.sqrt((y ** 2).mean()) + np.sqrt((yhat ** 2).mean()))),
        U2=rmse / naive,
        DA=float(100 * (np.sign(yhat - c_prev) == np.sign(y - c_prev)).mean()),
    )
