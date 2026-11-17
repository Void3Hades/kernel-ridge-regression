"""
Kernel Ridge Regression from scratch (NumPy + SciPy + Matplotlib).

Run everything:      python krr_project.py
Run some:            python krr_project.py --exp 1 3 7
Quick smoke test:    python krr_project.py --quick
Figures are saved to ./figures/

Experiments
  1  1D toy fit: underfit / good fit / overfit
  2  (lambda, gamma) validation-error heatmap using closed-form LOO
  3  Kernel comparison (linear / poly / RBF) on a nonlinear problem
  4  Closed-form LOO vs explicit k-fold CV (accuracy + speed)
  5  Scaling of fit time with n (O(n^3))
  6  Conditioning and eigenvalue decay
  7  Iterative solvers: GD vs CG vs Nystrom-preconditioned CG
  8  Low-rank approximation: Nystrom and Random Fourier Features
  9  Real data (California Housing, falls back to synthetic) vs baselines
 10  Validation against scikit-learn
"""
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import argparse
import os
import time

import numpy as np
np.seterr(all="ignore")
import scipy.linalg as sla
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG_DIR = "figures"
RNG = np.random.default_rng(0)


# ----------------------------------------------------------------------------
# Kernels
# ----------------------------------------------------------------------------
def sqdist(A, B):
    d = (A**2).sum(1)[:, None] + (B**2).sum(1)[None, :] - 2 * A @ B.T
    return np.maximum(d, 0.0)


def rbf_kernel(A, B, gamma=1.0):
    return np.exp(-gamma * sqdist(A, B))


def linear_kernel(A, B):
    return A @ B.T


def poly_kernel(A, B, degree=3, c=1.0):
    return (A @ B.T + c) ** degree


def make_kernel(name, **p):
    if name == "rbf":
        return lambda A, B: rbf_kernel(A, B, p.get("gamma", 1.0))
    if name == "linear":
        return linear_kernel
    if name == "poly":
        return lambda A, B: poly_kernel(A, B, p.get("degree", 3), p.get("c", 1.0))
    raise ValueError(name)


# ----------------------------------------------------------------------------
# Exact KRR
# ----------------------------------------------------------------------------
class KRR:
    """alpha = (K + lam I)^-1 y ; f(x) = sum_i alpha_i k(x_i, x)."""

    def __init__(self, kernel="rbf", lam=1e-2, solver="cholesky", **kparams):
        self.kernel_fn = make_kernel(kernel, **kparams)
        self.lam, self.solver = lam, solver

    def fit(self, X, y):
        self.X = X
        self.y_mean = y.mean()  # center targets (no intercept in the kernel model)
        K = self.kernel_fn(X, X)
        A = K + self.lam * np.eye(len(X))
        yc = y - self.y_mean
        if self.solver == "cholesky":
            self.alpha = sla.cho_solve(sla.cho_factor(A), yc)
        elif self.solver == "cg":
            self.alpha, _ = conjugate_gradient(A, yc, tol=1e-10, maxiter=2000)
        else:
            raise ValueError(self.solver)
        return self

    def predict(self, Xn):
        return self.kernel_fn(Xn, self.X) @ self.alpha + self.y_mean


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


# ----------------------------------------------------------------------------
# Closed-form leave-one-out CV via eigendecomposition
#   H = K (K + lam I)^-1 = U diag(s/(s+lam)) U^T ;  LOO resid_i = (y_i - yhat_i)/(1 - H_ii)
# One eigh of K gives LOO error for ALL lambdas cheaply.
# ----------------------------------------------------------------------------
def loo_curve(K, y, lams):
    yc = y - y.mean()
    s, U = np.linalg.eigh(K)
    s = np.maximum(s, 0)
    Uty = U.T @ yc
    U2 = U**2
    out = []
    for lam in lams:
        f = s / (s + lam)
        yhat = U @ (f * Uty)
        Hii = U2 @ f
        r = (yc - yhat) / (1 - Hii)
        out.append(np.mean(r**2))
    return np.array(out)


def kfold_mse(X, y, k, **krr_params):
    idx = RNG.permutation(len(X))
    folds = np.array_split(idx, k)
    errs = []
    for i in range(k):
        te = folds[i]
        tr = np.concatenate([folds[j] for j in range(k) if j != i])
        m = KRR(**krr_params).fit(X[tr], y[tr])
        errs.append(np.mean((m.predict(X[te]) - y[te]) ** 2))
    return float(np.mean(errs))


# ----------------------------------------------------------------------------
# Iterative solvers for (K + lam I) alpha = y
# ----------------------------------------------------------------------------
def gradient_descent(A, b, iters=500, lr=None):
    """Gradient descent on 1/2 a^T A a - a^T b. Returns iterates' residual history."""
    if lr is None:
        lr = 1.0 / np.linalg.eigvalsh(A)[-1]  # 1/L
    a = np.zeros_like(b)
    hist = []
    for _ in range(iters):
        g = A @ a - b
        hist.append(np.linalg.norm(g) / np.linalg.norm(b))
        a = a - lr * g
    return a, hist


def conjugate_gradient(A, b, tol=1e-8, maxiter=500, Minv=None):
    """(Preconditioned) CG. Minv: function v -> P^{-1} v."""
    a = np.zeros_like(b)
    r = b.copy()
    z = Minv(r) if Minv else r
    p = z.copy()
    rz = r @ z
    bnorm = np.linalg.norm(b)
    hist = [np.linalg.norm(r) / bnorm]
    for _ in range(maxiter):
        Ap = A @ p
        step = rz / (p @ Ap)
        a += step * p
        r -= step * Ap
        hist.append(np.linalg.norm(r) / bnorm)
        if hist[-1] < tol:
            break
        z = Minv(r) if Minv else r
        rz_new = r @ z
        p = z + (rz_new / rz) * p
        rz = rz_new
    return a, hist


def nystrom_preconditioner(X, kernel_fn, lam, m, rng=RNG):
    """P = L L^T + lam I with K ~ L L^T; applied via Woodbury."""
    idx = rng.choice(len(X), size=m, replace=False)
    Z = X[idx]
    Kmm = kernel_fn(Z, Z) + 1e-10 * np.eye(m)
    Knm = kernel_fn(X, Z)
    C = np.linalg.cholesky(Kmm)
    L = sla.solve_triangular(C, Knm.T, lower=True).T  # n x m
    inner = sla.cho_factor(lam * np.eye(m) + L.T @ L)

    def Minv(v):
        return (v - L @ sla.cho_solve(inner, L.T @ v)) / lam

    return Minv


# ----------------------------------------------------------------------------
# Low-rank approximations
# ----------------------------------------------------------------------------
class NystromKRR:
    """Subset-of-regressors / Nystrom KRR with m landmarks.
    Solves (Knm^T Knm + lam Kmm) beta = Knm^T y ; f(x) = k(x, Z) beta."""

    def __init__(self, m=100, lam=1e-2, gamma=1.0, seed=0):
        self.m, self.lam, self.gamma, self.seed = m, lam, gamma, seed

    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)
        idx = rng.choice(len(X), size=min(self.m, len(X)), replace=False)
        self.Z = X[idx]
        self.y_mean = y.mean()
        Kmm = rbf_kernel(self.Z, self.Z, self.gamma)
        Knm = rbf_kernel(X, self.Z, self.gamma)
        A = Knm.T @ Knm + self.lam * Kmm + 1e-10 * np.eye(len(self.Z))
        self.beta = np.linalg.solve(A, Knm.T @ (y - self.y_mean))
        return self

    def predict(self, Xn):
        return rbf_kernel(Xn, self.Z, self.gamma) @ self.beta + self.y_mean


class RFFRidge:
    """Random Fourier Features (Rahimi-Recht) for the RBF kernel exp(-gamma||x-x'||^2)."""

    def __init__(self, D=200, lam=1e-2, gamma=1.0, seed=0):
        self.D, self.lam, self.gamma, self.seed = D, lam, gamma, seed

    def _phi(self, X):
        return np.sqrt(2.0 / self.D) * np.cos(X @ self.W + self.b)

    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)
        self.W = rng.normal(0, np.sqrt(2 * self.gamma), size=(X.shape[1], self.D))
        self.b = rng.uniform(0, 2 * np.pi, size=self.D)
        Phi = self._phi(X)
        self.y_mean = y.mean()
        A = Phi.T @ Phi + self.lam * np.eye(self.D)
        self.w = np.linalg.solve(A, Phi.T @ (y - self.y_mean))
        return self

    def predict(self, Xn):
        return self._phi(Xn) @ self.w + self.y_mean


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def toy_1d(n, noise=0.2, seed=0):
    r = np.random.default_rng(seed)
    X = r.uniform(-3, 3, size=(n, 1))
    y = np.sin(2 * X[:, 0]) + 0.3 * X[:, 0] + noise * r.normal(size=n)
    return X, y


def nonlinear_nd(n, d=5, noise=0.1, seed=0):
    r = np.random.default_rng(seed)
    X = r.normal(size=(n, d))
    y = np.sin(X[:, 0] * X[:, 1]) + X[:, 2] ** 2 - 0.5 * X[:, 3] + noise * r.normal(size=n)
    return X, y


def split(X, y, frac=0.7, seed=0):
    r = np.random.default_rng(seed)
    p = r.permutation(len(X))
    k = int(frac * len(X))
    return X[p[:k]], y[p[:k]], X[p[k:]], y[p[k:]]


def standardize(Xtr, Xte):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-12
    return (Xtr - mu) / sd, (Xte - mu) / sd


def save(name):
    os.makedirs(FIG_DIR, exist_ok=True)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, name)
    plt.savefig(path, dpi=140)
    plt.close()
    print(f"  saved {path}")


# ----------------------------------------------------------------------------
# Experiments
# ----------------------------------------------------------------------------
def exp1(quick):
    print("[1] 1D toy fit")
    X, y = toy_1d(60 if quick else 80)
    xs = np.linspace(-3, 3, 400)[:, None]
    settings = [("underfit (lam=10, gamma=0.1)", 10.0, 0.1),
                ("good fit (lam=0.05, gamma=1)", 0.05, 1.0),
                ("overfit (lam=1e-8, gamma=30)", 1e-8, 30.0)]
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8), sharey=True)
    for a, (t, lam, g) in zip(ax, settings):
        m = KRR("rbf", lam=lam, gamma=g).fit(X, y)
        a.scatter(X, y, s=14, c="k", alpha=0.6)
        a.plot(xs, m.predict(xs), "r", lw=2)
        a.set_ylim(-2.5, 2.5)
        a.set_title(t, fontsize=10)
    save("exp1_toy_fit.png")


def exp2(quick):
    print("[2] (lambda, gamma) LOO heatmap")
    X, y = nonlinear_nd(300 if quick else 600, d=5)
    lams = np.logspace(-6, 1, 30)
    gams = np.logspace(-3, 1, 25)
    H = np.zeros((len(gams), len(lams)))
    for i, g in enumerate(gams):
        H[i] = loo_curve(rbf_kernel(X, X, g), y, lams)
    i, j = np.unravel_index(H.argmin(), H.shape)
    print(f"  best LOO MSE {H.min():.4f} at gamma={gams[i]:.3g}, lambda={lams[j]:.3g}")
    plt.figure(figsize=(6.5, 5))
    plt.pcolormesh(np.log10(lams), np.log10(gams), np.log10(H), shading="auto")
    plt.colorbar(label="log10 LOO MSE")
    plt.plot(np.log10(lams[j]), np.log10(gams[i]), "r*", ms=14)
    plt.xlabel("log10 lambda")
    plt.ylabel("log10 gamma")
    plt.title("Closed-form LOO error landscape")
    save("exp2_heatmap.png")


def exp3(quick):
    print("[3] Kernel comparison")
    X, y = nonlinear_nd(800 if quick else 1500, d=5)
    Xtr, ytr, Xte, yte = split(X, y)
    lams = np.logspace(-6, 2, 25)
    configs = {"linear": dict(kernel="linear"),
               "poly (deg 3)": dict(kernel="poly", degree=3, c=1.0)}
    for g in [0.03, 0.1, 0.3]:
        configs[f"rbf (gamma={g})"] = dict(kernel="rbf", gamma=g)
    names, errs = [], []
    for name, cfg in configs.items():
        K = make_kernel(cfg["kernel"], **{k: v for k, v in cfg.items() if k != "kernel"})(Xtr, Xtr)
        lam = lams[np.argmin(loo_curve(K, ytr, lams))]  # tune by LOO on train only
        m = KRR(lam=lam, **cfg).fit(Xtr, ytr)
        e = rmse(m.predict(Xte), yte)
        print(f"  {name:18s} lam={lam:.2e}  test RMSE={e:.4f}")
        names.append(name)
        errs.append(e)
    plt.figure(figsize=(7, 4))
    plt.bar(names, errs, color="steelblue")
    plt.xticks(rotation=20)
    plt.ylabel("test RMSE")
    plt.title("Kernel comparison (lambda tuned by LOO)")
    save("exp3_kernels.png")


def exp4(quick):
    print("[4] LOO vs k-fold")
    X, y = nonlinear_nd(300 if quick else 500, d=5)
    lams = np.logspace(-5, 1, 12)
    g = 0.1
    t0 = time.perf_counter()
    loo = loo_curve(rbf_kernel(X, X, g), y, lams)
    t_loo = time.perf_counter() - t0
    t0 = time.perf_counter()
    kf = np.array([kfold_mse(X, y, 5, kernel="rbf", lam=l, gamma=g) for l in lams])
    t_kf = time.perf_counter() - t0
    print(f"  closed-form LOO: {t_loo:.3f}s   5-fold CV: {t_kf:.3f}s")
    plt.figure(figsize=(6, 4))
    plt.loglog(lams, loo, "o-", label=f"closed-form LOO ({t_loo:.2f}s)")
    plt.loglog(lams, kf, "s--", label=f"5-fold CV ({t_kf:.2f}s)")
    plt.xlabel("lambda")
    plt.ylabel("MSE")
    plt.legend()
    save("exp4_loo_vs_kfold.png")


def exp5(quick):
    print("[5] Scaling")
    ns = [250, 500, 1000, 1500, 2000] if quick else [250, 500, 1000, 2000, 3000, 4000]
    times = []
    for n in ns:
        X, y = nonlinear_nd(n, d=5)
        t0 = time.perf_counter()
        KRR("rbf", lam=1e-2, gamma=0.1).fit(X, y)
        times.append(time.perf_counter() - t0)
        print(f"  n={n:5d}  fit {times[-1]:.3f}s")
    slope = np.polyfit(np.log(ns), np.log(times), 1)[0]
    plt.figure(figsize=(6, 4))
    plt.loglog(ns, times, "o-", label=f"measured (slope ~ {slope:.2f})")
    plt.loglog(ns, times[0] * (np.array(ns) / ns[0]) ** 3, "k--", label="n^3 reference")
    plt.xlabel("n")
    plt.ylabel("fit time (s)")
    plt.legend()
    plt.xticks(ns, [str(n) for n in ns])
    plt.minorticks_off()
    save("exp5_scaling.png")


def exp6(quick):
    print("[6] Conditioning and eigenvalue decay")
    X, y = nonlinear_nd(400 if quick else 800, d=5)
    lams = np.logspace(-8, 2, 40)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for g in [0.01, 0.1, 1.0]:
        K = rbf_kernel(X, X, g)
        s = np.maximum(np.linalg.eigvalsh(K)[::-1], 1e-18)
        ax[0].semilogy(s, label=f"gamma={g}")
        ax[1].loglog(lams, (s[0] + lams) / (s[-1] + lams), label=f"gamma={g}")
    ax[0].set_title("Eigenvalue decay of K")
    ax[0].set_xlabel("index")
    ax[0].legend()
    ax[1].set_title("cond(K + lam I)")
    ax[1].set_xlabel("lambda")
    ax[1].legend()
    save("exp6_conditioning.png")


def exp7(quick):
    print("[7] GD vs CG vs preconditioned CG")
    n = 1000 if quick else 2000
    X, y = nonlinear_nd(n, d=5)
    g, lam = 0.1, 1e-3
    kf = lambda A, B: rbf_kernel(A, B, g)
    K = kf(X, X)
    A = K + lam * np.eye(n)
    b = y - y.mean()
    ev = np.linalg.eigvalsh(A)
    print(f"  cond(A) = {ev[-1] / ev[0]:.2e}")
    iters = 2000
    _, h_gd = gradient_descent(A, b, iters=iters)
    _, h_cg = conjugate_gradient(A, b, tol=1e-12, maxiter=iters)
    plt.figure(figsize=(7, 4.5))
    plt.semilogy(h_gd, label="gradient descent")
    plt.semilogy(h_cg, label="CG")
    for m in ([50, 200] if quick else [100, 400]):
        Minv = nystrom_preconditioner(X, kf, lam, m)
        _, h = conjugate_gradient(A, b, tol=1e-12, maxiter=iters, Minv=Minv)
        plt.semilogy(h, label=f"PCG (Nystrom m={m})")
        print(f"  PCG m={m}: {len(h) - 1} iterations to rel. residual {h[-1]:.1e}")
    plt.xlabel("iteration")
    plt.ylabel("relative residual")
    plt.title(f"Solver convergence (n={n}, lambda={lam})")
    plt.legend()
    save("exp7_solvers.png")

    # early stopping as implicit regularization.
    # GD on 1/2||K a - y||^2 started at 0 has the closed form (in the eigenbasis of K):
    #   a_t = U diag((1 - (1 - lr*s)^t) / s) U^T y   -> a spectral filter, like ridge's s/(s+lam)
    Xs, ys = nonlinear_nd(600, d=5, noise=0.5, seed=1)
    Xtr, ytr, Xte, yte = split(Xs, ys)
    g2 = 0.5
    Ktr, Kte = rbf_kernel(Xtr, Xtr, g2), rbf_kernel(Xte, Xtr, g2)
    bc = ytr - ytr.mean()
    s, U = np.linalg.eigh(Ktr)
    s = np.maximum(s, 1e-14)
    lr = 1.0 / s[-1]
    Uty = U.T @ bc
    ts = np.unique(np.logspace(0, 7, 120).astype(int))
    tr_e, te_e = [], []
    for t in ts:
        coef = U @ (((1 - (1 - lr * s) ** t) / s) * Uty)
        tr_e.append(rmse(Ktr @ coef + ytr.mean(), ytr))
        te_e.append(rmse(Kte @ coef + ytr.mean(), yte))
    best_t = ts[int(np.argmin(te_e))]
    plt.figure(figsize=(6, 4))
    plt.semilogx(ts, tr_e, label="train RMSE")
    plt.semilogx(ts, te_e, label="test RMSE")
    plt.axvline(best_t, color="gray", ls=":", label=f"best t = {best_t}")
    plt.xlabel("GD iteration t")
    plt.ylabel("RMSE")
    plt.title("Early stopping acts as regularization")
    plt.legend()
    save("exp7b_early_stopping.png")
    print(f"  best early-stopping iteration: {best_t}, test RMSE {min(te_e):.4f}, final {te_e[-1]:.4f}")


def exp8(quick):
    print("[8] Nystrom and RFF vs exact")
    n = 1500 if quick else 3000
    X, y = nonlinear_nd(n, d=5, noise=0.1)
    Xtr, ytr, Xte, yte = split(X, y)
    g, lam = 0.1, 1e-3
    t0 = time.perf_counter()
    ex = KRR("rbf", lam=lam, gamma=g).fit(Xtr, ytr)
    t_ex = time.perf_counter() - t0
    e_ex = rmse(ex.predict(Xte), yte)
    print(f"  exact: RMSE {e_ex:.4f}, {t_ex:.2f}s")
    ranks = [10, 25, 50, 100, 200, 400]
    res = {"Nystrom": ([], []), "RFF": ([], [])}
    for r in ranks:
        for name, mk in [("Nystrom", lambda: NystromKRR(r, lam, g)), ("RFF", lambda: RFFRidge(r, lam, g))]:
            t0 = time.perf_counter()
            m = mk().fit(Xtr, ytr)
            res[name][0].append(time.perf_counter() - t0)
            res[name][1].append(rmse(m.predict(Xte), yte))
        print(f"  rank {r:4d}: Nystrom {res['Nystrom'][1][-1]:.4f} ({res['Nystrom'][0][-1]:.3f}s) | "
              f"RFF {res['RFF'][1][-1]:.4f} ({res['RFF'][0][-1]:.3f}s)")
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for name, (t, e) in res.items():
        ax[0].loglog(ranks, e, "o-", label=name)
        ax[1].loglog(t, e, "o-", label=name)
    ax[0].axhline(e_ex, color="k", ls="--", label="exact KRR")
    ax[1].axvline(t_ex, color="k", ls="--", label="exact time")
    ax[0].set_xlabel("rank m / features D")
    ax[0].set_ylabel("test RMSE")
    ax[1].set_xlabel("fit time (s)")
    ax[1].set_ylabel("test RMSE")
    ax[0].legend()
    ax[1].legend()
    save("exp8_lowrank.png")


def exp9(quick):
    print("[9] Real data")
    try:
        from sklearn.datasets import fetch_california_housing
        d = fetch_california_housing()
        X, y = d.data, d.target
        name = "California Housing"
    except Exception as e:  # offline or sklearn missing
        print(f"  (could not load California Housing: {type(e).__name__}; using synthetic data)")
        X, y = nonlinear_nd(3000, d=8, noise=0.2)
        name = "synthetic"
    n = 2000 if quick else 4000
    p = np.random.default_rng(0).permutation(len(X))[:n]
    X, y = X[p], y[p]
    Xtr, ytr, Xte, yte = split(X, y)
    Xtr, Xte = standardize(Xtr, Xte)
    results = {}

    # Ridge (linear baseline)
    Xa, Xb = np.c_[Xtr, np.ones(len(Xtr))], np.c_[Xte, np.ones(len(Xte))]
    w = np.linalg.solve(Xa.T @ Xa + 1e-2 * np.eye(Xa.shape[1]), Xa.T @ ytr)
    results["Ridge"] = rmse(Xb @ w, yte)

    # KRR, hyperparameters by closed-form LOO
    best = (np.inf, None, None)
    lams = np.logspace(-4, 1, 12)
    for g in np.logspace(-3, 0, 7) / Xtr.shape[1]:
        l = loo_curve(rbf_kernel(Xtr, Xtr, g), ytr, lams)
        if l.min() < best[0]:
            best = (l.min(), g, lams[l.argmin()])
    _, g, lam = best
    print(f"  LOO-selected gamma={g:.3g}, lambda={lam:.3g}")
    results["KRR (RBF)"] = rmse(KRR("rbf", lam=lam, gamma=g).fit(Xtr, ytr).predict(Xte), yte)
    results["Nystrom m=300"] = rmse(NystromKRR(300, lam, g).fit(Xtr, ytr).predict(Xte), yte)

    try:
        from sklearn.ensemble import RandomForestRegressor
        rf = RandomForestRegressor(200, n_jobs=-1, random_state=0).fit(Xtr, ytr)
        results["Random Forest"] = rmse(rf.predict(Xte), yte)
    except Exception:
        pass
    print(f"  Dataset: {name}, n={n}")
    for k, v in results.items():
        print(f"  {k:16s} test RMSE = {v:.4f}")
    plt.figure(figsize=(6.5, 4))
    bars = plt.bar(results.keys(), results.values(), color="seagreen")
    plt.bar_label(bars, fmt="%.3f")
    plt.xticks(rotation=20)
    plt.ylabel("test RMSE")
    plt.title(name)
    save("exp9_real_data.png")


def exp10(quick):
    print("[10] sklearn validation")
    try:
        from sklearn.kernel_ridge import KernelRidge
    except Exception:
        print("  scikit-learn not installed; skipping.")
        return
    X, y = nonlinear_nd(400, d=5)
    Xtr, ytr, Xte, _ = split(X, y)
    lam, g = 1e-2, 0.2
    # sklearn's KernelRidge has no intercept; compare on centered targets (as our class does)
    mine = KRR("rbf", lam=lam, gamma=g).fit(Xtr, ytr).predict(Xte)
    ref = KernelRidge(alpha=lam, kernel="rbf", gamma=g).fit(Xtr, ytr - ytr.mean()).predict(Xte) + ytr.mean()
    cg = KRR("rbf", lam=lam, gamma=g, solver="cg").fit(Xtr, ytr).predict(Xte)
    print(f"  max |mine - sklearn|     = {np.abs(mine - ref).max():.2e}")
    print(f"  max |CG solver - chol|   = {np.abs(cg - mine).max():.2e}")
    assert np.allclose(mine, ref, atol=1e-6), "Mismatch with sklearn!"
    print("  OK: matches scikit-learn")


EXPS = {1: exp1, 2: exp2, 3: exp3, 4: exp4, 5: exp5, 6: exp6, 7: exp7, 8: exp8, 9: exp9, 10: exp10}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", nargs="*", type=int, default=list(EXPS))
    ap.add_argument("--quick", action="store_true", help="smaller sizes for a fast run")
    args = ap.parse_args()
    for k in args.exp:
        t0 = time.perf_counter()
        EXPS[k](args.quick)
        print(f"  done in {time.perf_counter() - t0:.1f}s\n")
