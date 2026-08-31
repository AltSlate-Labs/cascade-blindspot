"""
Phase-0 synthetic mechanism study for the blind-spot conservation conjecture.

This is NOT a real-LLM experiment. It is a controlled synthetic model whose
purpose is to test whether the conservation law EMERGES from a plausible
mechanism rather than being assumed.

Task: K-way classification standing in for "produce the right answer". Ground
truth y*(phi) is a fixed random 2-layer network. The STUDENT is a logistic model
on random ReLU features, trained on a pool that the loop grows; it improves with
data. The VERIFIER is FIXED and has two regimes: on a blind region (a fixed random
half-space set to cover input-space mass rho) it rubber-stamps whatever the student
says; elsewhere it re-derives its own top class with a strong classifier and
accepts only on agreement. So rho sets the verifier's blind-spot rate beta (this
models blind MASS, not correlation per se; shared student-verifier failure modes
are one interpretation of what makes beta large, but the blind region here is
independent of where the student errs -- the "lazy rubber-stamper" case). H2 tests
floor-against-beta.

The key point is that NOTHING tells the loop to spare blind-spot errors. Because
blind-region items are ACCEPTED, they never enter the training pool, so the
student receives no new data there -- but whether that yields a conserved error
floor, self-healing, or (under self-training) self-reinforcement is emergent from
the learning dynamics, not imposed.

Outputs: a results table (stdout) + four figures under ../figures/.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings("ignore")

RNG = np.random.default_rng(7)
FIGDIR = Path(__file__).resolve().parent.parent / "figures"
FIGDIR.mkdir(exist_ok=True)

# ----------------------------------------------------------------------------
# Ground-truth K-way labeller y*(phi): fixed random 2-layer net
# ----------------------------------------------------------------------------
D, K, HID = 20, 10, 64
W1 = RNG.normal(size=(HID, D)) / np.sqrt(D)
b1 = RNG.normal(size=HID) * 0.3
W2 = RNG.normal(size=(K, HID)) / np.sqrt(HID)

def y_star(phi):
    h = np.tanh(phi @ W1.T + b1)
    return (h @ W2.T).argmax(1)

# ----------------------------------------------------------------------------
# Random ReLU feature maps
# ----------------------------------------------------------------------------
def make_basis(m, d=D):
    return (RNG.normal(size=(m, d)) / np.sqrt(d), RNG.normal(size=m) * 0.5)

def feats(basis, phi):
    A, c = basis
    return np.maximum(0.0, phi @ A.T + c)

M_S = 300                      # student features (adequate capacity, data-starved)
S_BASIS = make_basis(M_S)
V_BASIS = make_basis(450)      # strong verifier basis (fixed)
RDIR = RNG.normal(size=D)      # fixed direction defining the verifier's blind half-space
_projbig = sample = None
_PROJ = (RNG.normal(size=(30000, D)) @ RDIR)   # for blind-region quantile thresholds

def fit_lr(X, y, C=3.0):
    return LogisticRegression(C=C, max_iter=200).fit(X, y)

# ----------------------------------------------------------------------------
# One loop run.  rng seeds ALL data sampling so variants at a given seed share
# the same random world (paired comparison).
# ----------------------------------------------------------------------------
def run(variant, rho, rounds=8, n0=140, n_batch=600, n_eval=3000, n_vfit=5000, seed=0):
    rng = np.random.default_rng(1000 + seed)
    samp = lambda n: rng.normal(size=(n, D))

    # fixed strong verifier (rho-independent); blind-region threshold from rho.
    # The blind region is {x : RDIR . x < thr}, a fixed random half-space whose
    # threshold is set so it covers input-space mass rho (measured on _PROJ).
    Xv = samp(n_vfit); Vstrong = fit_lr(feats(V_BASIS, Xv), y_star(Xv), C=6.0)
    thr = np.quantile(_PROJ, rho)
    def v_judge(phi, s_pred):
        """returns (accept_mask, v_effective_class). Accept if in the blind
        region (rubber-stamp) OR the strong classifier independently agrees;
        the latter is why measured beta0 slightly exceeds rho."""
        blind = (phi @ RDIR) < thr
        vp = Vstrong.predict(feats(V_BASIS, phi))
        accept = blind | (vp == s_pred)
        veff = np.where(blind, s_pred, vp)
        return accept, veff

    Xe = samp(n_eval); ye = y_star(Xe)
    Xtr = samp(n0); ytr = y_star(Xtr)

    hist = []; stu = None
    for t in range(rounds + 1):
        if not (variant == "frozen" and stu is not None):
            stu = fit_lr(feats(S_BASIS, Xtr), ytr)
        se = stu.predict(feats(S_BASIS, Xe))
        wrong = se != ye
        if variant in ("oracle", "matchoracle"):
            v_acc = ~wrong; veff = ye      # perfect verifier (beta0 = 0)
        else:
            v_acc, veff = v_judge(Xe, se)
        q = wrong.mean(); p_rej = (~v_acc).mean()
        beta = (wrong & v_acc).sum() / max(wrong.sum(), 1)
        delivered = np.where(v_acc, se, ye)
        eps = (delivered != ye).mean()
        dash = (veff != delivered).mean()
        hist.append(dict(t=t, q=q, eps=eps, beta=beta, p_reject=p_rej, dash=dash,
                         blind=(wrong & v_acc).mean(), detect=(wrong & ~v_acc).mean()))
        if t == rounds:
            break
        Xb = samp(n_batch); yb = y_star(Xb); sb = stu.predict(feats(S_BASIS, Xb))
        wb = sb != yb
        if variant == "oracle":
            aX, aY = Xb, yb                 # perfect verifier, train on ALL items
        elif variant == "matchoracle":
            aX, aY = Xb[wb], yb[wb]         # perfect verifier, train on WRONG only
        else:
            acc, _ = v_judge(Xb, sb); rej = ~acc
            if variant == "selftrain":
                aX = np.vstack([Xb[rej], Xb[acc]]); aY = np.concatenate([yb[rej], sb[acc]])
            else:                              # corrective / frozen
                aX, aY = Xb[rej], yb[rej]
        Xtr = np.vstack([Xtr, aX]); ytr = np.concatenate([ytr, aY])
    return hist

N_SEEDS = 20
def avg_runs(variant, rho, seeds=range(N_SEEDS), **kw):
    """Per-round mean and sd (across seeds) for every metric."""
    runs = [run(variant, rho, seed=s, **kw) for s in seeds]
    out = []
    for t in range(len(runs[0])):
        d = {"t": t}
        for k in runs[0][t]:
            if k == "t":
                continue
            v = np.array([r[t][k] for r in runs])
            d[k] = float(v.mean()); d[k + "_sd"] = float(v.std(ddof=1))
        out.append(d)
    return out

def floor_stats(variant, rho, seeds=range(N_SEEDS), **kw):
    """Per-seed last-3-round-mean floor -> (mean, sd); also mean/sd of beta0."""
    runs = [run(variant, rho, seed=s, **kw) for s in seeds]
    floors = np.array([np.mean([r[t]["eps"] for t in (-3, -2, -1)]) for r in runs])
    b0 = np.array([r[0]["beta"] for r in runs])
    return floors.mean(), floors.std(ddof=1), b0.mean(), b0.std(ddof=1)

# ----------------------------------------------------------------------------
# Experiments
# ----------------------------------------------------------------------------
RHO_MAIN = 0.7
print("=" * 74)
print(f"PHASE-0 SYNTHETIC MECHANISM STUDY  (blind-spot conservation, {N_SEEDS} seeds)")
print("=" * 74)
variants = ["corrective", "selftrain", "oracle", "matchoracle", "frozen"]
res = {v: avg_runs(v, RHO_MAIN) for v in variants}

print(f"\nMain comparison  rho={RHO_MAIN}, K={K} classes, {N_SEEDS} seeds, 8 rounds "
      f"(mean +/- sd)")
print(f"{'variant':<12}{'q0':>6}{'beta0':>7}{'q_T':>15}{'eps_T':>16}{'q0*b0':>8}{'dash_T':>8}")
for v in variants:
    h = res[v]; q0, b0 = h[0]["q"], h[0]["beta"]
    qT, qTsd = h[-1]["q"], h[-1]["q_sd"]; eT, eTsd = h[-1]["eps"], h[-1]["eps_sd"]
    print(f"{v:<12}{q0:>6.3f}{b0:>7.3f}{qT:>9.3f}+/-{qTsd:<4.3f}"
          f"{eT:>8.3f}+/-{eTsd:<4.3f}{q0*b0:>8.3f}{h[-1]['dash']:>8.3f}")

print("\nSelf-training raw error q_t per round (mean) -- H3 check (does it RISE?):")
print("  " + "  ".join(f"{d['q']:.3f}" for d in res["selftrain"]))
print("Corrective raw error q_t per round (mean):")
print("  " + "  ".join(f"{d['q']:.3f}" for d in res["corrective"]))

print("\nVerifier-induced excess in raw error (blind spot's cost to the student):")
cf = floor_stats("corrective", RHO_MAIN)
print(f"  corrective  q_T = {res['corrective'][-1]['q']:.3f}")
print(f"  matched-oracle (perfect V, train wrong-only) q_T = {res['matchoracle'][-1]['q']:.3f}")
print(f"  excess = {res['corrective'][-1]['q'] - res['matchoracle'][-1]['q']:+.3f}")

print("\nH2 -- user-facing floor vs blind-spot rate rho (corrective, mean +/- sd):")
rhos = [0.0, 0.2, 0.4, 0.6, 0.8, 0.95]
floor_rho, floor_sd, b0_rho = [], [], []
for rho in rhos:
    fm, fsd, b0m, _ = floor_stats("corrective", rho)
    floor_rho.append(fm); floor_sd.append(fsd); b0_rho.append(b0m)
    print(f"  rho={rho:.2f}  beta0={b0m:.3f}  floor(eps)={fm:.3f} +/- {fsd:.3f}")

# ----------------------------------------------------------------------------
# Figures  (mean lines with +/-1 sd shaded bands)
# ----------------------------------------------------------------------------
plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 130})
RED, GRN, BLU, GRY, PUR, ORG = "#c0392b", "#2e7d32", "#2962ff", "#777777", "#8e44ad", "#e67e22"
def ts(h, k): return np.array([d["t"] for d in h]), np.array([d[k] for d in h])
def band(ax, h, k, color, lw=2, ls="-", label=None):
    x, m = ts(h, k); _, sd = ts(h, k + "_sd")
    ax.plot(x, m, color=color, lw=lw, ls=ls, label=label)
    ax.fill_between(x, m - sd, m + sd, color=color, alpha=0.15, lw=0)

# A: scissors
h = res["corrective"]; fig, ax = plt.subplots(figsize=(5.2, 3.4))
band(ax, h, "q", GRY, lw=1.6, ls=":", label=r"raw error $q_t$")
band(ax, h, "dash", GRN, label="verifier-estimated (dashboard)")
band(ax, h, "eps", RED, label=r"gold user-facing $\varepsilon_t$")
x, gold = ts(h, "eps"); _, dash = ts(h, "dash")
ax.fill_between(x, dash, gold, color=GRY, alpha=0.12)
ax.axhline(h[0]["q"]*h[0]["beta"], color=GRY, ls="--", lw=1)
ax.annotate(r"$q_0\beta_0$", (x[-1]-0.5, h[0]["q"]*h[0]["beta"]+0.004), fontsize=9)
ax.set_xlabel("round $t$"); ax.set_ylabel("error rate"); ax.set_ylim(bottom=0)
ax.legend(frameon=False, fontsize=8.5); ax.set_title("Dashboard blindness (synthetic)", fontsize=10)
fig.tight_layout(); fig.savefig(FIGDIR/"exp_scissors.pdf"); fig.savefig(FIGDIR/"exp_scissors.png")

# B: conservation (stacked) + matched-oracle capacity floor
fig, ax = plt.subplots(figsize=(5.2, 3.4))
x, det = ts(h, "detect"); _, bl = ts(h, "blind")
ax.fill_between(x, 0, bl, color=RED, alpha=0.30, label=r"blind $q_t\beta_t$ (V-accepted)")
ax.fill_between(x, bl, bl+det, color=BLU, alpha=0.25, label=r"detectable $q_t r_t$ (V-rejected)")
_, qo = ts(res["matchoracle"], "q")
ax.plot(x, qo, color="k", ls="--", lw=1.4, label=r"$\beta_0{=}0$ control $q_t$ (capacity floor)")
ax.set_xlabel("round $t$"); ax.set_ylabel(r"raw error $q_t$"); ax.set_ylim(bottom=0)
ax.legend(frameon=False, fontsize=8.5); ax.set_title("Blind-spot conservation (synthetic)", fontsize=10)
fig.tight_layout(); fig.savefig(FIGDIR/"exp_conservation.pdf"); fig.savefig(FIGDIR/"exp_conservation.png")

# C: floor vs beta0 (H2), with error bars + oracle origin point
fig, ax = plt.subplots(figsize=(5.2, 3.4))
ax.errorbar(b0_rho, floor_rho, yerr=floor_sd, fmt="o-", color=RED, lw=1.8, capsize=3, zorder=3)
ax.scatter([0], [0], s=55, color="k", zorder=4)
ax.annotate("oracle\n($\\beta_0{=}0$)", (0, 0), fontsize=7.5, xytext=(6, 2), textcoords="offset points")
for rho, bx, fy in zip(rhos, b0_rho, floor_rho):
    ax.annotate(f"$\\rho$={rho:.2f}", (bx, fy), fontsize=7, xytext=(5, -3), textcoords="offset points")
ax.set_xlabel(r"verifier blind-spot rate $\beta_0$"); ax.set_ylabel(r"user-facing floor $\varepsilon_\infty$")
ax.set_xlim(left=-0.02); ax.set_ylim(bottom=-0.01)
ax.set_title(r"H2: floor scales with $\beta_0$ ($\pm$1 sd)", fontsize=10)
fig.tight_layout(); fig.savefig(FIGDIR/"exp_floor_vs_rho.pdf"); fig.savefig(FIGDIR/"exp_floor_vs_rho.png")

# D: loop-design fork, with bands
fig, ax = plt.subplots(figsize=(5.2, 3.4))
band(ax, res["oracle"], "eps", "k", label="oracle-in-loop (C)")
band(ax, res["corrective"], "eps", RED, label="corrective (A)")
band(ax, res["selftrain"], "eps", PUR, label="self-training (B)")
q0b0 = res["corrective"][0]["q"]*res["corrective"][0]["beta"]
ax.axhline(q0b0, color=GRY, ls="--", lw=1); ax.annotate(r"$q_0\beta_0$", (0.1, q0b0+0.004), fontsize=9)
ax.set_xlabel("round $t$"); ax.set_ylabel(r"user-facing error $\varepsilon_t$"); ax.set_ylim(bottom=0)
ax.legend(frameon=False, fontsize=8.5); ax.set_title("Loop-design fork (synthetic)", fontsize=10)
fig.tight_layout(); fig.savefig(FIGDIR/"exp_loop_fork.pdf"); fig.savefig(FIGDIR/"exp_loop_fork.png")

print(f"\nfigures -> {FIGDIR}\ndone.")
