"""
H5 -- mitigation exchange rate, on the same synthetic model as phase0_synthetic.

Two mitigations for the blind-spot floor, each buying reliability with cost:

  1. DECORRELATED VERIFIER ENSEMBLE. m verifiers, each competent but with an
     INDEPENDENT blind half-space; the cascade rejects if ANY member dissents. A
     wrong answer survives (is falsely accepted) only if it lies in the blind
     region of EVERY member, so the effective blind-spot rate falls ~rho**m.
     Cost: m verifier calls per query (+ more escalations to the teacher).

  2. ORACLE AUDITS. Spend a fraction x of budget gold-labelling random items each
     round and injecting them (with true labels) into the training pool. Unlike
     verifier-rejected items, audits reach BLIND-region cases the verifier
     accepts, so the student finally gets signal there. Cost: x oracle calls.

Cost axis = frontier calls per query = (verifier evals m) + (teacher escalations
p_reject) + (audit fraction x). Oracle-in-loop (perfect verifier) is the floor=0
reference. Produces a cost-reliability Pareto: ../figures/exp_mitigation.pdf.

This inherits every caveat of phase0_synthetic.py: synthetic, mechanism-level,
not evidence about real LLMs.
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

# ground truth + feature maps (same construction as phase0)
D, K, HID = 20, 10, 64
W1 = RNG.normal(size=(HID, D)) / np.sqrt(D); b1 = RNG.normal(size=HID) * 0.3
W2 = RNG.normal(size=(K, HID)) / np.sqrt(HID)
def y_star(phi): return (np.tanh(phi @ W1.T + b1) @ W2.T).argmax(1)
def make_basis(m, d=D): return (RNG.normal(size=(m, d)) / np.sqrt(d), RNG.normal(size=m) * 0.5)
def feats(b, phi): A, c = b; return np.maximum(0.0, phi @ A.T + c)
S_BASIS = make_basis(300); V_BASIS = make_basis(450)

MAXENS = 5
RDIRS = RNG.normal(size=(MAXENS, D))                          # independent blind directions
_PROJS = [RNG.normal(size=(30000, D)) @ RDIRS[i] for i in range(MAXENS)]
def fit_lr(X, y, C=3.0): return LogisticRegression(C=C, max_iter=200).fit(X, y)

def run_h5(ens_m=1, audit_x=0.0, oracle=False, rho=0.7,
           rounds=8, n0=140, n_batch=600, n_eval=3000, n_vfit=5000, seed=0):
    rng = np.random.default_rng(1000 + seed)
    samp = lambda n: rng.normal(size=(n, D))
    Xv = samp(n_vfit); Vstrong = fit_lr(feats(V_BASIS, Xv), y_star(Xv), C=6.0)
    thr = [np.quantile(_PROJS[i], rho) for i in range(ens_m)]
    def accept_of(phi, s_pred):
        if oracle:
            return s_pred == y_star(phi)                     # perfect verifier
        agree = Vstrong.predict(feats(V_BASIS, phi)) == s_pred
        blind_all = np.ones(len(phi), bool)
        for i in range(ens_m):
            blind_all &= (phi @ RDIRS[i]) < thr[i]
        return agree | blind_all

    Xe = samp(n_eval); ye = y_star(Xe)
    Xtr = samp(n0); ytr = y_star(Xtr)
    hist = []; stu = None
    for t in range(rounds + 1):
        stu = fit_lr(feats(S_BASIS, Xtr), ytr)
        se = stu.predict(feats(S_BASIS, Xe)); wrong = se != ye
        acc = accept_of(Xe, se)
        q = wrong.mean(); p_rej = float((~acc).mean())
        beta = (wrong & acc).sum() / max(wrong.sum(), 1)
        eps = float((np.where(acc, se, ye) != ye).mean())
        hist.append(dict(t=t, q=float(q), eps=eps, beta=float(beta), p_reject=p_rej))
        if t == rounds: break
        Xb = samp(n_batch); yb = y_star(Xb); sb = stu.predict(feats(S_BASIS, Xb))
        if oracle:
            aX, aY = Xb, yb
        else:
            rej = ~accept_of(Xb, sb)
            aX, aY = Xb[rej], yb[rej]
            if audit_x > 0:
                aud = rng.random(len(Xb)) < audit_x
                aX = np.vstack([aX, Xb[aud]]); aY = np.concatenate([aY, yb[aud]])
        Xtr = np.vstack([Xtr, aX]); ytr = np.concatenate([ytr, aY])
    return hist

def summ(seeds=range(20), **kw):
    hs = [run_h5(seed=s, **kw) for s in seeds]
    per_seed = np.array([np.mean([h[t]["eps"] for t in (-3, -2, -1)]) for h in hs])
    floor = float(per_seed.mean()); floor_sd = float(per_seed.std(ddof=1))
    b0 = float(np.mean([h[0]["beta"] for h in hs]))
    prej = float(np.mean([h[-1]["p_reject"] for h in hs]))
    return floor, b0, prej, floor_sd

print("=" * 72)
print("H5 -- MITIGATION EXCHANGE RATE  (rho=0.7, 20 seeds, 8 rounds)")
print("=" * 72)
print(f"{'config':<16}{'ens_m':>6}{'audit_x':>9}{'beta0':>8}{'p_rej_T':>9}"
      f"{'floor':>8}{'cost':>7}")

configs = [
    ("baseline",     dict(ens_m=1, audit_x=0.0)),
    ("ensemble m=2", dict(ens_m=2, audit_x=0.0)),
    ("ensemble m=3", dict(ens_m=3, audit_x=0.0)),
    ("ensemble m=4", dict(ens_m=4, audit_x=0.0)),
    ("audit 5%",     dict(ens_m=1, audit_x=0.05)),
    ("audit 10%",    dict(ens_m=1, audit_x=0.10)),
    ("audit 20%",    dict(ens_m=1, audit_x=0.20)),
    ("audit 40%",    dict(ens_m=1, audit_x=0.40)),
]
print(f"{'config':<16}{'ens_m':>6}{'audit_x':>9}{'beta0':>8}{'p_rej':>8}{'floor':>13}{'cost':>7}")
rows = {}
for name, kw in configs:
    floor, b0, prej, fsd = summ(**kw)
    cost = kw["ens_m"] + prej + kw["audit_x"]         # frontier calls per query
    rows[name] = dict(floor=floor, fsd=fsd, b0=b0, prej=prej, cost=cost, **kw)
    print(f"{name:<16}{kw['ens_m']:>6}{kw['audit_x']:>9.2f}{b0:>8.3f}"
          f"{prej:>8.3f}{floor:>8.3f}+/-{fsd:<4.3f}{cost:>7.2f}")
# oracle reference
ofloor, _, oprej, _ = summ(oracle=True)
print(f"{'oracle-in-loop':<16}{'--':>6}{'--':>9}{0.0:>8.3f}{oprej:>8.3f}{ofloor:>8.3f}{'--':>9}")

# ----------------------------------------------------------------------------
# Pareto figure
# ----------------------------------------------------------------------------
plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 130})
RED, BLU, GRY = "#c0392b", "#2962ff", "#777777"
fig, ax = plt.subplots(figsize=(5.6, 3.7))

ens = [rows[n] for n in ("baseline", "ensemble m=2", "ensemble m=3", "ensemble m=4")]
aud = [rows[n] for n in ("baseline", "audit 5%", "audit 10%", "audit 20%", "audit 40%")]
ax.plot([r["cost"] for r in ens], [r["floor"] for r in ens], "o-", color=BLU, lw=1.8, label="verifier ensemble (m)")
ax.plot([r["cost"] for r in aud], [r["floor"] for r in aud], "s-", color=RED, lw=1.8, label="oracle audits (x%)")
ax.scatter([rows["baseline"]["cost"]], [rows["baseline"]["floor"]], s=70, color="k", zorder=5)
ax.annotate("baseline\n(single verifier)", (rows["baseline"]["cost"], rows["baseline"]["floor"]),
            fontsize=7.5, xytext=(8, 2), textcoords="offset points")
for r, lab in zip(ens[1:], ["m=2", "m=3", "m=4"]):
    ax.annotate(lab, (r["cost"], r["floor"]), fontsize=7.5, xytext=(4, 4), textcoords="offset points", color=BLU)
for r, lab in zip(aud[1:], ["5%", "10%", "20%", "40%"]):
    ax.annotate(lab, (r["cost"], r["floor"]), fontsize=7.5, xytext=(4, -9), textcoords="offset points", color=RED)
ax.axhline(ofloor, color=GRY, ls="--", lw=1)
ax.annotate("oracle-in-loop (perfect verifier)", (max(r["cost"] for r in ens)*0.42, ofloor+0.004),
            fontsize=7.5, color=GRY)
ax.set_xlabel("cost: frontier calls per query  (verifier $m$ + escalations + audit $x$)")
ax.set_ylabel(r"user-facing floor $\varepsilon_\infty$")
ax.set_ylim(bottom=0); ax.legend(frameon=False, fontsize=8.5, loc="upper right")
ax.set_title("H5: buying back reliability (synthetic)", fontsize=10)
fig.tight_layout()
fig.savefig(FIGDIR / "exp_mitigation.pdf"); fig.savefig(FIGDIR / "exp_mitigation.png")
print(f"\nfigure -> {FIGDIR/'exp_mitigation.pdf'}\ndone.")
