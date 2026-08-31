"""
Render the real-model figures for the paper, one consistent visual system, from the
committed result JSONs. Vector PDF (for crisp embeds) + PNG (for preview).
  python3 make_real_figures.py
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
FIG = HERE.parent / "figures"
FIG.mkdir(exist_ok=True)

# --- one visual system (harmonise with the paper) ---
plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "cm", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.7, "figure.dpi": 140,
})
RED, GRN, BLU, GRY, ORG = "#b5341f", "#2e7d32", "#2453c4", "#6b6b6b", "#d08a1d"

Z = 1.96  # 95%
def wilson(p, n):
    """95% Wilson score interval for a proportion p estimated from n trials.
    Returns (minus, plus) = asymmetric half-widths for matplotlib yerr."""
    if n <= 0:
        return 0.0, 0.0
    c = (p + Z*Z/(2*n)) / (1 + Z*Z/n)
    h = Z/(1 + Z*Z/n) * (p*(1-p)/n + Z*Z/(4*n*n))**0.5
    lo, hi = max(0.0, c - h), min(1.0, c + h)
    return p - lo, hi - p

def yerr(ps, ns):
    lo, hi = zip(*[wilson(p, n) for p, n in zip(ps, ns)])
    return np.array([lo, hi])

def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / f"{name}.pdf"); fig.savefig(FIG / f"{name}.png", dpi=140)
    plt.close(fig); print("wrote", name)

# ============================================================ 1. real scissors
d = json.load(open(HERE / "results_real.json"))
NEV = d["config"]["n_eval"]  # 300; eps/dash/q are proportions over this
corr = d["runs"]["corrective"]; froz = d["runs"]["frozen"]
x = [r["t"] for r in corr]
eps = [r["eps"] for r in corr]; dash = [r["dash"] for r in corr]; q = [r["q"] for r in corr]
fig, ax = plt.subplots(figsize=(5.4, 3.5))
ax.plot(x, q, ":", color=GRY, lw=1.6, label=r"raw error $q_t$")
ax.fill_between(x, dash, eps, color=GRY, alpha=0.13)
# 95% Wilson CIs (n=300) on the two lines whose gap is the claim
ax.errorbar(x, eps, yerr=yerr(eps, [NEV]*len(eps)), fmt="-", color=RED, lw=2.2,
            capsize=2, elinewidth=0.8, label=r"gold user-facing $\varepsilon_t$")
ax.errorbar(x, dash, yerr=yerr(dash, [NEV]*len(dash)), fmt="-", color=GRN, lw=2.2,
            capsize=2, elinewidth=0.8, label="verifier-estimated (dashboard)")
ax.plot(x, [r["eps"] for r in froz], "--", color=RED, lw=1.1, alpha=0.7)
ax.annotate("frozen $\\varepsilon_t$ (no training)", (x[-1], froz[-1]["eps"]),
            fontsize=7.5, color=RED, xytext=(-4, -12), textcoords="offset points", ha="right")
ax.set_xlabel("round $t$"); ax.set_ylabel("error rate"); ax.set_ylim(bottom=0)
ax.legend(frameon=False, fontsize=8.5, loc="center right")
ax.set_title("Real LLMs (Qwen2.5-7B, GSM8K): the dashboard is blind", fontsize=9.5)
ax.text(0.02, 0.02, "error bars: 95% Wilson CI, $n=300$", transform=ax.transAxes,
        fontsize=6.8, color=GRY, style="italic")
save(fig, "real_scissors")

# ============================================================ 2. beta vs student
s = sorted(json.load(open(HERE / "sweep_results.json")), key=lambda r: r["params_b"])
xs = [r["params_b"] for r in s]
bs = [r["beta"] for r in s]
# beta is measured only over WRONG answers: n = q * n_eval (n_eval=300)
nw = [max(1, round(r["q"] * 300)) for r in s]
fig, ax = plt.subplots(figsize=(5.4, 3.5))
ax.errorbar(xs, bs, yerr=yerr(bs, nw), fmt="o-", color=RED, lw=2, ms=6,
            capsize=3, elinewidth=0.9)
for r, n in zip(s, nw):
    ax.annotate(f"{r['size']}\n($n{{=}}{n}$)", (r["params_b"], r["beta"]), fontsize=6.8,
                xytext=(5, 6), textcoords="offset points", color=RED)
ax.set_xscale("log")
ax.set_xticks(xs); ax.set_xticklabels([r["size"] for r in s], fontsize=7.5)
ax.minorticks_off()
ax.set_xlabel("student size (parameters, log scale)")
ax.set_ylabel(r"blind-spot rate $\beta_0$"); ax.set_ylim(0, 0.72)
ax.set_title("The blind spot grows with student capability\n(GSM8K, fixed gpt-4o-mini verifier)", fontsize=9.5)
ax.text(0.02, 0.93, "error bars: 95% Wilson CI on the wrong-answer subset",
        transform=ax.transAxes, fontsize=6.8, color=GRY, style="italic")
save(fig, "beta_vs_student")

# ============================================================ 3. beta vs verifier (hard MATH)
m = json.load(open(HERE / "math_sweep_results.json"))
NM = 200  # math_sweep N_EVAL
labels = [r["verifier"] for r in m]; xi = list(range(len(m)))
betas = [r["beta"] for r in m]; prej = [r["p_reject"] for r in m]
nw_m = [max(1, round(r["q"] * NM)) for r in m]        # beta denominator = wrong count
fig, ax = plt.subplots(figsize=(5.4, 3.5))
ax.errorbar(xi, betas, yerr=yerr(betas, nw_m), fmt="o-", color=RED, lw=2, ms=6,
            capsize=3, elinewidth=0.9, label=r"blind spot $\beta$")
ax.errorbar(xi, prej, yerr=yerr(prej, [NM]*len(m)), fmt="s--", color=GRY, lw=1.6, ms=5,
            capsize=3, elinewidth=0.8, label="escalation rate (cost)")
# true error rate the escalation must beat to be worth its price
ax.axhline(m[0]["q"], color=GRN, lw=1.0, ls=":", alpha=0.9)
ax.text(len(m)-1.02, m[0]["q"]+0.008, f"true error rate {m[0]['q']:.2f}",
        fontsize=6.8, color=GRN, ha="right")
for i, r in enumerate(m):
    ax.annotate(f"{r['beta']:.2f}", (i, r["beta"]), fontsize=7.5,
                xytext=(7, 2), textcoords="offset points", ha="left", color=RED)
ax.set_xticks(xi); ax.set_xticklabels(["gpt-4o-mini\n(cheap)", "gpt-4.1", "gpt-5-mini"], fontsize=7.5)
ax.set_xlim(-0.35, len(m)-0.5)
ax.set_ylabel("rate"); ax.set_ylim(0, 0.62)
ax.legend(frameon=False, fontsize=8.5, loc="upper center")
ax.set_title("Blind spot is the price of a cheap verifier\n(hard MATH L4-5, Qwen2.5-7B student)", fontsize=9.5)
ax.text(0.02, 0.02, r"$\beta$ CI over wrong-answer subset ($n\approx78$); escalation CI $n=200$",
        transform=ax.transAxes, fontsize=6.5, color=GRY, style="italic")
save(fig, "beta_vs_verifier")

print("done ->", FIG)
