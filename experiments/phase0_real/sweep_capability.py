"""
Blind-spot rate beta vs STUDENT CAPABILITY (round-0 only; no fine-tuning).

Holds the verifier fixed (gpt-4o-mini, blinded) and measures, on the same 300
GSM8K problems, how the blind-spot rate beta = P(V accepts | student wrong) and the
dashboard gap change as the student scales 0.5B -> 32B. Prediction (Corollary 1):
a stronger student makes subtler errors, so the verifier is fooled MORE -> beta
rises with capability.

Reuses run_phase0.py (Student, evaluate, verifier). Run on the H100:
  setsid ~/bin/micromamba run -n ml python sweep_capability.py > sweep.log 2>&1 &
"""
import json, gc, time
from pathlib import Path
import torch
import run_phase0 as R

HERE = Path(__file__).resolve().parent
SIZES = ["0.5B", "1.5B", "3B", "7B", "14B", "32B"]
PARAMS = {"0.5B": 0.5, "1.5B": 1.5, "3B": 3.0, "7B": 7.0, "14B": 14.0, "32B": 32.0}
N_EVAL = 300
BACKEND, VMODEL, TMODEL = "openai", "gpt-4o-mini", "gpt-4o-mini"

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train, test = R.load_gsm8k()
    eval_set = test[:N_EVAL]
    print(f"device={device} | verifier={VMODEL} | eval={len(eval_set)}", flush=True)
    teacher_cache = {}                      # shared: teacher solves are student-independent
    results = []
    for s in SIZES:
        name = f"Qwen/Qwen2.5-{s}-Instruct"
        t0 = time.time()
        try:
            stu = R.Student(name, device)
            stu.gen_bs = 8 if PARAMS[s] >= 14 else 24   # smaller batch for big models
            stu.load_base_only()
            m = R.evaluate(stu, eval_set, BACKEND, VMODEL, TMODEL, teacher_cache)
            m.update(size=s, params_b=PARAMS[s], minutes=round((time.time() - t0) / 60, 1))
            results.append(m)
            print(f"{s}: q={m['q']:.3f} beta={m['beta']:.3f} eps={m['eps']:.3f} "
                  f"dash={m['dash']:.3f}  ({m['minutes']}min)", flush=True)
            del stu; gc.collect(); torch.cuda.empty_cache()
        except Exception as e:
            print(f"{s} FAILED: {type(e).__name__}: {str(e)[:120]}", flush=True)
            gc.collect(); torch.cuda.empty_cache()
        json.dump(results, open(HERE / "sweep_results.json", "w"), indent=2)

    # ---- plot ----
    if len(results) >= 2:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        xs = [r["params_b"] for r in results]
        beta = [r["beta"] for r in results]
        gap = [r["eps"] - r["dash"] for r in results]
        fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
        ax[0].plot(xs, beta, "o-", color="#c0392b", lw=2)
        for r in results: ax[0].annotate(r["size"], (r["params_b"], r["beta"]),
                                          fontsize=7.5, xytext=(4, 4), textcoords="offset points")
        ax[0].set_xscale("log"); ax[0].set_xlabel("student size (B params, log)")
        ax[0].set_ylabel(r"blind-spot rate $\beta_0$"); ax[0].set_ylim(bottom=0)
        ax[0].set_title("Blind spot grows with student capability", fontsize=10)
        ax[1].plot(xs, [r["dash"] for r in results], "s-", color="#2e7d32", lw=2, label="dashboard")
        ax[1].plot(xs, [r["eps"] for r in results], "o-", color="#c0392b", lw=2, label="gold $\\varepsilon$")
        ax[1].set_xscale("log"); ax[1].set_xlabel("student size (B params, log)")
        ax[1].set_ylabel("error rate"); ax[1].set_ylim(bottom=0)
        ax[1].legend(frameon=False, fontsize=8.5); ax[1].set_title("Dashboard vs true error", fontsize=10)
        for a in ax: a.spines[["top", "right"]].set_visible(False)
        fig.tight_layout(); fig.savefig(HERE / "exp_beta_vs_capability.png", dpi=130)
        print(f"plot -> {HERE/'exp_beta_vs_capability.png'}", flush=True)
    print("done.", flush=True)

if __name__ == "__main__":
    main()
