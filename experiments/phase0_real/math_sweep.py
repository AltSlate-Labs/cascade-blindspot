"""
Blind-spot rate beta vs VERIFIER STRENGTH, on HARD MATH (Hendrycks MATH-500, L4-5).

The point (answering "use a strong verifier"): on a task at the frontier's edge, even
a top-benchmark verifier still accepts a real fraction of the student's wrong answers.
Student solutions are generated ONCE; each verifier (weak -> frontier) judges the same
cache. Oracle = math-verify symbolic equivalence (independent of the verifier).
"""
import json, re
from pathlib import Path
import run_phase0 as R
from math_verify import parse, verify

HERE = Path(__file__).resolve().parent
STUDENT = "Qwen/Qwen2.5-7B-Instruct"
N_EVAL = 200
VERIFIERS = ["gpt-4o-mini", "gpt-4.1", "gpt-5-mini"]     # weak -> strong -> frontier
MATH_SYS = ("You are an expert mathematician. Solve the problem step by step and give the "
            "final answer in \\boxed{}.")

def load_hard_math():
    from datasets import load_dataset
    d = load_dataset("HuggingFaceH4/MATH-500", split="test")
    return [(x["problem"], x["answer"]) for x in d if x["level"] in (4, 5)][:N_EVAL]

def is_correct(sol, gold):
    try:
        g = gold if "\\boxed" in gold else f"\\boxed{{{gold}}}"
        return bool(verify(parse(g), parse(sol)))
    except Exception:
        return False

def math_verifier_accept(vmodel, problem, sol):
    msg = [{"role": "user", "content":
        f"Problem:\n{problem}\n\nProposed solution:\n{sol}\n\n"
        "Is the final answer in this solution correct? Reason briefly, then end with "
        "exactly 'VERDICT: YES' or 'VERDICT: NO'."}]
    out = R.chat("openai", vmodel, msg, num_predict=500)
    m = re.search(r"VERDICT:\s*(YES|NO)", out, re.I)
    return (m.group(1).upper() == "YES") if m else ("yes" in out.lower()[-40:])

def main():
    data = load_hard_math()
    probs = [p for p, _ in data]; golds = [g for _, g in data]
    print(f"hard MATH L4-5: {len(data)} problems | student={STUDENT}", flush=True)

    R.SYS = MATH_SYS                                     # Student reads module-level SYS
    stu = R.Student(STUDENT, "cuda"); stu.gen_bs = 24; stu.load_base_only()
    sols = stu.generate(probs, max_new=1024)            # MATH solutions are long
    s_ok = [is_correct(sols[i], golds[i]) for i in range(len(data))]
    wrong = sum(1 for x in s_ok if not x)
    q = wrong / len(data)
    print(f"student raw error q on hard MATH: {q:.3f}  ({wrong}/{len(data)} wrong)", flush=True)

    results = []
    for vm in VERIFIERS:
        v_acc = R.pmap(lambda i: math_verifier_accept(vm, probs[i], sols[i]), range(len(data)), workers=16)
        blind = sum(1 for i in range(len(data)) if (not s_ok[i]) and v_acc[i])
        res = dict(verifier=vm, q=round(q, 4), beta=round(blind / max(wrong, 1), 4),
                   blind_mass=round(blind / len(data), 4),
                   p_reject=round(sum(1 for a in v_acc if not a) / len(data), 4))
        results.append(res); print("RESULT", json.dumps(res), flush=True)
        json.dump(results, open(HERE / "math_sweep_results.json", "w"), indent=2)

    # plot beta vs verifier strength
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5.4, 3.6))
        xs = list(range(len(results)))
        ax.plot(xs, [r["beta"] for r in results], "o-", color="#c0392b", lw=2)
        ax.set_xticks(xs); ax.set_xticklabels([r["verifier"] for r in results], rotation=15)
        ax.set_ylabel(r"blind-spot rate $\beta$"); ax.set_ylim(bottom=0)
        ax.set_title(f"Hard MATH: blind spot persists even for a frontier verifier\n(student q={q:.2f})", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout(); fig.savefig(HERE / "exp_beta_vs_verifier_math.png", dpi=130)
        print("plot -> exp_beta_vs_verifier_math.png", flush=True)
    except Exception as e:
        print("plot skipped:", e)
    print("done.", flush=True)

if __name__ == "__main__":
    main()
