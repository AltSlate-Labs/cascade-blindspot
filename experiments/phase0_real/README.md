# Phase-0 real-LLM wind-tunnel (GSM8K)

The real-model counterpart to the synthetic study — turns the paper's §6 design
into an actual run. Validated end-to-end on this machine (smoke test passed; the
full loop generate → verify → correct → LoRA-train → re-eval runs on MPS).

## Status: round 0 validated; multi-round trajectory → GPU (see `GPU_RUN.md`)

A real Mac run (Qwen2.5-1.5B student, gpt-4o-mini verifier, GSM8K) gave, at round 0:
**β = 0.31** (an *independent* frontier verifier accepts 31% of the student's wrong
answers) and a **4× dashboard gap** (verifier-estimated error 0.03 vs gold 0.13).
That is the blind spot + dashboard-blindness demonstrated on a real LLM — the core
claim, in hand (`round01_data.txt`). The multi-round *scissors* needs bigger
per-round batches than MPS can chew (round-1 q rose: too-small batches + high lr,
both now fixed — lr 2e-5, grad clip, `--gen-bs/--train-bs` flags). Run the full
trajectory on a GPU: **see `GPU_RUN.md`**.

## What it does

| Role | Instantiation |
|------|---------------|
| Student | small open model (Qwen2.5-1.5B-Instruct), **LoRA fine-tuned each round** (transformers, MPS) |
| Verifier | `--backend openai` (gpt-4o-mini) or `ollama` — judges the student's solution **without the gold answer** (blinded → fuzzy) |
| Teacher | same model — writes a fresh solution for rejected items |
| Oracle | exact-match of the extracted number to GSM8K gold — **measurement only**, never routes or trains |

Cumulative-from-base each round (reload base + fresh LoRA, train on all accumulated
teacher corrections), matching the paper's protocol. Variants: `corrective` (train
on corrections of rejects) vs `frozen` (never retrain).

## Metrics per round (on the held-out eval set, via the oracle)

- `q` raw student error · `p_reject` escalation rate · `beta` = P(V accepts | student wrong) — the **blind-spot rate**
- `eps` user-facing error (deliver student if accepted, else teacher) · `dash` verifier-estimated error of the delivered stream (the **dashboard**, no gold)

**Go/no-go:** does the `dash` vs `eps` gap open while both stay > 0 (the scissors)?

## Run

```sh
export OPENAI_API_KEY=...            # for --backend openai (default)
python3 run_phase0.py --smoke                                   # ~10 min pipeline check
python3 run_phase0.py --rounds 3 --n-eval 100 --n-batch 64      # go/no-go (hours on MPS)
```
Outputs `results_real.json` + `exp_real_scissors.png`.

## Design note — the β regime matters

The effect only appears when the verifier is **genuinely fuzzy** (β > 0): the
student's wrong answers must be plausible enough to fool the judge. A *strong,
independent* verifier (e.g. gpt-4o-mini on GSM8K) catches blatant errors → β ≈ 0 →
**no floor** — which is exactly the paper's Corollary 1 ("strong verifier → no
blind spot"), not a refutation. Levers that raise β: a **stronger student** (subtler
errors), a **same-family verifier** (correlated blind spots — swap in a Qwen verifier
via transformers or `--backend ollama qwen2.5:3b`), or a **harder task** (MATH).

## Environment notes

- Ollama's server runs, but model *pulls* failed with EOF here (registry/CDN through
  this environment) — hence the OpenAI backend as the reliable default.
- 17 GB RAM: 1.5B student + API verifier fits comfortably; a *local* 3B+ verifier via
  transformers alongside training is tight (use a 0.5B student or a GPU).
- A rented GPU (Modal/RunPod) is the clean way to run the same-family, larger-student
  config that best exhibits the blind spot.
