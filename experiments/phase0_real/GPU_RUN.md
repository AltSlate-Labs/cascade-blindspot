# Running Phase-0 on a GPU

The harness (`run_phase0.py`) is device-agnostic (CUDA > MPS > CPU) and configurable.
On a Mac (MPS) each round is ~80 min and per-round batches are too small for a 1.5B
student to visibly improve. A single small cloud GPU fixes both: minutes per round,
and batches large enough that the loop actually raises the student's accuracy — which
is what the multi-round *scissors* needs.

## What's already validated (Mac, round 0)

The blind spot is real on real LLMs, in the *hardest* case (independent verifier):

| metric | round 0 (Qwen2.5-1.5B student, gpt-4o-mini verifier, GSM8K) |
|---|---|
| raw error q | 0.35 |
| **blind-spot rate β** | **0.31** (verifier accepts 31% of the student's wrong answers) |
| user-facing error ε | 0.13 |
| **dashboard (verifier-estimated) error** | **0.03** — 4× below the truth |

The GPU run's job is the **trajectory** over rounds 1–N: does ε *floor* near q₀β₀ (≈0.11)
while q falls and the dashboard stays low? (See `round01_data.txt` for the Mac round 0–1;
round 1's q rose because the MPS run used tiny batches + too-high lr — both fixed here.)

## Any 24 GB GPU is plenty (1.5B student). ~1–2 h, a few dollars.

RunPod / Lambda / Modal / Vast — an A10, L4, A100, or 4090 all work.

### RunPod (simplest)

1. Launch a pod: **"RunPod PyTorch 2.x" template**, GPU = A10/L4/A100 (24 GB+), 40 GB disk.
2. In the pod's web terminal:

```bash
git clone <your repo>  # or scp this phase0_real/ folder up
cd phase0_real
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...            # ROTATE the one used on the Mac first
python3 run_phase0.py \
    --student Qwen/Qwen2.5-1.5B-Instruct \
    --rounds 5 --n-eval 300 --n-batch 512 \
    --gen-bs 48 --train-bs 8 --epochs 2 \
    --variants corrective,frozen
```

3. Copy `results_real.json` and `exp_real_scissors.png` back (RunPod file browser or `scp`).
4. Stop the pod (billing is per-minute).

### Modal (serverless, no pod management)

Wrap the same command in a `modal run` function with a T4/A10 and the `OPENAI_API_KEY`
secret; Modal spins the GPU up only for the run.

## Cost / time

- GPU: ~$0.4–2/h × 1–2 h ≈ **$1–4**.
- OpenAI (gpt-4o-mini verifier + teacher): 300 eval × 6 evals + 512 batch × 5 rounds
  of verifier calls ≈ 5–8k calls ≈ **$2–5**.
- Wall clock: **~1–2 h** at these sizes.

## Reading the result (the go/no-go)

Success = the **scissors**:
- `q` (raw error) **falls** over rounds — the loop improves the student (needs the bigger
  `--n-batch`; if q is flat, raise it / raise `--epochs`).
- `eps` (gold user-facing error) **floors** at a positive value near q₀·β₀ — the
  conservation prediction.
- `dash` (verifier-estimated error) **stays low** — a persistent, large gap = the
  hidden harm. `exp_real_scissors.png` plots all three.

Also compare against the `frozen` control (no training). If `eps` under `corrective`
floors above the `frozen`/oracle baseline and tracks q₀β₀, that's the real-model
confirmation of the paper's Corollary + conservation claim.

## Folding into the paper (v2)

- If the scissors appear: add a real-LLM figure beside the synthetic ones in §7, and
  change the abstract/intro from "no real-LLM experiments" to "a real-model Phase-0 on
  GSM8K confirms β>0 and the dashboard gap; the multi-round trajectory floors near q₀β₀."
  That single figure moves the paper from position-piece to demonstrated-on-real-LLMs.
- Even round 0 alone (already in hand) supports a "real-model probe" paragraph:
  an *independent* frontier verifier has β=0.31 on a 1.5B student, and its estimated
  error understates the truth 4×.

## Notes

- Verifier/teacher are the OpenAI API, so they work identically from the GPU box — only
  the student runs on the GPU.
- For a *same-family* verifier (higher β, tests Corollary 1 directly), add
  `--backend ollama --verifier-model qwen2.5:7b --teacher-model qwen2.5:7b` after
  `ollama pull qwen2.5:7b` on the GPU box (downloads work fine there).
- Rotate the OpenAI key that was pasted into the earlier chat session.
