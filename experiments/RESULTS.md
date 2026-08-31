# Phase-0 synthetic mechanism study — results

**What this is.** A controlled synthetic experiment (K=10 classification;
logistic student on random features improving with data; a fixed verifier that
rubber-stamps in a random fraction ρ of input space and verifies accurately
elsewhere). It tests whether the blind-spot conservation dynamics **emerge** from
the loop, not whether the effect holds on real LLMs. See the module docstring for
the model. Reproducible: `python3 phase0_synthetic.py` (fixed seeds, ~65 s CPU).

**What it is NOT.** Evidence about real language models. The blind spot's
*existence* is modelled (via ρ); its *dynamics under the loop* are the emergent
output being tested. The real-LLM Phase-0 (GSM8K, blinded frontier verifier)
remains the necessary next step and needs a GPU + a frontier API.

## Headline numbers (ρ = 0.7, K = 10, **20 seeds**, 8 rounds, mean ± sd)

| variant           | q₀    | β₀    | q_T       | ε_T        | q₀·β₀ | dash_T |
|-------------------|-------|-------|-----------|------------|-------|--------|
| corrective (A)    | 0.438 | 0.764 | 0.304±.01 | **0.249±.01** | 0.334 | 0.046  |
| self-training (B) | 0.438 | 0.764 | 0.356±.02 | **0.296±.01** | 0.334 | 0.045  |
| oracle-in-loop (C)| 0.438 | 0.000 | 0.248±.01 | 0.000      | —     | 0.000  |
| β₀=0 matched      | 0.438 | 0.000 | 0.283±.01 | 0.000      | —     | 0.000  |
| frozen (D)        | 0.438 | 0.764 | 0.438±.02 | **0.334±.02** | 0.334 | 0.048  |

Self-training raw error **per round**: 0.438, 0.404, 0.388, 0.378, 0.371, 0.368,
0.364, 0.359, 0.356 — it **falls**, it does not rise.

## Verdict per hypothesis

- **H1 (positive floor) — supported.** User-facing error floors at ε_T = 0.249±.01
  rather than vanishing; oracle-in-loop reaches 0. The **data-matched β₀=0 control**
  (perfect verifier, trained on wrong items only) floors at q_T = 0.283, so the
  verifier-induced excess in raw error is 0.304 − 0.283 = **+0.021** — the floor is
  real, not merely the student's capacity limit.
- **H2 (blind-spot scaling) — supported, cleanly.** Floor rises monotonically with
  β₀: 0.094 → 0.130 → 0.173 → 0.223 → 0.284 → 0.362 (±.005–.014) as β₀ goes
  0.19 → 0.96. At the lowest β₀ the floor slightly exceeds q₀β₀ (capacity floor
  binds), so ε∞ ≲ q₀β₀ holds when β₀ is large relative to capacity. (`exp_floor_vs_rho`)
- **H3 (loop-design hazard) — supported, CORRECTED.** Self-training floors strictly
  higher than corrective (ε_T 0.296 vs 0.249; q_T 0.356 vs 0.304). But its raw error
  **falls** over rounds (0.438→0.356), it does **not** rise — an earlier "rises over
  rounds" claim was wrong (0.356 vs 0.304 is B-vs-A at the final round, not a rise).
  The effect is reinforcement *relative to* corrective; the absolute non-monotone
  rise is a real-model prediction, not seen here. (`exp_loop_fork`)
- **H4 (dashboard blindness) — supported as a LEVEL gap.** The dashboard reads 4.6%
  while the truth is 24.9% — a large *persistent* gap. It does **not** widen under
  either loop (gold error falls, dashboard flat, so the gap narrows); the widening
  form needs the blind mass to grow and is a real-model prediction, untested here.
  (`exp_scissors`)
- **H5 (mitigation exchange rate) — supported; the two remedies differ sharply.**
  See `h5_mitigation.py` / `results_h5.txt` (ρ=0.7, 20 seeds).

  | config          | β₀    | floor ε   | cost (calls/query) |
  |-----------------|-------|-----------|--------------------|
  | baseline        | 0.764 | 0.252±.01 | 1.09               |
  | ensemble m=2    | 0.585 | 0.189±.01 | 2.15               |
  | ensemble m=3    | 0.452 | 0.155±.01 | 3.19               |
  | ensemble m=4    | 0.404 | 0.143±.01 | 4.21               |
  | audit 5%        | 0.764 | 0.239±.01 | 1.13               |
  | audit 20%       | 0.764 | 0.225±.01 | 1.28               |
  | audit 40%       | 0.764 | 0.216±.01 | 1.48               |
  | oracle-in-loop  | 0.000 | 0.000     | —                  |

  The **decorrelated ensemble is the strong lever** — β₀ falls ~ρᵐ and the floor
  drops 0.25→0.14 — but cost scales with m. **Random audits are cheap but weak**:
  40% auditing only reaches 0.22, because most of the budget lands on non-blind
  items (an untargeted audit spends most checks where the verifier already sees).
  Neither reaches the oracle floor within budget. Refinement for the real study:
  audit where the verifier is *least certain*, not at random. (`exp_mitigation`)

## The honest nuance (a finding, not a failure)

The conjecture anchored the corrective floor at q₀·β₀ = 0.334. **Both** learning
variants land *below* it — corrective 0.249, self-training 0.296 — and only the
frozen control sits *on* it (0.334, no learning to spill). The predicted
*above*-anchor regime for self-training was **not** observed: self-healing
(generalization spillover into the blind region) dominates even when accepted
outputs are recycled as labels, so self-training merely floors above corrective,
not above q₀·β₀. The ordering that held: frozen (= q₀·β₀) > self-training >
corrective > oracle. The equality ε_∞ = q₀·β₀ should be read as ε_∞ ≲ q₀·β₀ for the
corrective loop, the 0.334 − 0.249 gap measuring self-healing (fitted spillover
γ ≈ 0.83; see Appendix A of the paper). The two-population model predicts
self-training crosses the anchor only when reinforcement outweighs γ·d₀ — a regime
this readily-generalizing synthetic student does not reach.

## Figures (in ../figures/)

- `exp_scissors.{png,pdf}` — dashboard-vs-gold divergence (H4)
- `exp_conservation.{png,pdf}` — detectable band drains, blind band conserved
- `exp_floor_vs_rho.{png,pdf}` — floor vs β₀ (H2)
- `exp_loop_fork.{png,pdf}` — corrective vs self-training vs oracle (H1, H3)
- `exp_mitigation.{png,pdf}` — cost–reliability Pareto (H5)

## Scripts

- `phase0_synthetic.py` — H1–H4 (main study). `results.txt` = its stdout.
- `h5_mitigation.py` — H5 ensemble + audit Pareto. `results_h5.txt` = its stdout.

Both reproducible on CPU (~1 min each), fixed seeds.
