"""
One student size for the beta-vs-capability sweep (vLLM generation + gpt-4o-mini
verifier). Run per-size as a subprocess so GPU memory frees cleanly between models:
  micromamba run -n vllm python sweep_one.py 7B
Appends a result row to sweep_results.json.
"""
import sys, json
from pathlib import Path
import run_phase0 as R          # load_gsm8k, extract_final_number, verifier_accept, pmap, SYS
from vllm import LLM, SamplingParams

SIZE = sys.argv[1]
PARAMS = {"0.5B": 0.5, "1.5B": 1.5, "3B": 3.0, "7B": 7.0, "14B": 14.0, "32B": 32.0}
N_EVAL = 300
BACKEND, VMODEL = "openai", "gpt-4o-mini"
HERE = Path(__file__).resolve().parent

_, test = R.load_gsm8k()
qs = [q for q, _ in test[:N_EVAL]]
golds = [g for _, g in test[:N_EVAL]]

name = f"Qwen/Qwen2.5-{SIZE}-Instruct"
gmu = 0.94 if PARAMS[SIZE] >= 14 else 0.85
llm = LLM(model=name, dtype="bfloat16", gpu_memory_utilization=gmu,
          max_model_len=2048, enforce_eager=True, trust_remote_code=True)
sp = SamplingParams(temperature=0.0, max_tokens=512)
msgs = [[{"role": "system", "content": R.SYS}, {"role": "user", "content": q}] for q in qs]
outs = llm.chat(msgs, sp, use_tqdm=False)
sols = [o.outputs[0].text for o in outs]

s_ok = [R.extract_final_number(sols[i]) == golds[i] for i in range(N_EVAL)]
v_acc = R.pmap(lambda i: R.verifier_accept(BACKEND, VMODEL, qs[i], sols[i]), range(N_EVAL), workers=32)
wrong = sum(1 for x in s_ok if not x)
blind = sum(1 for i in range(N_EVAL) if (not s_ok[i]) and v_acc[i])
res = dict(size=SIZE, params_b=PARAMS[SIZE],
           q=round(wrong / N_EVAL, 4),
           beta=round(blind / max(wrong, 1), 4),
           blind_mass=round(blind / N_EVAL, 4),
           p_reject=round(sum(1 for a in v_acc if not a) / N_EVAL, 4))
print("RESULT", json.dumps(res), flush=True)
f = HERE / "sweep_results.json"
data = json.loads(f.read_text()) if f.exists() else []
data.append(res); f.write_text(json.dumps(data, indent=2))
