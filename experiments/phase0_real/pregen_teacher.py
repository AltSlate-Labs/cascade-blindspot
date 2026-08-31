"""
Pre-generate a same-family teacher's solutions for GSM8K training problems, so the
loop can train on them without holding the big teacher in memory alongside the student.
  micromamba run -n ml python pregen_teacher.py Qwen/Qwen2.5-32B-Instruct 2800 teacher_32b.json
Writes {question: solution}.
"""
import sys, json
from pathlib import Path
import run_phase0 as R

MODEL = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 2800
OUT = sys.argv[3] if len(sys.argv) > 3 else "teacher_cache.json"
GEN_BS = int(sys.argv[4]) if len(sys.argv) > 4 else 8

train, _ = R.load_gsm8k()
probs = [q for q, _ in train[:N]]
print(f"pre-generating {len(probs)} teacher solutions with {MODEL} (gen_bs={GEN_BS})", flush=True)

R.SYS = "You are a careful math tutor. Solve step by step and end with 'The final answer is: <number>'."
stu = R.Student(MODEL, "cuda"); stu.gen_bs = GEN_BS; stu.load_base_only()
sols = stu.generate(probs, max_new=512)
cache = {probs[i]: sols[i] for i in range(len(probs))}
Path(OUT).write_text(json.dumps(cache))
print(f"wrote {len(cache)} teacher solutions -> {OUT}", flush=True)
