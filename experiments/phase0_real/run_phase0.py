"""
Phase-0 REAL-LLM wind-tunnel experiment for the blind-spot conservation claim.

Student  = a small open model fine-tuned with LoRA each round (transformers, MPS).
Verifier = a stronger model (Ollama) that judges the student's solution WITHOUT the
           gold answer -> genuinely fuzzy, blind-spot rate beta > 0.
Teacher  = same stronger model, produces a fresh solution on rejected items.
Oracle   = exact-match of the extracted final number to GSM8K gold (MEASUREMENT ONLY;
           never routes or trains).

Per round we log, on a held-out eval set (via the oracle):
  q      = raw student error
  p_rej  = escalation rate (verifier rejects)
  beta   = P(verifier accepts | student wrong)   [the blind-spot rate]
  eps    = user-facing error (deliver student if accepted, else teacher)
  dash   = verifier-estimated error of the delivered stream (no gold) = the DASHBOARD

Go/no-go: does the gap between `dash` and `eps` open while both floor above zero
(scissors)?  Variants: corrective (train on teacher corrections of rejects) vs
frozen (never retrain).  Cumulative-from-base each round.

Usage:
  python3 run_phase0.py --smoke                 # tiny end-to-end validation (minutes)
  python3 run_phase0.py --rounds 4 --n-eval 150 --n-batch 96   # real go/no-go (hours)
"""
import argparse, json, os, re, time, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import torch

def pmap(fn, items, workers=24):
    """Parallel map (order-preserving) — for concurrent verifier/teacher API calls."""
    items = list(items)
    if not items:
        return []
    with ThreadPoolExecutor(max_workers=min(workers, len(items))) as ex:
        return list(ex.map(fn, items))

HERE = Path(__file__).resolve().parent
OLLAMA = "http://localhost:11434/api/chat"
OPENAI = "https://api.openai.com/v1/chat/completions"

# ----------------------------------------------------------------------------
# GSM8K data + answer extraction
# ----------------------------------------------------------------------------
def load_gsm8k():
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main")
    def gold(a):  # answers end with "#### <number>"
        m = re.search(r"####\s*([-\d,\.]+)", a)
        return norm_num(m.group(1)) if m else None
    train = [(q, gold(a)) for q, a in zip(ds["train"]["question"], ds["train"]["answer"])]
    test = [(q, gold(a)) for q, a in zip(ds["test"]["question"], ds["test"]["answer"])]
    return train, test

def norm_num(s):
    if s is None: return None
    s = str(s).replace(",", "").replace("$", "").strip().rstrip(".")
    try:
        f = float(s)
        return str(int(f)) if f == int(f) else str(f)
    except ValueError:
        return None

def extract_final_number(text):
    if not text: return None
    m = re.findall(r"(?:final answer|answer)\s*(?:is|:)?\s*\$?\s*(-?[\d,]+\.?\d*)", text, re.I)
    if m: return norm_num(m[-1])
    nums = re.findall(r"-?\d[\d,]*\.?\d*", text)
    return norm_num(nums[-1]) if nums else None

# ----------------------------------------------------------------------------
# Ollama verifier + teacher
# ----------------------------------------------------------------------------
def _post(url, body, headers):
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", **headers})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                return json.loads(r.read())
        except (urllib.error.URLError, TimeoutError, urllib.error.HTTPError) as e:
            if attempt == 3: raise
            time.sleep(2 + 2 * attempt)

def chat(backend, model, messages, temperature=0.0, num_predict=400):
    """Unified chat: backend in {'openai','ollama'}."""
    if backend == "openai":
        reasoning = any(model.startswith(p) for p in ("gpt-5", "o1", "o3", "o4"))
        body = {"model": model, "messages": messages}
        if reasoning:                                   # gpt-5/o-series: different param, needs headroom
            body["max_completion_tokens"] = max(num_predict, 3000)
        else:
            body["temperature"] = temperature; body["max_tokens"] = num_predict
        d = _post(OPENAI, body, {"Authorization": "Bearer " + os.environ["OPENAI_API_KEY"]})
        return d["choices"][0]["message"]["content"]
    d = _post(OLLAMA, {"model": model, "messages": messages, "stream": False,
                       "options": {"temperature": temperature, "num_predict": num_predict}}, {})
    return d["message"]["content"]

def verifier_accept(backend, model, question, solution):
    """Blinded verifier: judge correctness WITHOUT the gold answer."""
    msg = [{"role": "user", "content":
        f"Problem:\n{question}\n\nProposed solution:\n{solution}\n\n"
        "Is the final numerical answer in this solution correct? "
        "Think briefly, then end your reply with exactly 'VERDICT: YES' or 'VERDICT: NO'."}]
    out = chat(backend, model, msg, num_predict=300)
    m = re.search(r"VERDICT:\s*(YES|NO)", out, re.I)
    return (m.group(1).upper() == "YES") if m else ("yes" in out.lower()[-40:])

CACHED_TEACHER = {}     # question -> pre-generated solution (e.g. a same-family teacher)

def teacher_solve(backend, model, question):
    if question in CACHED_TEACHER:
        return CACHED_TEACHER[question]
    msg = [{"role": "user", "content":
        f"Solve this math problem step by step. End with 'The final answer is: <number>'.\n\n{question}"}]
    return chat(backend, model, msg, num_predict=500)

# ----------------------------------------------------------------------------
# Student (transformers + LoRA on MPS)
# ----------------------------------------------------------------------------
SYS = "You are a careful math tutor. Solve step by step and end with 'The final answer is: <number>'."

class Student:
    def __init__(self, name, device):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.name, self.device = name, device
        self.tok = AutoTokenizer.from_pretrained(name)
        if self.tok.pad_token is None: self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "left"
        self._base = None
        self.model = None
        self.gen_bs = 8
        self.train_bs = 2

    def _load_base(self):
        from transformers import AutoModelForCausalLM
        return AutoModelForCausalLM.from_pretrained(
            self.name, torch_dtype=torch.bfloat16).to(self.device)

    def fresh_lora(self, r=16):
        """Reload base + a fresh LoRA adapter (cumulative-from-base protocol)."""
        from peft import LoraConfig, get_peft_model
        base = self._load_base()
        cfg = LoraConfig(r=r, lora_alpha=2 * r, lora_dropout=0.05, bias="none",
                         task_type="CAUSAL_LM",
                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                         "gate_proj", "up_proj", "down_proj"])
        self.model = get_peft_model(base, cfg)

    def load_base_only(self):
        self.model = self._load_base()

    def _prompt_ids(self, question):
        msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": question}]
        return self.tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)

    @torch.no_grad()
    def generate(self, questions, max_new=320, bs=None):
        bs = bs or self.gen_bs
        self.model.eval()
        outs = []
        for i in range(0, len(questions), bs):
            chunk = [self._prompt_ids(q) for q in questions[i:i + bs]]
            enc = self.tok(chunk, return_tensors="pt", padding=True).to(self.device)
            gen = self.model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                                      pad_token_id=self.tok.pad_token_id)
            for j in range(len(chunk)):
                new = gen[j][enc["input_ids"].shape[1]:]
                outs.append(self.tok.decode(new, skip_special_tokens=True))
        return outs

    def train_lora(self, pairs, epochs=2, lr=1e-5, bs=None, max_len=640):
        """pairs = [(question, solution_text)]; train on the assistant span only.
        Low lr + warmup/cosine + grad clipping: high lr on few examples destabilises
        the student (it then DEGRADES rather than improves)."""
        if not pairs: return
        bs = bs or self.train_bs
        self.model.train()
        opt = torch.optim.AdamW([p for p in self.model.parameters() if p.requires_grad], lr=lr)
        n_steps = max(1, epochs * ((len(pairs) + bs - 1) // bs))
        try:
            from transformers import get_cosine_schedule_with_warmup
            sched = get_cosine_schedule_with_warmup(opt, int(0.1 * n_steps), n_steps)
        except Exception:
            sched = None
        examples = []
        for q, sol in pairs:
            prompt = self._prompt_ids(q)
            full = prompt + sol + self.tok.eos_token
            pids = self.tok(prompt, add_special_tokens=False)["input_ids"]
            fids = self.tok(full, add_special_tokens=False)["input_ids"][:max_len]
            labels = [-100] * min(len(pids), len(fids)) + fids[len(pids):]
            labels = labels[:len(fids)]
            examples.append((fids, labels))
        for _ in range(epochs):
            for i in range(0, len(examples), bs):
                batch = examples[i:i + bs]
                mlen = max(len(f) for f, _ in batch)
                pad = self.tok.pad_token_id
                input_ids = torch.tensor([f + [pad] * (mlen - len(f)) for f, _ in batch]).to(self.device)
                labels = torch.tensor([l + [-100] * (mlen - len(l)) for _, l in batch]).to(self.device)
                attn = (input_ids != pad).long()
                out = self.model(input_ids=input_ids, attention_mask=attn, labels=labels)
                out.loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad], 1.0)
                opt.step(); opt.zero_grad()
                if sched: sched.step()

# ----------------------------------------------------------------------------
# One variant run
# ----------------------------------------------------------------------------
def evaluate(student, eval_set, backend, vmodel, tmodel, teacher_cache):
    qs = [q for q, _ in eval_set]; golds = [g for _, g in eval_set]
    sols = student.generate(qs)                                        # GPU
    n = len(eval_set)
    v_acc = pmap(lambda i: verifier_accept(backend, vmodel, qs[i], sols[i]), range(n))  # parallel API
    rej = [i for i in range(n) if not v_acc[i]]
    need = [qs[i] for i in rej if qs[i] not in teacher_cache]
    for q, t in zip(need, pmap(lambda q: teacher_solve(backend, tmodel, q), need)):
        teacher_cache[q] = t
    dash_rej = dict(zip(rej, pmap(
        lambda i: verifier_accept(backend, vmodel, qs[i], teacher_cache[qs[i]]), rej)))
    wrong = acc = blind = deliver_wrong = dash_wrong = 0
    for i in range(n):
        s_ok = (extract_final_number(sols[i]) == golds[i])
        if not s_ok: wrong += 1
        if v_acc[i]: acc += 1
        if (not s_ok) and v_acc[i]: blind += 1
        if v_acc[i]:
            deliver_ok, dash_ok = s_ok, True    # verifier accepted -> dashboard sees "correct"
        else:
            deliver_ok = (extract_final_number(teacher_cache[qs[i]]) == golds[i])
            dash_ok = dash_rej[i]               # dashboard re-judges the delivered teacher solution
        if not deliver_ok: deliver_wrong += 1
        if not dash_ok: dash_wrong += 1
    return dict(q=wrong / n, p_reject=(n - acc) / n, beta=blind / max(wrong, 1),
                eps=deliver_wrong / n, dash=dash_wrong / n)

def run_variant(variant, args, train, eval_set, device):
    student = Student(args.student, device)
    student.gen_bs, student.train_bs = args.gen_bs, args.train_bs
    teacher_cache = {}
    accumulated = []          # (question, teacher_solution) pairs for corrective
    hist = []
    rng = list(range(len(train)))
    import random; random.Random(args.seed).shuffle(rng)
    ptr = 0
    for t in range(args.rounds + 1):
        # (re)build student
        if variant == "frozen":
            if student.model is None: student.load_base_only()
        else:
            student.fresh_lora(r=args.lora_r)
            if accumulated: student.train_lora(accumulated, epochs=args.epochs, lr=args.lr)
        m = evaluate(student, eval_set, args.backend, args.verifier_model, args.teacher_model, teacher_cache)
        m["t"] = t; hist.append(m)
        print(f"[{variant}] round {t}: q={m['q']:.3f} p_rej={m['p_reject']:.3f} "
              f"beta={m['beta']:.3f} eps={m['eps']:.3f} dash={m['dash']:.3f}", flush=True)
        if t == args.rounds: break
        if variant == "frozen": continue   # never retrains -> no batch step needed
        # ---- loop step: student attempts a fresh batch, verifier judges, teacher corrects rejects
        batch = [train[rng[(ptr + i) % len(rng)]] for i in range(args.n_batch)]; ptr += args.n_batch
        bqs = [q for q, _ in batch]
        sols = student.generate(bqs)                                   # GPU
        v = pmap(lambda i: verifier_accept(args.backend, args.verifier_model, bqs[i], sols[i]), range(len(batch)))
        rej = [i for i in range(len(batch)) if not v[i]]
        need = [bqs[i] for i in rej if bqs[i] not in teacher_cache]
        for q, t in zip(need, pmap(lambda q: teacher_solve(args.backend, args.teacher_model, q), need)):
            teacher_cache[q] = t
        for i in rej:
            accumulated.append((bqs[i], teacher_cache[bqs[i]]))        # corrective training target
    return hist

# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--student", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--backend", default="openai", choices=["openai", "ollama"])
    ap.add_argument("--verifier-model", default="gpt-4o-mini")
    ap.add_argument("--teacher-model", default="gpt-4o-mini")
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--n-eval", type=int, default=150)
    ap.add_argument("--n-batch", type=int, default=96)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-5, help="LoRA lr (lower = safer against degradation)")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--teacher-file", default=None,
                    help="JSON {question: solution} of pre-generated (e.g. same-family) teacher solutions")
    ap.add_argument("--gen-bs", type=int, default=8, help="generation batch (GPU: 32-64)")
    ap.add_argument("--train-bs", type=int, default=2, help="LoRA train batch (GPU: 8-16)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--variants", default="corrective,frozen")
    ap.add_argument("--out", default=str(HERE / "results_real.json"))
    args = ap.parse_args()
    if args.smoke:
        args.rounds, args.n_eval, args.n_batch = 2, 6, 6
        args.student = "Qwen/Qwen2.5-0.5B-Instruct"

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device={device} student={args.student} verifier={args.verifier_model} "
          f"gen_bs={args.gen_bs} train_bs={args.train_bs}", flush=True)
    train, test = load_gsm8k()
    eval_set = test[:args.n_eval]
    if args.teacher_file:
        CACHED_TEACHER.update(json.loads(Path(args.teacher_file).read_text()))
        train = [(q, g) for (q, g) in train if q in CACHED_TEACHER]   # only draw covered problems
        print(f"cached teacher: {len(CACHED_TEACHER)} solutions; train pool -> {len(train)}", flush=True)
    print(f"gsm8k: {len(train)} train, evaluating on {len(eval_set)} test", flush=True)

    results = {"config": vars(args), "runs": {}}
    for v in args.variants.split(","):
        t0 = time.time()
        results["runs"][v] = run_variant(v, args, train, eval_set, device)
        print(f"[{v}] done in {(time.time()-t0)/60:.1f} min", flush=True)
        Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nsaved -> {args.out}")
    try:
        plot(results, HERE / "exp_real_scissors.png")
    except Exception as e:
        print("plot skipped:", e)

def plot(results, path):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    corr = results["runs"].get("corrective", [])
    if not corr: return
    x = [d["t"] for d in corr]
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    ax.plot(x, [d["q"] for d in corr], ":", color="#777", label="raw error $q_t$")
    ax.plot(x, [d["dash"] for d in corr], "-", color="#2e7d32", lw=2, label="verifier-estimated (dashboard)")
    ax.plot(x, [d["eps"] for d in corr], "-", color="#c0392b", lw=2, label="gold user-facing $\\varepsilon_t$")
    ax.fill_between(x, [d["dash"] for d in corr], [d["eps"] for d in corr], color="#777", alpha=0.15)
    ax.set_xlabel("round $t$"); ax.set_ylabel("error rate"); ax.set_ylim(bottom=0)
    ax.set_title("Phase-0 real-LLM (GSM8K, corrective)", fontsize=10)
    ax.legend(frameon=False, fontsize=8.5)
    fig.tight_layout(); fig.savefig(path, dpi=130)
    print("plot ->", path)

if __name__ == "__main__":
    main()
