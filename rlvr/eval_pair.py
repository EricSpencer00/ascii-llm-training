"""Paired per-task eval of the base model vs the trained adapter.

The aggregate held-out means (base 0.0969, trained 0.1096) are a 13% relative
difference on 40 tasks with no variance estimate, which is not enough to call
a win. Same tasks, same decoding, per-task rewards, paired sign test.
"""
from __future__ import annotations

import argparse, json, time
from pathlib import Path

from rlvr.tasks import generate_tasks
from rlvr.train_grpo import _generate
from rlvr.reward import score_one

LOG_DIR = Path(__file__).parent / "logs"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--adapter", default=str(Path(__file__).parent / "checkpoints" / "adapter"))
    p.add_argument("--seed", type=int, default=10000)
    p.add_argument("--n-tasks", type=int, default=40)
    p.add_argument("--cols", type=int, default=24)
    p.add_argument("--rows", type=int, default=12)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(args.model).to(args.device).eval()
    tuned = PeftModel.from_pretrained(
        AutoModelForCausalLM.from_pretrained(args.model), args.adapter
    ).to(args.device).eval()

    tasks = generate_tasks(args.n_tasks, seed=args.seed, cols=args.cols, rows=args.rows)
    rows = []
    for t in tasks:
        rec = {"task_id": t.task_id, "description": t.description}
        for name, m in (("base", base), ("trained", tuned)):
            text = _generate(m, tok, t.prompt, t.cols, t.rows, t.charset, args.device, True)
            rec[name] = score_one(text, t.image, t.cols, t.rows, t.charset)["reward"]
        rec["oracle"] = score_one(t.oracle_text(), t.image, t.cols, t.rows, t.charset)["reward"]
        rows.append(rec)

    wins = sum(1 for r in rows if r["trained"] > r["base"])
    losses = sum(1 for r in rows if r["trained"] < r["base"])
    n = len(rows)
    mean = lambda k: sum(r[k] for r in rows) / n
    # two-sided sign test on the paired wins/losses, exact binomial
    from math import comb
    k, m = min(wins, losses), wins + losses
    pval = min(1.0, 2 * sum(comb(m, i) for i in range(k + 1)) / (2 ** m)) if m else 1.0

    summary = {
        "n_tasks": n,
        "base_mean": mean("base"),
        "trained_mean": mean("trained"),
        "oracle_mean": mean("oracle"),
        "trained_wins": wins,
        "trained_losses": losses,
        "ties": n - wins - losses,
        "sign_test_p": pval,
    }
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    out = LOG_DIR / f"pair_{int(time.time())}.json"
    out.write_text(json.dumps({"summary": summary, "per_task": rows}, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"log: {out}")


if __name__ == "__main__":
    main()
