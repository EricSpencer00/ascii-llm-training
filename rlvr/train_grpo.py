"""GRPO training entrypoint (roadmap P3): RLVR against `asciiart.verify`'s
verifier, on a small open model with LoRA, using constrained decoding
(roadmap P2) so the policy never has to learn width/charset constraints.

Usage (see scripts/rlvr_sophia.pbs for the full sophia invocation):

    python -m rlvr.train_grpo --eval-only --cols 24 --rows 12
    python -m rlvr.train_grpo --eval-only --eval-mode decoding --cols 24 --rows 12
    python -m rlvr.train_grpo --steps 50 --cols 24 --rows 12 --num-generations 8

`--eval-only` runs a *paired* held-out eval: the base model and the trained
adapter, on the same tasks, in the same loop, with the same constrained
decoding, scored per task and compared with an exact sign test. Every
model-vs-model number this repo reports comes from that one path.

Pairing is the default because the unpaired version gave a wrong answer. The
earlier "+13% held-out" gain (0.1096 vs 0.0969) came from two separate
stochastic decoding runs. Run the same models on the same tasks and the
difference disappears: 20 wins, 19 losses, 1 tie, sign test p = 1.00 (job
175888). Sampling variance between runs is larger than the claimed effect.

`--eval-mode decoding` is the one eval that does not compare two models: it
samples the base model with and without the constrained-decoding logits
processor and reports format pass-rate. That is the P2 acceptance check (0
constraint violations under constrained decoding) and it needs no adapter.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path

from rlvr.tasks import DEFAULT_ASCII_CHARSET, generate_tasks
from rlvr.reward import format_reward, reward_fn, score_one

LOG_DIR = Path(__file__).parent / "logs"


def build_dataset(tasks):
    from datasets import Dataset

    rows = [
        {
            "prompt": t.prompt,
            "target_image": t.image,
            "cols": t.cols,
            "rows": t.rows,
            "charset": t.charset,
            "task_id": t.task_id,
        }
        for t in tasks
    ]
    return Dataset.from_list(rows)


def _generate(model, tokenizer, prompt, cols, rows, charset, device, constrained, max_new_tokens=None):
    import torch
    from transformers import LogitsProcessorList

    from rlvr.constrained import GridConstraintLogitsProcessor

    if max_new_tokens is None:
        max_new_tokens = cols * rows + rows + 8

    if getattr(tokenizer, "chat_template", None):
        messages = [{"role": "user", "content": prompt}]
        enc = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt",
            return_dict=True,
        )
        # transformers>=5 returns a BatchEncoding; older versions a tensor.
        input_ids = (enc["input_ids"] if hasattr(enc, "keys") else enc).to(device)
    else:
        # Fallback for base/tiny models with no chat template (used for
        # local CPU smoke tests; the sophia run uses an -Instruct model
        # with a real chat template).
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    prompt_len = input_ids.shape[1]

    processors = LogitsProcessorList()
    if constrained:
        processors.append(
            GridConstraintLogitsProcessor(
                tokenizer=tokenizer,
                cols=cols,
                rows=rows,
                charset=charset,
                prompt_len=prompt_len,
            )
        )

    with torch.no_grad():
        out = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=1.0,
            logits_processor=processors,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    completion_ids = out[0, prompt_len:]
    return tokenizer.decode(completion_ids, skip_special_tokens=True)


def _sign_test(wins, losses) -> float:
    """Two-sided exact sign test on paired wins/losses."""
    from math import comb

    k, m = min(wins, losses), wins + losses
    if not m:
        return 1.0
    return min(1.0, 2 * sum(comb(m, i) for i in range(k + 1)) / (2 ** m))


def _paired_summary(per_task, arms=("base", "trained")) -> dict:
    """Aggregate the per-task rows of a paired eval into means, format
    pass-rates, wins/losses/ties and an exact sign test. Pure function, so
    the statistics are testable without a model."""
    base, trained = arms
    n = len(per_task)
    wins = sum(1 for r in per_task if r[trained] > r[base])
    losses = sum(1 for r in per_task if r[trained] < r[base])
    mean = lambda key: (sum(r[key] for r in per_task) / n) if n else 0.0
    return {
        "n_tasks": n,
        f"{base}_mean": mean(base),
        f"{trained}_mean": mean(trained),
        "oracle_mean": mean("oracle"),
        f"{base}_format_pass_rate": mean(f"{base}_format_ok"),
        f"{trained}_format_pass_rate": mean(f"{trained}_format_ok"),
        f"{trained}_wins": wins,
        f"{trained}_losses": losses,
        "ties": n - wins - losses,
        "sign_test_p": _sign_test(wins, losses),
    }


def _paired_eval(arms, tokenizer, tasks, device, constrained=True) -> dict:
    """Score every arm on the same tasks, in the same loop, with the same
    decoding, and return per-task rewards plus a paired sign test.

    `arms` is a list of (name, model, context) where `context` is a no-arg
    context manager factory that selects that arm on the model -- for a
    LoRA policy the base arm is `peft_model.disable_adapter`, so the two
    arms share every weight except the adapter and nothing else can drift
    between them.

    Two means from two separate runs are not a comparison: decoding is
    stochastic and the between-run spread is larger than the effect. Only
    the paired numbers below are reportable.
    """
    per_task = []
    for t in tasks:
        rec = {"task_id": t.task_id, "description": t.description}
        for name, model, context in arms:
            with context():
                text = _generate(
                    model, tokenizer, t.prompt, t.cols, t.rows, t.charset, device, constrained
                )
            r = score_one(text, t.image, t.cols, t.rows, t.charset)
            rec[name] = r["reward"]
            rec[f"{name}_format_ok"] = 1.0 if r["constraints"]["ok"] else 0.0
        rec["oracle"] = score_one(t.oracle_text(), t.image, t.cols, t.rows, t.charset)["reward"]
        per_task.append(rec)

    summary = _paired_summary(per_task, arms=[a[0] for a in arms])
    summary["constrained"] = constrained
    return {"summary": summary, "per_task": per_task}


def _lora_arms(model):
    """The two arms of a paired eval on a LoRA policy: adapter off, adapter
    on. Same weights, same tokenizer, same decoding."""
    return [("base", model, model.disable_adapter), ("trained", model, nullcontext)]


def _heldout_tasks(args):
    """The held-out task set: the training seed offset by `--heldout-offset`,
    so `--eval-only` scores exactly the tasks the post-training eval scores."""
    return generate_tasks(
        args.n_tasks,
        seed=args.seed + args.heldout_offset,
        cols=args.cols,
        rows=args.rows,
        charset=args.charset,
    )


def run_eval(args):
    """Paired held-out eval of the base model against the trained adapter.
    Default for `--eval-only`; see the module docstring for why."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    adapter = args.adapter or str(Path(args.output_dir) / "adapter")
    if not Path(adapter).exists():
        sys.exit(
            f"paired eval needs a trained adapter, none found at {adapter}. "
            "Pass --adapter, or run --eval-mode decoding for the base-only "
            "constrained-decoding check."
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = PeftModel.from_pretrained(
        AutoModelForCausalLM.from_pretrained(args.model), adapter
    ).to(args.device)
    model.eval()

    tasks = _heldout_tasks(args)
    out = _paired_eval(_lora_arms(model), tokenizer, tasks, args.device)
    out["summary"].update(
        {"model": args.model, "adapter": adapter, "heldout_seed": args.seed + args.heldout_offset}
    )

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"pair_{int(time.time())}.json"
    log_path.write_text(json.dumps(out, indent=2))

    print(json.dumps(out["summary"], indent=2))
    print(f"log: {log_path}")
    return out["summary"]


def run_decoding_check(args):
    """P2 acceptance check: the base model sampled with and without the grid
    constraint processor, format pass-rate for each. One model, so there is
    nothing to pair -- for base vs trained use the default paired eval."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = args.device
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device)
    model.eval()

    tasks = generate_tasks(
        args.n_tasks, seed=args.seed, cols=args.cols, rows=args.rows, charset=args.charset
    )

    results = {"constrained": [], "unconstrained": []}
    for constrained in (True, False):
        for t in tasks:
            text = _generate(
                model, tokenizer, t.prompt, t.cols, t.rows, t.charset, device, constrained
            )
            r = score_one(text, t.image, t.cols, t.rows, t.charset)
            results["constrained" if constrained else "unconstrained"].append(
                {
                    "task_id": t.task_id,
                    "format_ok": r["constraints"]["ok"],
                    "reward": r["reward"],
                }
            )

    oracle_rewards = []
    for t in tasks:
        oracle_text = t.oracle_text()
        r = score_one(oracle_text, t.image, t.cols, t.rows, t.charset)
        oracle_rewards.append(r["reward"])

    summary = {}
    for mode, rows_ in results.items():
        n = len(rows_)
        pass_rate = sum(1 for r in rows_ if r["format_ok"]) / n if n else 0.0
        mean_reward = sum(r["reward"] for r in rows_) / n if n else 0.0
        summary[mode] = {"format_pass_rate": pass_rate, "mean_reward": mean_reward}
    summary["oracle_mean_reward"] = sum(oracle_rewards) / len(oracle_rewards) if oracle_rewards else 0.0

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"eval_{int(time.time())}.jsonl"
    with open(log_path, "w") as f:
        f.write(json.dumps(summary) + "\n")

    print(json.dumps(summary, indent=2))
    print(f"log: {log_path}")
    return summary


class RewardLogger:
    """Wraps reward_fn/format_reward to also append per-call jsonl stats
    (mean reward/ssim/edge/format across the batch) -- TRL calls reward
    functions once per generation batch during training, so this doubles as
    a per-step log."""

    __name__ = "ascii_verifier_reward"  # TRL names reward columns from this

    def __init__(self, log_path):
        self.log_path = log_path
        self.step = 0

    def __call__(self, completions, **kwargs):
        n = len(completions)
        images = kwargs.get("target_image")
        cols = kwargs.get("cols", 24)
        rows = kwargs.get("rows", 12)
        charset = kwargs.get("charset")
        cols = cols if isinstance(cols, (list, tuple)) else [cols] * n
        rows = rows if isinstance(rows, (list, tuple)) else [rows] * n
        charset = charset if isinstance(charset, (list, tuple)) else [charset] * n

        from rlvr.reward import _completion_text

        rewards, ssims, edges, fmts = [], [], [], []
        for i in range(n):
            text = _completion_text(completions[i])
            r = score_one(text, images[i], cols[i], rows[i], charset[i])
            rewards.append(r["reward"])
            ssims.append(r["ssim"])
            edges.append(r["edge_score"])
            fmts.append(1.0 if r["constraints"]["ok"] else 0.0)

        record = {
            "step": self.step,
            "mean_reward": sum(rewards) / n if n else 0.0,
            "mean_ssim": sum(ssims) / n if n else 0.0,
            "mean_edge": sum(edges) / n if n else 0.0,
            "format_pass_rate": sum(fmts) / n if n else 0.0,
        }
        self.step += 1
        with open(self.log_path, "a") as f:
            f.write(json.dumps(record) + "\n")

        return rewards



def _patch_generate_with_constraints(trainer, args):
    """Force TRL's rollout sampling through the grid constraint processor.

    TRL calls `unwrapped_model.generate(**inputs, generation_config=...)` and
    exposes no hook for a `LogitsProcessor`, so wrap the bound `generate` of
    the underlying model. Without this, every rollout is unconstrained, every
    completion fails `check_constraints`, and the verifier reward is
    identically 0 -- which is exactly what job 175620 showed (50 steps,
    reward 0.0, grad_norm 0.0, no learning signal at all).
    """
    from transformers import LogitsProcessorList

    from rlvr.constrained import GridConstraintLogitsProcessor

    model = trainer.model
    tokenizer = trainer.processing_class
    original_generate = model.generate

    def generate(*a, **kw):
        input_ids = kw.get("input_ids")
        if input_ids is None and a:
            input_ids = a[0]
        if input_ids is not None:
            processor = GridConstraintLogitsProcessor(
                tokenizer=tokenizer,
                cols=args.cols,
                rows=args.rows,
                charset=args.charset,
                prompt_len=int(input_ids.shape[1]),
            )
            existing = kw.get("logits_processor") or LogitsProcessorList()
            existing.append(processor)
            kw["logits_processor"] = existing
        return original_generate(*a, **kw)

    model.generate = generate
    print("rollout generation is constrained (GridConstraintLogitsProcessor attached)")


def run_train(args):
    from peft import LoraConfig
    from trl import GRPOConfig, GRPOTrainer

    tasks = generate_tasks(
        args.n_tasks, seed=args.seed, cols=args.cols, rows=args.rows, charset=args.charset
    )
    dataset = build_dataset(tasks)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"train_{int(time.time())}.jsonl"
    logger = RewardLogger(log_path)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )

    grpo_config = GRPOConfig(
        output_dir=args.output_dir,
        max_steps=args.steps,
        num_generations=args.num_generations,
        per_device_train_batch_size=args.num_generations,
        max_completion_length=args.cols * args.rows + args.rows + 8,
        temperature=1.0,
        logging_steps=1,
        save_strategy="no",  # adapter is saved explicitly after training instead
        report_to=[],
    )

    trainer = GRPOTrainer(
        model=args.model,
        reward_funcs=logger,
        args=grpo_config,
        train_dataset=dataset,
        peft_config=lora_config,
    )

    if args.constrained_rollout:
        _patch_generate_with_constraints(trainer, args)

    trainer.train()

    # Persist the adapter and run a held-out eval with it. Run 175665 trained
    # fine but saved nothing, so its gain could only be read off training-time
    # reward -- there was no artifact left to sample or to score on tasks the
    # policy had not been trained on.
    adapter_dir = Path(args.output_dir) / "adapter"
    trainer.model.save_pretrained(str(adapter_dir))
    print(f"adapter saved: {adapter_dir}")

    # Paired held-out eval, base vs trained, in one loop on one task set.
    # A trained-only mean here was the source of the void "+13%" claim: it
    # had to be compared against a base number from a separate run, and that
    # comparison did not survive pairing (job 175888, p = 1.00).
    eval_tasks = _heldout_tasks(args)  # disjoint from the training tasks
    model = trainer.model
    model.eval()
    post = _paired_eval(
        _lora_arms(model), trainer.processing_class, eval_tasks, args.device
    )
    model.train()
    post["summary"]["heldout_seed"] = args.seed + args.heldout_offset
    post_path = LOG_DIR / f"posteval_{int(time.time())}.json"
    post_path.write_text(json.dumps(post, indent=2))
    print("post-training paired held-out eval:", json.dumps(post["summary"]))
    print(f"training log: {log_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--cols", type=int, default=24)
    p.add_argument("--rows", type=int, default=12)
    p.add_argument("--charset", default=DEFAULT_ASCII_CHARSET)
    p.add_argument("--num-generations", type=int, default=8)
    p.add_argument("--n-tasks", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu", help="cpu or cuda; pass --device cuda on GPU nodes")
    p.add_argument("--output-dir", default=str(Path(__file__).parent / "checkpoints"))
    p.add_argument("--eval-only", action="store_true")
    p.add_argument(
        "--eval-mode",
        choices=("paired", "decoding"),
        default="paired",
        help="paired: base vs trained adapter on the same held-out tasks with the "
        "same decoding (default; the only reportable model-vs-model comparison). "
        "decoding: base model with and without constrained decoding (P2 check).",
    )
    p.add_argument(
        "--adapter",
        default=None,
        help="adapter for the trained arm of the paired eval (default: <output-dir>/adapter)",
    )
    p.add_argument(
        "--heldout-offset",
        type=int,
        default=10_000,
        help="held-out task seed = --seed + this; the same offset the post-training "
        "eval uses, so --eval-only scores the same tasks",
    )
    p.add_argument(
        "--constrained-rollout",
        action="store_true",
        default=True,
        help="sample rollouts through the grid constraint processor (default on; "
        "without it the verifier reward is identically zero)",
    )
    p.add_argument("--no-constrained-rollout", dest="constrained_rollout", action="store_false")
    args = p.parse_args()

    if args.eval_only and args.eval_mode == "decoding":
        run_decoding_check(args)
    elif args.eval_only:
        run_eval(args)
    else:
        run_train(args)


if __name__ == "__main__":
    main()
