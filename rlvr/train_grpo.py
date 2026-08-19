"""GRPO training entrypoint (roadmap P3): RLVR against `asciiart.verify`'s
verifier, on a small open model with LoRA, using constrained decoding
(roadmap P2) so the policy never has to learn width/charset constraints.

Usage (see scripts/rlvr_sophia.pbs for the full sophia invocation):

    python -m rlvr.train_grpo --eval-only --cols 24 --rows 12
    python -m rlvr.train_grpo --steps 50 --cols 24 --rows 12 --num-generations 8

`--eval-only` samples completions with and without the constrained-decoding
logits processor and reports format pass-rate + mean reward vs the
`asciiart.render` oracle -- no training, no gradient step, no LoRA. This is
the P2 acceptance check (0 constraint violations under constrained decoding)
plus a first look at the base model's zero-shot reward gap.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
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


def run_eval(args):
    import torch
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
        save_strategy="no",
        report_to=[],
    )

    trainer = GRPOTrainer(
        model=args.model,
        reward_funcs=logger,
        args=grpo_config,
        train_dataset=dataset,
        peft_config=lora_config,
    )
    trainer.train()
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
    args = p.parse_args()

    if args.eval_only:
        run_eval(args)
    else:
        run_train(args)


if __name__ == "__main__":
    main()
