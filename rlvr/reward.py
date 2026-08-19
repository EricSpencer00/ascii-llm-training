"""Reward functions for TRL's GRPOTrainer, wrapping `asciiart.verify.score`.

TRL calls a reward function as `reward_fn(prompts=..., completions=..., **kwargs)`
where `completions` is a list of strings (or list of chat-message lists,
already normalized to plain text here) and `**kwargs` carries every other
column of the training dataset, broadcast/aligned 1:1 with `completions`.
This module expects the dataset to carry a `target_image` column (PIL.Image
per row) plus `cols`/`rows`/`charset` columns (as produced by
`rlvr.tasks.generate_tasks`, via `rlvr.train_grpo.build_dataset`). The
column is deliberately not named `image`/`images` -- TRL's GRPOTrainer
special-cases those names to mean "this is multimodal (VLM) training" and
requires conversational-format prompts as a result, which this text-only
setup does not use.
"""

from __future__ import annotations

from asciiart.verify import score as verify_score


def _completion_text(completion) -> str:
    """TRL completions are either plain strings or a list of chat-message
    dicts (`[{"role": "assistant", "content": "..."}]`) depending on
    trainer config; normalize to plain text either way."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        parts = [m.get("content", "") for m in completion if isinstance(m, dict)]
        return "".join(parts)
    return str(completion)


def _broadcast(value, n):
    if isinstance(value, (list, tuple)):
        return value
    return [value] * n


def reward_fn(completions, **kwargs) -> list[float]:
    """Full RLVR reward: 0 if hard constraints fail, else
    `0.6*ssim + 0.4*edge_score`, per `asciiart.verify.score`."""
    n = len(completions)
    images = kwargs.get("target_image")
    cols = _broadcast(kwargs.get("cols", 24), n)
    rows = _broadcast(kwargs.get("rows", 12), n)
    charset = _broadcast(kwargs.get("charset"), n)
    if images is None:
        raise ValueError("reward_fn requires a 'target_image' column in the dataset")

    rewards = []
    for i in range(n):
        text = _completion_text(completions[i])
        result = verify_score(
            text,
            images[i],
            cols=cols[i],
            rows=rows[i],
            charset=charset[i],
        )
        rewards.append(float(result["reward"]))
    return rewards


def format_reward(completions, **kwargs) -> list[float]:
    """Constraints-only reward: 1.0 if the completion is a well-formed
    `cols x rows` grid over `charset`, else 0.0. Useful as a secondary/
    shaping reward, or standalone to measure format compliance without
    paying for a rasterize-and-compare pass."""
    from asciiart.verify import check_constraints

    n = len(completions)
    cols = _broadcast(kwargs.get("cols", 24), n)
    rows = _broadcast(kwargs.get("rows", 12), n)
    charset = _broadcast(kwargs.get("charset"), n)

    rewards = []
    for i in range(n):
        text = _completion_text(completions[i])
        result = check_constraints(text, cols[i], rows[i], charset[i])
        rewards.append(1.0 if result["ok"] else 0.0)
    return rewards


def score_one(text: str, image, cols: int, rows: int, charset: str) -> dict:
    """Non-batched convenience wrapper (used by eval-only mode / tests)."""
    return verify_score(text, image, cols=cols, rows=rows, charset=charset)
