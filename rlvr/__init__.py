"""rlvr: constrained decoding + RLVR (GRPO) training for ASCII-art
generation, built on top of the `asciiart` package's deterministic
converter and verifier.

Modules:
    constrained  -- LogitsProcessor enforcing exact grid shape / charset.
    tasks        -- synthetic target-image + prompt generator.
    reward       -- TRL-compatible reward functions wrapping asciiart.verify.
    train_grpo   -- GRPOTrainer entrypoint (Qwen2.5-0.5B-Instruct + LoRA).
"""
