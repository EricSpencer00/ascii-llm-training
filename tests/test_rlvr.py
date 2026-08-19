import pytest

from rlvr.tasks import DEFAULT_ASCII_CHARSET, generate_tasks
from rlvr.reward import format_reward, reward_fn, score_one
from asciiart.verify import check_constraints


def test_generate_tasks_deterministic():
    a = generate_tasks(6, seed=42, cols=10, rows=5)
    b = generate_tasks(6, seed=42, cols=10, rows=5)
    assert [t.description for t in a] == [t.description for t in b]
    assert [t.prompt for t in a] == [t.prompt for t in b]
    assert len(a) == 6


def test_task_prompt_contains_shape_and_grid():
    tasks = generate_tasks(3, seed=1, cols=16, rows=8)
    for t in tasks:
        assert "16x8" in t.prompt
        assert t.description in t.prompt
        assert t.charset in t.prompt


def test_oracle_text_is_well_formed_grid():
    tasks = generate_tasks(2, seed=0, cols=12, rows=6, charset=DEFAULT_ASCII_CHARSET)
    for t in tasks:
        oracle = t.oracle_text()
        c = check_constraints(oracle, t.cols, t.rows, t.charset)
        assert c["ok"], c


def test_format_reward_perfect_and_broken():
    tasks = generate_tasks(1, seed=0, cols=8, rows=4, charset=DEFAULT_ASCII_CHARSET)
    t = tasks[0]
    good = t.oracle_text()
    bad = "not a grid"
    rewards = format_reward(
        [good, bad],
        target_image=[t.image, t.image],
        cols=[t.cols, t.cols],
        rows=[t.rows, t.rows],
        charset=[t.charset, t.charset],
    )
    assert rewards[0] == 1.0
    assert rewards[1] == 0.0


def test_reward_fn_matches_verify_score():
    tasks = generate_tasks(1, seed=0, cols=8, rows=4, charset=DEFAULT_ASCII_CHARSET)
    t = tasks[0]
    text = t.oracle_text()
    [r] = reward_fn(
        [text], target_image=[t.image], cols=[t.cols], rows=[t.rows], charset=[t.charset]
    )
    direct = score_one(text, t.image, t.cols, t.rows, t.charset)
    assert r == pytest.approx(direct["reward"])


def test_reward_fn_zero_on_constraint_violation():
    tasks = generate_tasks(1, seed=0, cols=8, rows=4, charset=DEFAULT_ASCII_CHARSET)
    t = tasks[0]
    [r] = reward_fn(
        ["short\n"], target_image=[t.image], cols=[t.cols], rows=[t.rows], charset=[t.charset]
    )
    assert r == 0.0


def test_completion_text_normalizes_chat_format():
    from rlvr.reward import _completion_text

    assert _completion_text("plain") == "plain"
    assert _completion_text([{"role": "assistant", "content": "hi"}]) == "hi"


# --- constrained decoding -------------------------------------------------

transformers = pytest.importorskip("transformers")


@pytest.fixture(scope="module")
def tiny_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")


def test_build_allowed_token_map_single_chars_only(tiny_tokenizer):
    from rlvr.constrained import build_allowed_token_map

    charset = " .:-=+*#%@"
    table = build_allowed_token_map(tiny_tokenizer, charset)
    assert table["char_token_ids"], "expected at least one single-char token"
    for tid, ch in table["char_token_ids"].items():
        assert len(ch) == 1
        assert ch in charset
    # every mapped id really does round-trip through decode() to that char
    for tid, ch in list(table["char_token_ids"].items())[:20]:
        assert tiny_tokenizer.decode([tid]) == ch


def test_grid_processor_forces_newline_at_row_end(tiny_tokenizer):
    import torch

    from rlvr.constrained import GridConstraintLogitsProcessor

    charset = "ab"
    cols, rows = 3, 2
    proc = GridConstraintLogitsProcessor(
        tokenizer=tiny_tokenizer,
        cols=cols,
        rows=rows,
        charset=charset,
        prompt_len=0,
        eos_token_id=tiny_tokenizer.eos_token_id,
    )
    # fabricate 3 already-generated "a" tokens (one full row) using any 3
    # ids mapped to a charset char, to check the processor forces a newline
    # afterward.
    a_id = next(tid for tid, ch in proc._table["char_token_ids"].items() if ch in charset)
    input_ids = torch.tensor([[a_id, a_id, a_id]])
    vocab_size = len(tiny_tokenizer)
    scores = torch.zeros(1, vocab_size)

    out = proc(input_ids, scores)
    finite_ids = set((out[0] > float("-inf")).nonzero().flatten().tolist())
    assert finite_ids == proc._table["newline_token_ids"]


def test_grid_processor_forces_eos_at_end_of_last_row(tiny_tokenizer):
    """No trailing newline after the final row: EOS follows directly once
    the last row's column count is reached, matching
    asciiart.verify.check_constraints (which expects exactly `rows` lines,
    not `rows` lines plus one trailing empty line)."""
    import torch

    from rlvr.constrained import GridConstraintLogitsProcessor

    charset = "a"
    cols, rows = 2, 1
    proc = GridConstraintLogitsProcessor(
        tokenizer=tiny_tokenizer,
        cols=cols,
        rows=rows,
        charset=charset,
        prompt_len=0,
        eos_token_id=tiny_tokenizer.eos_token_id,
    )
    a_id = next(tid for tid, ch in proc._table["char_token_ids"].items() if ch in charset)
    input_ids = torch.tensor([[a_id, a_id]])
    vocab_size = len(tiny_tokenizer)
    scores = torch.zeros(1, vocab_size)

    out = proc(input_ids, scores)
    finite_ids = set((out[0] > float("-inf")).nonzero().flatten().tolist())
    assert finite_ids == {tiny_tokenizer.eos_token_id}


def test_grid_processor_forces_eos_after_eos_token_seen(tiny_tokenizer):
    import torch

    from rlvr.constrained import GridConstraintLogitsProcessor

    charset = "a"
    cols, rows = 2, 1
    proc = GridConstraintLogitsProcessor(
        tokenizer=tiny_tokenizer,
        cols=cols,
        rows=rows,
        charset=charset,
        prompt_len=0,
        eos_token_id=tiny_tokenizer.eos_token_id,
    )
    a_id = next(tid for tid, ch in proc._table["char_token_ids"].items() if ch in charset)
    input_ids = torch.tensor([[a_id, a_id, tiny_tokenizer.eos_token_id]])
    vocab_size = len(tiny_tokenizer)
    scores = torch.zeros(1, vocab_size)

    out = proc(input_ids, scores)
    finite_ids = set((out[0] > float("-inf")).nonzero().flatten().tolist())
    assert finite_ids == {tiny_tokenizer.eos_token_id}
