"""Constrained decoding for grid-structured ASCII output (roadmap P2).

`GridConstraintLogitsProcessor` enforces, at decode time, the same hard
predicates `asciiart.verify.check_constraints` checks after the fact:

  - every row is exactly `cols` characters long
  - every character is drawn from `charset`
  - a newline token follows every row
  - EOS is forced immediately after the `rows`-th row

This removes the "ragged grid / wrong glyph" failure class structurally so
that RLVR (P3) never has to spend policy capacity learning it -- the reward
signal can be spent entirely on structure/likeness.

BPE caveat: a byte-pair tokenizer merges characters (and especially
whitespace runs) into multi-character tokens with no notion of column
position. This processor sidesteps that by restricting the *allowed*
vocabulary, at every decoding step, to tokens whose decoded string is
*exactly one character* long and a member of `charset` (plus a single
one-character newline token). Any token that would decode to more than one
character -- including common multi-space tokens like `"  "` or `"    "` --
is excluded from the allowed set entirely, even if part of it would have
been legal. This is a real limitation: it means every row-fill and
run-of-spaces must be spelled out one token at a time, which is slower and
burns more of the context window than an unconstrained model would use, but
it is what guarantees the exact-width invariant holds by construction. A
grammar-constrained decoder (e.g. outlines/llguidance, if installed) could
in principle allow safe multi-character tokens whose *prefix* still fits the
row, at the cost of a more complex per-step feasibility check; this
processor takes the simpler, strictly-single-char-token approach and
documents the trade-off rather than silently working around it.
"""

from __future__ import annotations

from dataclasses import dataclass, field


def build_allowed_token_map(tokenizer, charset: str) -> dict:
    """Build the id->char map for every single-character token in
    `tokenizer`'s vocabulary that decodes to a character in `charset`,
    plus (separately) the newline token id(s).

    Returns dict with keys:
        char_token_ids: {token_id: char} for charset members
        newline_token_ids: set[int] of token ids that decode to exactly "\n"
        vocab_size: int
    """
    vocab_size = len(tokenizer)
    char_token_ids: dict[int, str] = {}
    newline_token_ids: set[int] = set()
    charset_set = set(charset)

    for token_id in range(vocab_size):
        try:
            decoded = tokenizer.decode([token_id])
        except Exception:
            continue
        if len(decoded) != 1:
            continue
        if decoded == "\n":
            newline_token_ids.add(token_id)
        elif decoded in charset_set:
            char_token_ids[token_id] = decoded

    return {
        "char_token_ids": char_token_ids,
        "newline_token_ids": newline_token_ids,
        "vocab_size": vocab_size,
    }


@dataclass
class GridConstraintLogitsProcessor:
    """`transformers.LogitsProcessor`-compatible callable enforcing an exact
    `cols x rows` grid drawn from `charset`.

    Construct once per (tokenizer, cols, rows, charset); the allowed-id
    tables are built once in `__post_init__` and reused across the whole
    generation call. `prompt_len` must be the length (in tokens) of the
    prompt, i.e. where generation begins -- state is tracked by reading the
    generated suffix of `input_ids` back out on every step, so it must know
    where the prompt ends.
    """

    tokenizer: object
    cols: int
    rows: int
    charset: str
    prompt_len: int
    eos_token_id: int | None = None
    _table: dict = field(default=None, repr=False)

    def __post_init__(self):
        if self.eos_token_id is None:
            self.eos_token_id = self.tokenizer.eos_token_id
        self._table = build_allowed_token_map(self.tokenizer, self.charset)

    def _row_col(self, generated_ids) -> tuple[int, int]:
        """Return (rows_completed, col_in_current_row) from a sequence of
        already-generated token ids, using the id->char table. Any token id
        outside the table (shouldn't happen once masking is active, but can
        happen on the very first step before any masking took effect, or if
        this processor is used read-only for inspection) is treated as a
        single opaque character for row/col bookkeeping purposes.
        """
        char_map = self._table["char_token_ids"]
        newline_ids = self._table["newline_token_ids"]
        rows_completed = 0
        col = 0
        for tid in generated_ids:
            tid = int(tid)
            if tid in newline_ids:
                rows_completed += 1
                col = 0
            elif tid == self.eos_token_id:
                break
            else:
                col += 1
        return rows_completed, col

    def __call__(self, input_ids, scores):
        import torch

        batch = input_ids.shape[0]
        mask = torch.full_like(scores, float("-inf"))

        char_ids = list(self._table["char_token_ids"].keys())
        newline_ids = list(self._table["newline_token_ids"])

        for b in range(batch):
            generated = input_ids[b, self.prompt_len :].tolist()
            rows_completed, col = self._row_col(generated)

            if rows_completed >= self.rows:
                # Grid is complete: only EOS is legal.
                mask[b, self.eos_token_id] = scores[b, self.eos_token_id]
            elif col >= self.cols:
                if rows_completed == self.rows - 1:
                    # Last row just filled: no trailing newline, straight
                    # to EOS (matches asciiart.verify.check_constraints,
                    # which expects exactly `rows` lines with no trailing
                    # empty line from a final "\n").
                    mask[b, self.eos_token_id] = scores[b, self.eos_token_id]
                else:
                    # End of a non-final row: only a newline is legal.
                    for tid in newline_ids:
                        mask[b, tid] = scores[b, tid]
            else:
                # Mid-row: only charset characters are legal.
                for tid in char_ids:
                    mask[b, tid] = scores[b, tid]

        return mask
