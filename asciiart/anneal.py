"""Simulated annealing over the ASCII character grid.

Energy = -reward, where reward is the same SSIM + edge-F1 combination used
by `asciiart/verify.py`. This establishes a non-neural, compute-unbounded
upper baseline: what's the best achievable score when the only constraint
is search budget, so later learned methods have a real ceiling to be read
against.

Moves:
    - single-cell substitution, biased toward glyphs with similar measured
      ink coverage to the current cell (keeps moves local in luminance
      space so most proposals are plausible rather than wild).
    - row shift: shift a contiguous run of cells in one row left/right by
      one, refilling the vacated cell from the charset.
    - cell swap: swap two cells (anywhere in the grid).

The pixel raster is maintained incrementally: a proposed move only
touches one or two glyph-cell blocks of the rasterized image, so only
those blocks are rewritten before rescoring, instead of calling
`rasterize()` on the whole grid every step.
"""

from __future__ import annotations

import math
import random

import numpy as np

from asciiart.font import Font, default_font
from asciiart.render import _to_gray_array, _sobel, render as render_fn
from asciiart.verify import _ssim_windowed, _edge_f1, _resize_gray, W_SSIM, W_EDGE


def _grid_from_text(text: str) -> list[list[str]]:
    return [list(line) for line in text.split("\n")]


def _text_from_grid(grid: list[list[str]]) -> str:
    return "\n".join("".join(row) for row in grid)


def _paint_cell(pixels: np.ndarray, font: Font, r: int, c: int, ch: str) -> None:
    gh, gw = font.cell_h, font.cell_w
    if " " <= ch <= "~":
        pixels[r * gh : (r + 1) * gh, c * gw : (c + 1) * gw] = font.glyph(ch)
    else:
        pixels[r * gh : (r + 1) * gh, c * gw : (c + 1) * gw] = 0.0


def _full_reward(pixels: np.ndarray, target: np.ndarray, font: Font, w1: float, w2: float) -> float:
    win = max(4, min(font.cell_h, font.cell_w))
    ssim = _ssim_windowed(pixels, target, win=win)
    edge = _edge_f1(pixels, target)
    return w1 * ssim + w2 * edge


def _coverage_biased_charset(font: Font, charset: str, current_ch: str, k: int = 8) -> list[str]:
    """Return up to `k` charset glyphs nearest in measured coverage to
    `current_ch`, used to bias single-cell substitution proposals."""
    cur_cov = font.coverage[font.index_of(current_ch)] if " " <= current_ch <= "~" else 0.0
    scored = sorted(charset, key=lambda c: abs(font.coverage[font.index_of(c)] - cur_cov))
    return scored[:k] if len(scored) > k else list(charset)


def anneal(
    target_img,
    cols: int = 80,
    rows: int | None = None,
    steps: int = 2000,
    seed: int = 0,
    font: Font | None = None,
    charset: str | None = None,
    w1: float = W_SSIM,
    w2: float = W_EDGE,
    t0: float | None = None,
    alpha: float = 0.995,
    init_text: str | None = None,
):
    """Anneal a character grid against `target_img`.

    Returns (text, history) where history is a list of (step, energy,
    reward, temperature) tuples sampled periodically.
    """
    rng = random.Random(seed)
    font = font or default_font()
    charset = charset or font.luminance_ramp()

    if init_text is None:
        init_text = render_fn(target_img, cols=cols, rows=rows, mode="structure", font=font, charset=charset)
    grid = _grid_from_text(init_text)
    rows_n = len(grid)
    cols_n = max((len(r) for r in grid), default=cols)
    # normalize ragged rows (shouldn't happen from render_fn, but be safe)
    for r in grid:
        while len(r) < cols_n:
            r.append(" ")

    gh, gw = font.cell_h, font.cell_w
    pixels = np.zeros((rows_n * gh, cols_n * gw), dtype=np.float32)
    for r in range(rows_n):
        for c in range(cols_n):
            _paint_cell(pixels, font, r, c, grid[r][c])

    target_gray = _to_gray_array(target_img)
    target_resized = _resize_gray(target_gray, pixels.shape[0], pixels.shape[1])

    cur_reward = _full_reward(pixels, target_resized, font, w1, w2)
    cur_energy = -cur_reward

    if t0 is None:
        t0 = 0.05

    best_grid = [row[:] for row in grid]
    best_reward = cur_reward

    history = []
    T = t0
    log_every = max(1, steps // 50)

    for step in range(steps):
        move_kind = rng.random()
        r = rng.randrange(rows_n)
        c = rng.randrange(cols_n)
        undo = None

        if move_kind < 0.6:
            # single-cell substitution, coverage-biased
            cur_ch = grid[r][c]
            candidates = _coverage_biased_charset(font, charset, cur_ch)
            new_ch = rng.choice(candidates)
            if new_ch == cur_ch:
                continue
            old_ch = grid[r][c]
            grid[r][c] = new_ch
            _paint_cell(pixels, font, r, c, new_ch)
            undo = [(r, c, old_ch)]
        elif move_kind < 0.8:
            # row shift: shift row `r` by +/-1, wrap-refill the vacated end
            direction = rng.choice([-1, 1])
            old_row = grid[r][:]
            if direction == 1:
                new_row = [old_row[-1]] + old_row[:-1]
            else:
                new_row = old_row[1:] + [old_row[0]]
            grid[r] = new_row
            for cc in range(cols_n):
                if new_row[cc] != old_row[cc]:
                    _paint_cell(pixels, font, r, cc, new_row[cc])
            undo = [(r, cc, old_row[cc]) for cc in range(cols_n) if new_row[cc] != old_row[cc]]
        else:
            # cell swap
            r2 = rng.randrange(rows_n)
            c2 = rng.randrange(cols_n)
            if (r2, c2) == (r, c):
                continue
            ch_a, ch_b = grid[r][c], grid[r2][c2]
            if ch_a == ch_b:
                continue
            grid[r][c], grid[r2][c2] = ch_b, ch_a
            _paint_cell(pixels, font, r, c, ch_b)
            _paint_cell(pixels, font, r2, c2, ch_a)
            undo = [(r, c, ch_a), (r2, c2, ch_b)]

        new_reward = _full_reward(pixels, target_resized, font, w1, w2)
        new_energy = -new_reward
        delta = new_energy - cur_energy

        accept = delta <= 0 or rng.random() < math.exp(-delta / max(T, 1e-9))

        if accept:
            cur_energy = new_energy
            cur_reward = new_reward
            if new_reward > best_reward:
                best_reward = new_reward
                best_grid = [row[:] for row in grid]
        else:
            for (rr, cc, ch) in undo:
                grid[rr][cc] = ch
                _paint_cell(pixels, font, rr, cc, ch)

        T *= alpha

        if step % log_every == 0 or step == steps - 1:
            history.append((step, cur_energy, cur_reward, T))

    return _text_from_grid(best_grid), history
