"""RLVR verifier: hard constraint checks + a checkable image-similarity reward.

`check_constraints(text, cols, rows, charset)` -> all-or-nothing hard
predicates (exact width, exact row count, charset membership, no trailing
whitespace on any line).

`score(text, target_img, font=None, cols=80, rows=None)` -> dict with
`ssim` (windowed/per-cell SSIM between `rasterize(text)` and the target
image resized to the same pixel grid), `edge_score` (Sobel-based edge-F1
between the two), and `reward = 0` if constraints fail else
`w1*ssim + w2*edge_score`.

The edge term exists specifically to close a reward hack: a uniform
mid-density fill can score close to a real render on plain SSIM (both are
locally smooth against a mid-gray target patch) but carries no edges, so
it scores much lower on edge_score. See `demonstrate_hack()`.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from asciiart.font import Font, default_font
from asciiart.render import _to_gray_array, _sobel, rasterize

W_SSIM = 0.6
W_EDGE = 0.4


def check_constraints(text: str, cols: int, rows: int, charset: str | None = None) -> dict:
    """Hard, all-or-nothing predicates on the raw ASCII text.

    Returns a dict with a boolean per predicate plus an aggregate `ok`.
    """
    lines = text.split("\n")

    row_count_ok = len(lines) == rows
    width_ok = row_count_ok and all(len(line) == cols for line in lines)
    trailing_ws_ok = all(line == line.rstrip() or line == "" for line in lines) if not width_ok else all(
        not line.endswith(" ") or len(line.rstrip(" ")) == len(line) for line in lines
    )
    # Simpler, unambiguous trailing-whitespace check: no line may end in a
    # space unless every character in it is a space (a deliberately blank
    # row is fine; a content row padded with trailing spaces is not, since
    # exact-width already requires padding via non-space-terminated content
    # or full-width glyphs).
    trailing_ws_ok = all((line.strip(" ") == "" or not line.endswith(" ")) for line in lines)

    allowed = set(charset) if charset is not None else None
    charset_ok = True
    if allowed is not None:
        for line in lines:
            for ch in line:
                if ch not in allowed:
                    charset_ok = False
                    break
            if not charset_ok:
                break

    ok = row_count_ok and width_ok and trailing_ws_ok and charset_ok
    return {
        "row_count_ok": row_count_ok,
        "width_ok": width_ok,
        "trailing_ws_ok": trailing_ws_ok,
        "charset_ok": charset_ok,
        "ok": ok,
    }


def _resize_gray(gray: np.ndarray, h: int, w: int) -> np.ndarray:
    im = Image.fromarray((np.clip(gray, 0, 1) * 255).astype(np.uint8))
    im = im.resize((w, h), Image.BILINEAR)
    return np.asarray(im, dtype=np.float32) / 255.0


def _ssim_windowed(x: np.ndarray, y: np.ndarray, win: int = 8) -> float:
    """Mean SSIM over non-overlapping win x win windows. Pure numpy, no
    scipy/scikit-image dependency."""
    h, w = x.shape
    L = 1.0
    c1 = (0.01 * L) ** 2
    c2 = (0.03 * L) ** 2
    vals = []
    for y0 in range(0, h - win + 1, win) or [0]:
        for x0 in range(0, w - win + 1, win) or [0]:
            wx = x[y0 : y0 + win, x0 : x0 + win]
            wy = y[y0 : y0 + win, x0 : x0 + win]
            if wx.size == 0:
                continue
            mu_x, mu_y = wx.mean(), wy.mean()
            var_x, var_y = wx.var(), wy.var()
            cov_xy = ((wx - mu_x) * (wy - mu_y)).mean()
            num = (2 * mu_x * mu_y + c1) * (2 * cov_xy + c2)
            den = (mu_x ** 2 + mu_y ** 2 + c1) * (var_x + var_y + c2)
            vals.append(num / den)
    if not vals:
        return 0.0
    return float(np.mean(vals))


def _edge_f1(x: np.ndarray, y: np.ndarray, tol: int = 1) -> float:
    """Edge-F1 between two grayscale images: Sobel-magnitude threshold to a
    binary edge map, then precision/recall with a small dilation tolerance
    to allow for sub-pixel misalignment between rasterized glyph edges and
    the target's edges."""
    mx, _ = _sobel(x)
    my, _ = _sobel(y)

    def binarize(mag, abs_thresh: float = 0.15):
        # Absolute threshold (Sobel magnitude on [0,1]-range images, max
        # ~4) rather than a percentile of nonzero values: a percentile
        # threshold always marks ~25% of pixels as "edges" even in a
        # visually flat/uniform image (tiny numerical gradients still
        # populate the nonzero-value distribution), which lets a uniform
        # fill fake a nontrivial edge score by chance alignment.
        return mag >= abs_thresh

    ex, ey = binarize(mx), binarize(my)

    def dilate(mask, r):
        if r <= 0:
            return mask
        out = mask.copy()
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                shifted = np.zeros_like(mask)
                ys0, ys1 = max(0, dy), mask.shape[0] + min(0, dy)
                xs0, xs1 = max(0, dx), mask.shape[1] + min(0, dx)
                yd0, yd1 = max(0, -dy), mask.shape[0] - max(0, dy)
                xd0, xd1 = max(0, -dx), mask.shape[1] - max(0, dx)
                shifted[yd0:yd1, xd0:xd1] = mask[ys0:ys1, xs0:xs1]
                out |= shifted
        return out

    ex_d = dilate(ex, tol)
    ey_d = dilate(ey, tol)

    if not ex.any() and not ey.any():
        return 1.0
    if not ex.any() or not ey.any():
        return 0.0

    tp_precision = np.logical_and(ex, ey_d).sum()
    precision = tp_precision / max(ex.sum(), 1)
    tp_recall = np.logical_and(ey, ex_d).sum()
    recall = tp_recall / max(ey.sum(), 1)
    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


def score(
    text: str,
    target_img,
    font: Font | None = None,
    cols: int = 80,
    rows: int | None = None,
    charset: str | None = None,
    w1: float = W_SSIM,
    w2: float = W_EDGE,
) -> dict:
    """Score ASCII `text` against `target_img`.

    Returns dict: {constraints, ssim, edge_score, reward}.
    """
    font = font or default_font()
    lines = text.split("\n")
    if rows is None:
        rows = len(lines)

    constraints = check_constraints(text, cols, rows, charset)

    rendered = rasterize(text, font=font)
    target_gray = _to_gray_array(target_img)
    target_resized = _resize_gray(target_gray, rendered.shape[0], rendered.shape[1])

    win = max(4, min(font.cell_h, font.cell_w))
    ssim = _ssim_windowed(rendered, target_resized, win=win)
    edge = _edge_f1(rendered, target_resized)

    if not constraints["ok"]:
        reward = 0.0
    else:
        reward = w1 * ssim + w2 * edge

    return {
        "constraints": constraints,
        "ssim": ssim,
        "edge_score": edge,
        "reward": reward,
    }


def demonstrate_hack(target_img=None, cols: int = 40, rows: int = 20, font: Font | None = None) -> dict:
    """Show that a uniform mid-density fill scores worse than a real
    structure-aware render under the combined reward, even when its plain
    SSIM is competitive.

    Returns a dict comparing the two, with the combined reward gap.
    """
    from asciiart.render import render as render_fn

    font = font or default_font()
    if target_img is None:
        # A simple synthetic image with real edge content: a filled circle
        # on a mid-gray background, so a uniform mid-density fill has a
        # plausible average-luminance match.
        # Convention matches the glyph atlas (font.py): bright pixel = ink,
        # dark pixel = background, i.e. white-on-black terminal rendering.
        # High contrast so the shape produces real edge content rather
        # than collapsing to a near-uniform tile under structure matching.
        size = 256
        yy, xx = np.mgrid[0:size, 0:size]
        cx, cy, r = size / 2, size / 2, size / 3
        img = np.full((size, size), 0.05, dtype=np.float32)
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2
        img[mask] = 0.95
        target_img = img

    real_text = render_fn(target_img, cols=cols, rows=rows, mode="edge", font=font)

    ramp = font.luminance_ramp()
    target_gray = _to_gray_array(target_img)
    mean_lum = float(target_gray.mean())
    covs = np.array([font.coverage[font.index_of(c)] for c in ramp])
    fill_char = ramp[int(np.argmin(np.abs(covs - mean_lum)))]
    hack_text = "\n".join(fill_char * cols for _ in range(rows))

    real_result = score(real_text, target_img, font=font, cols=cols, rows=rows)
    hack_result = score(hack_text, target_img, font=font, cols=cols, rows=rows)

    return {
        "real": real_result,
        "hack": hack_result,
        "ssim_gap": real_result["ssim"] - hack_result["ssim"],
        "reward_gap": real_result["reward"] - hack_result["reward"],
        "hack_beats_real_on_ssim_alone": hack_result["ssim"] >= real_result["ssim"],
        "real_beats_hack_on_reward": real_result["reward"] > hack_result["reward"],
    }
