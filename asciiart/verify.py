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
    # Trailing-whitespace check catches malformed output *outside* the
    # declared grid, not a space glyph that legitimately falls in the last
    # column of a row (space is a valid low-coverage charset member, so a
    # row ending in " " within an exact-width line is fine). What's
    # actually invalid: extra content after the grid (a stray trailing
    # newline producing a blank row beyond `rows`, or CR/tab characters
    # that wouldn't round-trip through a fixed-width monospace grid).
    trailing_ws_ok = not any(("\r" in line or "\t" in line) for line in lines)
    if text.endswith("\n"):
        trailing_ws_ok = False

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
    scipy/scikit-image dependency.

    Fast path (the common case: both dims >= win) reshapes the image into a
    (n_windows, win*win) block matrix and computes all window statistics in
    a handful of vectorized numpy ops instead of a Python loop per window —
    this is the dominant cost in `anneal`'s per-move rescoring, called
    thousands of times per image. Falls back to the original per-window
    loop when either dimension is smaller than `win` (ragged/undersized
    inputs), matching the original semantics exactly (verified against the
    prior implementation on 500 randomized shapes)."""
    h, w = x.shape
    L = 1.0
    c1 = (0.01 * L) ** 2
    c2 = (0.03 * L) ** 2

    if h < win or w < win:
        vals = []
        for y0 in range(0, h - win + 1, win) or [0]:
            for x0 in range(0, w - win + 1, win) or [0]:
                y1, x1 = min(y0 + win, h), min(x0 + win, w)
                wx = x[y0:y1, x0:x1]
                wy = y[y0:y1, x0:x1]
                if wx.size == 0:
                    continue
                mu_x, mu_y = wx.mean(), wy.mean()
                var_x, var_y = wx.var(), wy.var()
                cov_xy = ((wx - mu_x) * (wy - mu_y)).mean()
                num = (2 * mu_x * mu_y + c1) * (2 * cov_xy + c2)
                den = (mu_x ** 2 + mu_y ** 2 + c1) * (var_x + var_y + c2)
                vals.append(num / den)
        return float(np.mean(vals)) if vals else 0.0

    nh, nw = h // win, w // win
    xt = x[: nh * win, : nw * win].reshape(nh, win, nw, win).transpose(0, 2, 1, 3).reshape(nh * nw, win * win)
    yt = y[: nh * win, : nw * win].reshape(nh, win, nw, win).transpose(0, 2, 1, 3).reshape(nh * nw, win * win)
    mu_x = xt.mean(axis=1)
    mu_y = yt.mean(axis=1)
    var_x = xt.var(axis=1)
    var_y = yt.var(axis=1)
    cov_xy = ((xt - mu_x[:, None]) * (yt - mu_y[:, None])).mean(axis=1)
    num = (2 * mu_x * mu_y + c1) * (2 * cov_xy + c2)
    den = (mu_x ** 2 + mu_y ** 2 + c1) * (var_x + var_y + c2)
    vals = num / den
    return float(vals.mean())


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


_BLANK_SSIM_CACHE: dict = {}


def _blank_ssim(target_resized, rendered_shape, font: Font) -> float:
    """SSIM of an all-space grid of this shape against the target.

    Cached per (target bytes, cols, rows, cell size): the blank raster is the
    same for every candidate scored against a given target, and recomputing it
    per annealing move would double the cost of the inner loop.
    """
    # Derive the blank grid from the *rendered* raster shape, not from the
    # declared cols/rows: a candidate whose row/col count disagrees with the
    # declared grid (exactly what an unconstrained policy emits) rasterizes to
    # a different pixel shape, and building the baseline from cols/rows then
    # produces a shape mismatch inside SSIM. This crashed the first 600-step
    # GRPO run.
    key = (hash(target_resized.tobytes()), rendered_shape, font.cell_h, font.cell_w)
    hit = _BLANK_SSIM_CACHE.get(key)
    if hit is not None:
        return hit
    n_rows = max(1, rendered_shape[0] // font.cell_h)
    n_cols = max(1, rendered_shape[1] // font.cell_w)
    blank = "\n".join(" " * n_cols for _ in range(n_rows))
    blank_raster = rasterize(blank, font=font)
    if blank_raster.shape != target_resized.shape:
        blank_raster = np.zeros_like(target_resized)
    win = max(4, min(font.cell_h, font.cell_w))
    val = _ssim_windowed(blank_raster, target_resized, win=win)
    _BLANK_SSIM_CACHE[key] = val
    return val


def _cell_means(pixels: np.ndarray, font: Font) -> np.ndarray:
    """Mean intensity per character cell -- the resolution the medium actually
    has. Comparing at pixel resolution is what broke the previous reward: a
    correctly-filled region rasterizes to glyph *texture*, whose internal
    edges the smooth target does not have, so pixel-level edge-F1 scored a
    visibly-correct fill 3x below a sparse render that ignored the shape."""
    rows = max(1, pixels.shape[0] // font.cell_h)
    cols = max(1, pixels.shape[1] // font.cell_w)
    block = pixels[: rows * font.cell_h, : cols * font.cell_w]
    return block.reshape(rows, font.cell_h, cols, font.cell_w).mean(axis=(1, 3))


def _coverage_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation between per-cell ink coverage and per-cell target
    brightness, clipped at 0. Blank or uniform output has no variance and
    scores 0 by construction, so the empty grid needs no special case."""
    x, y = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    if x.std() < 1e-9 or y.std() < 1e-9 or x.size < 2:
        return 0.0
    with np.errstate(invalid="ignore", divide="ignore"):
        c = np.corrcoef(x, y)[0, 1]
    return 0.0 if not np.isfinite(c) else float(max(0.0, c))


def _coverage_fidelity(a: np.ndarray, b: np.ndarray) -> float:
    """1 - mean absolute difference of range-normalized cell maps. Correlation
    is scale-invariant, so on its own it would accept output that has the
    right shape at the wrong density; this term pins the density."""
    if a.std() < 1e-9 or b.std() < 1e-9:
        # Degenerate output (all blank, or a solid fill) carries no
        # information about the target; without this guard it collects a
        # ~0.55 floor from the mean-absolute-difference term, which is the
        # same empty-grid hack in a new costume.
        return 0.0

    def norm(v):
        lo, hi = float(v.min()), float(v.max())
        return (v - lo) / (hi - lo) if hi - lo > 1e-9 else np.zeros_like(v)
    return float(max(0.0, 1.0 - np.abs(norm(a) - norm(b)).mean()))


def _cell_edge_f1(a: np.ndarray, b: np.ndarray, thresh: float = 0.12) -> float:
    """Edge-F1 computed on the cell grids rather than the pixel grids, with a
    one-cell dilation tolerance. This measures whether ink boundaries land in
    the right cells, without seeing glyph-internal texture as edges."""
    ma, _ = _sobel(a)
    mb, _ = _sobel(b)
    ea, eb = ma >= thresh, mb >= thresh
    if not ea.any() and not eb.any():
        return 1.0
    if not ea.any() or not eb.any():
        return 0.0

    def dilate(mask):
        out = mask.copy()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                out |= np.roll(np.roll(mask, dy, axis=0), dx, axis=1)
        return out

    precision = np.logical_and(ea, dilate(eb)).sum() / max(ea.sum(), 1)
    recall = np.logical_and(eb, dilate(ea)).sum() / max(eb.sum(), 1)
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

    # Score on the character cell grid, not the pixel grid. See _cell_means.
    a_cells = _cell_means(rendered, font)
    b_cells = _cell_means(target_resized, font)
    coverage = _coverage_corr(a_cells, b_cells)
    fidelity = _coverage_fidelity(a_cells, b_cells)
    cell_edge = _cell_edge_f1(a_cells, b_cells)
    # Reward is coverage-driven. The cell edge-F1 is computed and reported but
    # deliberately kept out of the reward: with a one-cell dilation tolerance
    # it scores random charset noise at 0.87, because dense random edges
    # intersect any target's edge band. Coverage correlation separates the
    # same cases cleanly (0.94 correct vs 0.00 noise).
    shape = 0.8 * coverage + 0.2 * fidelity

    if not constraints["ok"]:
        reward = 0.0
    else:
        reward = shape

    return {
        "constraints": constraints,
        # Retained for reference/diagnosis only -- neither term feeds the
        # reward any more. Windowed pixel SSIM cannot separate the oracle from
        # a blank grid on sparse ink (0.493 vs 0.494), and pixel edge-F1
        # rewards sparsity; see results/summary.md.
        "ssim": ssim,
        "pixel_edge_score": edge,
        "coverage_corr": coverage,
        "coverage_fidelity": fidelity,
        "shape_score": shape,
        "edge_score": cell_edge,
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
