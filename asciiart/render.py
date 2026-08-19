"""Image -> ASCII rendering, pure numpy (+PIL for image IO).

Three tiers, selected via `mode`:
    luminance  - per-cell mean brightness -> nearest-coverage glyph.
    structure  - per-cell glyph chosen by minimizing a perceptual distance
                 (MSE + gradient-orientation term, in the spirit of Xu et
                 al.'s structure-based ASCII art) against the source tile
                 resized to glyph resolution.
    edge       - Sobel gradient magnitude/orientation; high-gradient cells
                 get a directional glyph (| / - \\), else luminance fill.

`rasterize(ascii_text, font)` is the shared, stable entry point other
components (e.g. a verifier) import to turn ASCII text back into a
grayscale numpy image in the same font, for pixel-level comparison against
a render's source tiles.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from asciiart.font import Font, default_font

MODES = ("luminance", "structure", "edge")


def _to_gray_array(img) -> np.ndarray:
    """Accept PIL.Image or np.ndarray, return float32 HxW grayscale in [0,1]."""
    if isinstance(img, Image.Image):
        arr = np.asarray(img.convert("L"), dtype=np.float32) / 255.0
    else:
        arr = np.asarray(img)
        if arr.ndim == 3:
            arr = arr[..., :3].mean(axis=-1)
        arr = arr.astype(np.float32)
        if arr.max() > 1.0:
            arr = arr / 255.0
    return arr


def _compute_grid(h: int, w: int, cols: int, rows: int | None, cell_h: int, cell_w: int):
    if rows is None:
        # cell aspect ~2:1 (h:w) so visually-square output: rows scale by
        # image aspect ratio corrected for the glyph cell's own aspect.
        aspect_correction = cell_w / cell_h
        rows = max(1, round((h / w) * cols * aspect_correction))
    return cols, rows


def _resize_to_grid(gray: np.ndarray, cols: int, rows: int) -> np.ndarray:
    im = Image.fromarray((np.clip(gray, 0, 1) * 255).astype(np.uint8))
    im = im.resize((cols, rows), Image.BILINEAR)
    return np.asarray(im, dtype=np.float32) / 255.0


def _sobel(gray: np.ndarray):
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    ky = kx.T
    padded = np.pad(gray, 1, mode="edge")
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    for i in range(3):
        for j in range(3):
            if kx[i, j] != 0:
                gx += kx[i, j] * padded[i : i + gray.shape[0], j : j + gray.shape[1]]
            if ky[i, j] != 0:
                gy += ky[i, j] * padded[i : i + gray.shape[0], j : j + gray.shape[1]]
    mag = np.hypot(gx, gy)
    ang = np.arctan2(gy, gx)
    return mag, ang


def _nearest_by_coverage(cell_mean: float, ramp: str, font: Font) -> str:
    covs = np.array([font.coverage[font.index_of(c)] for c in ramp])
    idx = int(np.argmin(np.abs(covs - cell_mean)))
    return ramp[idx]


def _render_luminance(gray: np.ndarray, cols: int, rows: int, ramp: str, font: Font, invert: bool) -> str:
    grid = _resize_to_grid(gray, cols, rows)
    if invert:
        grid = 1.0 - grid
    lines = []
    for r in range(rows):
        line = "".join(_nearest_by_coverage(float(grid[r, c]), ramp, font) for c in range(cols))
        lines.append(line)
    return "\n".join(lines)


def _tile_resized_to_glyph(gray: np.ndarray, r: int, c: int, rows: int, cols: int, h: int, w: int, gh: int, gw: int) -> np.ndarray:
    y0 = int(r * h / rows)
    y1 = max(y0 + 1, int((r + 1) * h / rows))
    x0 = int(c * w / cols)
    x1 = max(x0 + 1, int((c + 1) * w / cols))
    tile = gray[y0:y1, x0:x1]
    im = Image.fromarray((np.clip(tile, 0, 1) * 255).astype(np.uint8))
    im = im.resize((gw, gh), Image.BILINEAR)
    return np.asarray(im, dtype=np.float32) / 255.0


def _render_structure(gray: np.ndarray, cols: int, rows: int, ramp: str, font: Font, invert: bool) -> str:
    h, w = gray.shape
    gh, gw = font.cell_h, font.cell_w
    src = 1.0 - gray if invert else gray
    ramp_idx = [font.index_of(c) for c in ramp]
    ramp_glyphs = font.atlas[ramp_idx]  # (K, gh, gw)
    # precompute glyph gradients for the orientation term
    ramp_gmag = np.zeros((len(ramp), gh, gw), dtype=np.float32)
    ramp_gang = np.zeros((len(ramp), gh, gw), dtype=np.float32)
    for i in range(len(ramp)):
        m, a = _sobel(ramp_glyphs[i])
        ramp_gmag[i], ramp_gang[i] = m, a

    lines = []
    for r in range(rows):
        row_chars = []
        for c in range(cols):
            tile = _tile_resized_to_glyph(src, r, c, rows, cols, h, w, gh, gw)
            tmag, tang = _sobel(tile)
            # Batched distance against all K ramp glyphs at once (numpy
            # broadcasting over the glyph axis) instead of a Python loop
            # over the ramp — this K-loop was the dominant cost of
            # structure mode (K ~ dozens to ~90 deduped glyphs per cell).
            mse = ((tile[None, :, :] - ramp_glyphs) ** 2).mean(axis=(1, 2))
            # gradient-orientation term: penalize magnitude/angle mismatch
            # weighted by where the tile actually has strong edges.
            mag_diff = ((tmag[None, :, :] - ramp_gmag) ** 2).mean(axis=(1, 2))
            w_ang = tmag / (tmag.max() + 1e-6)
            ang_diff = (w_ang[None, :, :] * (1.0 - np.cos(tang[None, :, :] - ramp_gang))).mean(axis=(1, 2))
            dist = mse + 0.5 * mag_diff + 0.5 * ang_diff
            best_i = int(np.argmin(dist))
            row_chars.append(ramp[best_i])
        lines.append("".join(row_chars))
    return "\n".join(lines)


_EDGE_CHARS = {
    "vert": "|",
    "horiz": "-",
    "diag_fwd": "/",   # bottom-left to top-right
    "diag_back": "\\",  # top-left to bottom-right
}


def _angle_to_edge_char(ang: float) -> str:
    # gradient direction is perpendicular to the edge itself; rotate 90deg
    # to get edge orientation, then bucket into 4 directions.
    edge_ang = ang + np.pi / 2
    deg = np.degrees(edge_ang) % 180
    if deg < 22.5 or deg >= 157.5:
        return _EDGE_CHARS["horiz"]
    if 22.5 <= deg < 67.5:
        return _EDGE_CHARS["diag_back"]
    if 67.5 <= deg < 112.5:
        return _EDGE_CHARS["vert"]
    return _EDGE_CHARS["diag_fwd"]


def _render_edge(gray: np.ndarray, cols: int, rows: int, ramp: str, font: Font, invert: bool) -> str:
    h, w = gray.shape
    src = 1.0 - gray if invert else gray
    mag, ang = _sobel(src)
    grid_lum = _resize_to_grid(src, cols, rows)
    # per-cell max gradient magnitude and dominant angle
    lines = []
    nonzero = mag[mag > 1e-6]
    thresh = float(np.percentile(nonzero, 50)) if nonzero.size else 0.0
    for r in range(rows):
        row_chars = []
        y0 = int(r * h / rows)
        y1 = max(y0 + 1, int((r + 1) * h / rows))
        for c in range(cols):
            x0 = int(c * w / cols)
            x1 = max(x0 + 1, int((c + 1) * w / cols))
            cell_mag = mag[y0:y1, x0:x1]
            if cell_mag.size and cell_mag.max() >= thresh and thresh > 1e-6:
                flat_idx = int(np.argmax(cell_mag))
                cell_ang = ang[y0:y1, x0:x1].flat[flat_idx]
                row_chars.append(_angle_to_edge_char(float(cell_ang)))
            else:
                row_chars.append(_nearest_by_coverage(float(grid_lum[r, c]), ramp, font))
        lines.append("".join(row_chars))
    return "\n".join(lines)


def render(
    img,
    cols: int = 80,
    rows: int | None = None,
    mode: str = "structure",
    charset: str | None = None,
    invert: bool = False,
    font: Font | None = None,
) -> str:
    """Render an image to ASCII text.

    Args:
        img: PIL.Image or np.ndarray (grayscale or RGB, uint8 or float).
        cols: output width in characters.
        rows: output height in characters; if None, derived from image
            aspect ratio corrected for the font's cell aspect ratio (~2:1).
        mode: one of "luminance", "structure", "edge".
        charset: characters to draw from; if None, uses a luminance ramp
            measured from `font`'s glyph coverage (see Font.luminance_ramp).
        invert: invert luminance (for dark-background source images).
        font: a Font instance; defaults to a cached system default.

    Returns:
        Multi-line ASCII string, `rows` lines of `cols` characters each.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    font = font or default_font()
    gray = _to_gray_array(img)
    h, w = gray.shape
    cols, rows = _compute_grid(h, w, cols, rows, font.cell_h, font.cell_w)
    ramp = font.luminance_ramp(charset)

    if mode == "luminance":
        return _render_luminance(gray, cols, rows, ramp, font, invert)
    if mode == "structure":
        return _render_structure(gray, cols, rows, ramp, font, invert)
    return _render_edge(gray, cols, rows, ramp, font, invert)


def rasterize(ascii_text: str, font: Font | None = None) -> np.ndarray:
    """Render ASCII text back into a grayscale image using `font`'s glyph
    atlas. Stable shared entry point: `from asciiart.font import Font;
    from asciiart.render import rasterize`.

    Args:
        ascii_text: multi-line string (lines may have differing length;
            output width is the max line length, short lines are padded
            with spaces).
        font: a Font instance; defaults to a cached system default.

    Returns:
        np.ndarray, shape (rows * cell_h, cols * cell_w), float32 in [0, 1].
    """
    font = font or default_font()
    lines = ascii_text.split("\n")
    rows = len(lines)
    cols = max((len(l) for l in lines), default=0)
    gh, gw = font.cell_h, font.cell_w
    out = np.zeros((rows * gh, max(cols, 1) * gw), dtype=np.float32)
    for r, line in enumerate(lines):
        for c, ch in enumerate(line):
            if " " <= ch <= "~":
                out[r * gh : (r + 1) * gh, c * gw : (c + 1) * gw] = font.glyph(ch)
    return out
