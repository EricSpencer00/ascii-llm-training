"""Tests for asciiart.render, using synthetic PIL-drawn images."""

import numpy as np
import pytest
from PIL import Image, ImageDraw

from asciiart.font import Font, default_font
from asciiart.render import rasterize, render


@pytest.fixture(scope="module")
def font():
    return default_font()


def make_circle(size=128):
    img = Image.new("L", (size, size), color=0)
    draw = ImageDraw.Draw(img)
    draw.ellipse((size * 0.2, size * 0.2, size * 0.8, size * 0.8), fill=255)
    return img


def make_diagonal(size=128):
    img = Image.new("L", (size, size), color=0)
    draw = ImageDraw.Draw(img)
    draw.line((0, size - 1, size - 1, 0), fill=255, width=max(2, size // 32))
    return img


def make_gradient(size=128):
    arr = np.tile(np.linspace(0, 255, size, dtype=np.uint8), (size, 1))
    return Image.fromarray(arr, mode="L")


def test_render_dims_luminance(font):
    img = make_circle()
    out = render(img, cols=40, mode="luminance", font=font)
    lines = out.split("\n")
    assert len(lines) > 0
    assert all(len(line) == 40 for line in lines)


def test_render_dims_explicit_rows(font):
    img = make_gradient()
    out = render(img, cols=30, rows=10, mode="luminance", font=font)
    lines = out.split("\n")
    assert len(lines) == 10
    assert all(len(line) == 30 for line in lines)


def test_render_charset_restricted(font):
    img = make_gradient()
    charset = " .#"
    out = render(img, cols=20, rows=8, mode="luminance", charset=charset, font=font)
    used = set(out.replace("\n", ""))
    assert used <= set(charset)


@pytest.mark.parametrize("mode", ["luminance", "structure", "edge"])
def test_render_modes_run(font, mode):
    img = make_circle()
    out = render(img, cols=24, rows=12, mode=mode, font=font)
    assert isinstance(out, str)
    lines = out.split("\n")
    assert len(lines) == 12
    assert all(len(line) == 24 for line in lines)


def test_edge_mode_finds_diagonal(font):
    img = make_diagonal(size=128)
    out = render(img, cols=32, rows=16, mode="edge", font=font)
    chars_used = set(out.replace("\n", ""))
    assert ("/" in chars_used) or ("\\" in chars_used)


def test_structure_beats_luminance_on_diagonal(font):
    """Structure-mode rasterize->SSIM should beat luminance mode on a line
    image, since structure mode explicitly matches edge orientation."""
    img = make_diagonal(size=128)
    gray_src = np.asarray(img.convert("L"), dtype=np.float32) / 255.0

    lum_text = render(img, cols=32, rows=16, mode="luminance", font=font)
    struct_text = render(img, cols=32, rows=16, mode="structure", font=font)

    lum_raster = rasterize(lum_text, font=font)
    struct_raster = rasterize(struct_text, font=font)

    src_img = Image.fromarray((gray_src * 255).astype(np.uint8))
    lum_target = np.asarray(
        src_img.resize((lum_raster.shape[1], lum_raster.shape[0]), Image.BILINEAR), dtype=np.float32
    ) / 255.0
    struct_target = np.asarray(
        src_img.resize((struct_raster.shape[1], struct_raster.shape[0]), Image.BILINEAR), dtype=np.float32
    ) / 255.0

    lum_score = _ssim(lum_raster, lum_target)
    struct_score = _ssim(struct_raster, struct_target)

    assert struct_score >= lum_score - 0.02


def _ssim(a: np.ndarray, b: np.ndarray, win: int = 8) -> float:
    """Windowed (block) SSIM, averaged over non-overlapping win x win tiles.
    A coarse but locality-sensitive approximation of real SSIM — unlike a
    single global SSIM, it rewards renderings whose local structure lines
    up with the target rather than just matching overall mean/variance."""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    h, w = a.shape
    c1, c2 = (0.01 * 1) ** 2, (0.03 * 1) ** 2
    scores = []
    for y in range(0, h - win + 1, win):
        for x in range(0, w - win + 1, win):
            wa = a[y : y + win, x : x + win]
            wb = b[y : y + win, x : x + win]
            mu_a, mu_b = wa.mean(), wb.mean()
            var_a, var_b = wa.var(), wb.var()
            cov = ((wa - mu_a) * (wb - mu_b)).mean()
            s = ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / (
                (mu_a**2 + mu_b**2 + c1) * (var_a + var_b + c2)
            )
            scores.append(s)
    return float(np.mean(scores)) if scores else 0.0


def test_rasterize_shape(font):
    text = "ab\ncd"
    out = rasterize(text, font=font)
    assert out.shape == (2 * font.cell_h, 2 * font.cell_w)


def test_luminance_ramp_measured(font):
    ramp = font.luminance_ramp()
    assert len(ramp) >= 2
    covs = [font.coverage[font.index_of(c)] for c in ramp]
    assert covs == sorted(covs)
