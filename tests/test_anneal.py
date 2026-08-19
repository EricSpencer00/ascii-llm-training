import numpy as np

from asciiart.anneal import anneal
from asciiart.font import default_font
from asciiart.verify import score


def _gradient_image(rows=6, cols=10, font=None):
    font = font or default_font()
    h, w = rows * font.cell_h, cols * font.cell_w
    yy, xx = np.mgrid[0:h, 0:w]
    img = ((xx / w) * 255).astype(np.uint8)
    return img


def test_anneal_returns_correct_shape():
    font = default_font()
    cols, rows = 12, 6
    img = _gradient_image(rows, cols, font)
    text, history = anneal(img, cols=cols, rows=rows, steps=50, seed=1, font=font)
    lines = text.split("\n")
    assert len(lines) == rows
    assert all(len(l) == cols for l in lines)
    assert len(history) > 0


def test_anneal_improves_or_matches_initial_reward():
    font = default_font()
    cols, rows = 12, 6
    img = _gradient_image(rows, cols, font)

    from asciiart.render import render as render_fn

    init_text = render_fn(img, cols=cols, rows=rows, mode="structure", font=font)
    init_score = score(init_text, img, font=font, cols=cols, rows=rows)

    text, history = anneal(img, cols=cols, rows=rows, steps=300, seed=2, font=font)
    final_score = score(text, img, font=font, cols=cols, rows=rows)

    assert final_score["reward"] >= init_score["reward"] - 1e-6


def test_anneal_deterministic_with_seed():
    font = default_font()
    cols, rows = 10, 5
    img = _gradient_image(rows, cols, font)
    text1, _ = anneal(img, cols=cols, rows=rows, steps=100, seed=42, font=font)
    text2, _ = anneal(img, cols=cols, rows=rows, steps=100, seed=42, font=font)
    assert text1 == text2
