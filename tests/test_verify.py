import numpy as np
import pytest

from asciiart.font import default_font
from asciiart.verify import check_constraints, score, demonstrate_hack


def test_check_constraints_ok():
    text = "ab\ncd"
    result = check_constraints(text, cols=2, rows=2, charset="abcd")
    assert result["ok"] is True
    assert result["row_count_ok"] and result["width_ok"] and result["charset_ok"] and result["trailing_ws_ok"]


def test_check_constraints_wrong_width():
    text = "abc\ncd"
    result = check_constraints(text, cols=2, rows=2)
    assert result["ok"] is False
    assert result["width_ok"] is False


def test_check_constraints_wrong_row_count():
    text = "ab\ncd\nef"
    result = check_constraints(text, cols=2, rows=2)
    assert result["ok"] is False
    assert result["row_count_ok"] is False


def test_check_constraints_bad_charset():
    text = "ab\ncZ"
    result = check_constraints(text, cols=2, rows=2, charset="abcd")
    assert result["ok"] is False
    assert result["charset_ok"] is False


def test_check_constraints_trailing_whitespace():
    text = "ab \ncd"
    result = check_constraints(text, cols=3, rows=2, charset="abcd ")
    assert result["trailing_ws_ok"] is False


def test_check_constraints_blank_row_allowed():
    text = "  \ncd"
    result = check_constraints(text, cols=2, rows=2, charset="cd ")
    assert result["trailing_ws_ok"] is True


def test_score_perfect_self_match():
    font = default_font()
    cols, rows = 10, 6
    from asciiart.render import render, rasterize

    size_h, size_w = rows * font.cell_h, cols * font.cell_w
    yy, xx = np.mgrid[0:size_h, 0:size_w]
    img = ((xx / size_w) * 255).astype(np.uint8)

    text = render(img, cols=cols, rows=rows, mode="structure", font=font)
    rendered = rasterize(text, font=font)
    result = score(text, rendered, font=font, cols=cols, rows=rows)
    assert result["constraints"]["ok"] is True
    assert result["ssim"] > 0.9
    assert result["reward"] > 0.5


def test_score_zero_reward_on_constraint_violation():
    font = default_font()
    cols, rows = 10, 6
    img = np.full((rows * font.cell_h, cols * font.cell_w), 128, dtype=np.uint8)
    bad_text = "x" * (cols + 1) + "\n" + ("x" * cols + "\n") * (rows - 1)
    bad_text = bad_text.rstrip("\n")
    result = score(bad_text, img, font=font, cols=cols, rows=rows, charset="x")
    assert result["constraints"]["ok"] is False
    assert result["reward"] == 0.0


def test_demonstrate_hack_uniform_fill_loses_on_combined_reward():
    result = demonstrate_hack(cols=30, rows=15)
    assert result["real_beats_hack_on_reward"] is True
    assert result["real"]["edge_score"] > result["hack"]["edge_score"]
    # sanity: the hack's SSIM should be roughly comparable to the real
    # render's (not wildly lower), otherwise the demonstration is trivial
    # and doesn't isolate the edge term's contribution specifically.
    assert abs(result["hack"]["ssim"] - result["real"]["ssim"]) < 0.05
