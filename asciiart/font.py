"""Glyph rasterization and measured ink-coverage ramps.

`Font` rasterizes every printable ASCII glyph (32-126) into a fixed-size
bitmap atlas using a monospace TTF, via PIL. Ink coverage per glyph is
MEASURED from the rendered bitmap (mean pixel intensity), not assumed from
a hardcoded ramp string. The atlas is cached as a numpy array on the
instance (`Font.atlas`, shape (95, H, W), dtype float32 in [0, 1]).
"""

from __future__ import annotations

import functools
import os
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageDraw, ImageFont

FIRST_CODEPOINT = 32
LAST_CODEPOINT = 126

# Candidate monospace TTF/TTC paths, searched in order. macOS first, then
# common Linux locations, so this works cross-platform without a bundled font.
_CANDIDATE_FONT_PATHS = [
    "/System/Library/Fonts/Menlo.ttc",
    "/System/Library/Fonts/Supplemental/Andale Mono.ttf",
    "/System/Library/Fonts/Supplemental/PTMono.ttc",
    "/Library/Fonts/DejaVuSansMono.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
    "/usr/share/fonts/truetype/msttcorefonts/Andale_Mono.ttf",
]

DEFAULT_CHARSET = "".join(chr(c) for c in range(FIRST_CODEPOINT, LAST_CODEPOINT + 1))


def _find_font_path() -> str | None:
    for path in _CANDIDATE_FONT_PATHS:
        if os.path.exists(path):
            return path
    return None


@dataclass
class Font:
    """A rasterized monospace glyph atlas with measured ink coverage.

    Attributes:
        cell_h, cell_w: glyph bitmap size in pixels (aspect ~2:1, h:w).
        path: resolved font file path (or None if PIL's built-in bitmap
            font was used as a last-resort fallback).
        atlas: np.ndarray, shape (96, cell_h, cell_w), float32 in [0, 1].
            Index i corresponds to chr(32 + i).
        coverage: np.ndarray, shape (96,), float32 in [0, 1]; mean
            intensity of each glyph bitmap (measured ink coverage).
    """

    cell_h: int = 16
    cell_w: int = 8
    path: str | None = None
    atlas: np.ndarray = field(default=None, repr=False)
    coverage: np.ndarray = field(default=None, repr=False)
    chars: str = DEFAULT_CHARSET

    def __post_init__(self):
        if self.atlas is not None and self.coverage is not None:
            return
        self.path = self.path or _find_font_path()
        pil_font, px_size = self._load_pil_font(self.path, self.cell_h)
        atlas = np.zeros((len(self.chars), self.cell_h, self.cell_w), dtype=np.float32)
        for i, ch in enumerate(self.chars):
            atlas[i] = self._rasterize_glyph(pil_font, ch, self.cell_h, self.cell_w)
        self.atlas = atlas
        self.coverage = atlas.reshape(len(self.chars), -1).mean(axis=1)

    @staticmethod
    def _load_pil_font(path: str | None, cell_h: int):
        px_size = int(cell_h * 0.92)
        if path:
            try:
                return ImageFont.truetype(path, px_size), px_size
            except Exception:
                pass
        return ImageFont.load_default(), px_size

    @staticmethod
    def _rasterize_glyph(pil_font, ch: str, h: int, w: int) -> np.ndarray:
        img = Image.new("L", (w, h), color=0)
        draw = ImageDraw.Draw(img)
        if ch != " ":
            try:
                bbox = draw.textbbox((0, 0), ch, font=pil_font)
                tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                x = (w - tw) / 2 - bbox[0]
                y = (h - th) / 2 - bbox[1]
            except Exception:
                x, y = 0, 0
            draw.text((x, y), ch, fill=255, font=pil_font)
        return np.asarray(img, dtype=np.float32) / 255.0

    def index_of(self, ch: str) -> int:
        return ord(ch) - FIRST_CODEPOINT

    def glyph(self, ch: str) -> np.ndarray:
        return self.atlas[self.index_of(ch)]

    def luminance_ramp(self, charset: str | None = None) -> str:
        """Return `charset` (default: this font's full printable set) sorted
        by MEASURED ink coverage, ascending (sparse/blank -> dense/dark-on-
        light... actually here: low coverage -> high coverage), deduplicated
        by coverage value to avoid redundant steps.

        This replaces a hardcoded ramp like " .:-=+*#%@" with one derived
        from how the actual font renders.
        """
        cs = charset if charset is not None else self.chars
        pairs = [(ch, float(self.coverage[self.index_of(ch)])) for ch in cs if " " <= ch <= "~"]
        pairs.sort(key=lambda p: p[1])
        seen = set()
        ramp = []
        for ch, cov in pairs:
            key = round(cov, 4)
            if key in seen:
                continue
            seen.add(key)
            ramp.append(ch)
        if not ramp:
            ramp = [" ", "#"]
        return "".join(ramp)


@functools.lru_cache(maxsize=4)
def _default_font_cached(cell_h: int, cell_w: int) -> Font:
    return Font(cell_h=cell_h, cell_w=cell_w)


def default_font(cell_h: int = 16, cell_w: int = 8) -> Font:
    """Return a process-cached default Font for the given cell size."""
    return _default_font_cached(cell_h, cell_w)
