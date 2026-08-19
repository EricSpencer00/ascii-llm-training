"""asciiart: deterministic image-to-ASCII rendering, used as a reference
implementation and testbed for verifiable grid-structured text generation.

Public API:
    from asciiart.font import Font
    from asciiart.render import render, rasterize
"""

from asciiart.font import Font
from asciiart.render import render, rasterize

__all__ = ["Font", "render", "rasterize"]
