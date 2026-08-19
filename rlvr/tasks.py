"""Synthetic RLVR task generator.

Each task pairs a deterministically-generated target image (simple shapes /
text rendered with PIL, no external dataset) with a natural-language prompt
asking the policy to draw it as a `cols x rows` ASCII grid over a fixed
charset. `asciiart.render.render` (mode="structure") on the same image is
included as the oracle/reference the policy is judged against, per
docs/design.md's evaluation protocol.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass

from PIL import Image, ImageDraw, ImageFont

from asciiart.render import render

DEFAULT_ASCII_CHARSET = " .:-=+*#%@"

_SHAPES = ["circle", "square", "triangle", "cross", "letter"]
_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

PROMPT_TEMPLATE = (
    "Draw {description} as ASCII art, exactly {cols}x{rows} characters, "
    'using only this charset (space is index 0): "{charset}". '
    "Output only the grid, {rows} lines of exactly {cols} characters each, "
    "nothing else."
)


@dataclass
class Task:
    task_id: str
    description: str
    image: Image.Image
    cols: int
    rows: int
    charset: str
    prompt: str

    def oracle_text(self) -> str:
        """Deterministic converter output for this task's image -- the
        reference every prompt-driven method is judged against."""
        return render(
            self.image,
            cols=self.cols,
            rows=self.rows,
            mode="structure",
            charset=self.charset,
        )


def _draw_shape(shape: str, size: int, rng: random.Random) -> tuple[Image.Image, str]:
    img = Image.new("L", (size, size), color=255)
    draw = ImageDraw.Draw(img)
    pad = size // 6
    box = (pad, pad, size - pad, size - pad)

    if shape == "circle":
        draw.ellipse(box, fill=0)
        desc = "a filled circle"
    elif shape == "square":
        draw.rectangle(box, fill=0)
        desc = "a filled square"
    elif shape == "triangle":
        x0, y0, x1, y1 = box
        draw.polygon([(x0 + (x1 - x0) / 2, y0), (x0, y1), (x1, y1)], fill=0)
        desc = "a filled triangle"
    elif shape == "cross":
        cx, cy = size // 2, size // 2
        arm = size // 8
        draw.rectangle((cx - arm, pad, cx + arm, size - pad), fill=0)
        draw.rectangle((pad, cy - arm, size - pad, cy + arm), fill=0)
        desc = "a plus-shaped cross"
    else:  # letter
        letter = rng.choice(_LETTERS)
        try:
            font = ImageFont.load_default(size=size // 2)
        except TypeError:
            font = ImageFont.load_default()
        bbox = draw.textbbox((0, 0), letter, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text(
            ((size - tw) / 2 - bbox[0], (size - th) / 2 - bbox[1]),
            letter,
            fill=0,
            font=font,
        )
        desc = f"the letter {letter}"

    return img, desc


def generate_tasks(
    n: int,
    seed: int = 0,
    cols: int = 24,
    rows: int = 12,
    charset: str = DEFAULT_ASCII_CHARSET,
    image_size: int = 128,
) -> list[Task]:
    """Deterministically generate `n` tasks. Same `seed` -> same tasks,
    every time -- required for the evaluation protocol's fixed set."""
    rng = random.Random(seed)
    tasks = []
    for i in range(n):
        shape = _SHAPES[i % len(_SHAPES)]
        img, desc = _draw_shape(shape, image_size, rng)
        prompt = PROMPT_TEMPLATE.format(
            description=desc, cols=cols, rows=rows, charset=charset
        )
        task_id = hashlib.sha1(f"{seed}-{i}-{desc}".encode()).hexdigest()[:10]
        tasks.append(
            Task(
                task_id=task_id,
                description=desc,
                image=img,
                cols=cols,
                rows=rows,
                charset=charset,
                prompt=prompt,
            )
        )
    return tasks
