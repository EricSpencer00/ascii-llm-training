"""Evaluation protocol: a fixed synthetic image set, scored against every
render mode plus the annealing baseline, written to results/bench.md.

Run: `.venv/bin/python -m asciiart.bench`
"""

from __future__ import annotations

import os
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont as PILImageFont

from asciiart.anneal import anneal
from asciiart.font import default_font
from asciiart.render import MODES, render
from asciiart.verify import score

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
N_IMAGES = 20
IMG_SIZE = 128
COLS = 40
ANNEAL_STEPS = 2000


def _make_images(n: int = N_IMAGES, size: int = IMG_SIZE, seed: int = 0):
    """Deterministic synthetic set: shapes, gradients, and text, at
    varying contrast/position so structure/edge modes are exercised."""
    rng = np.random.RandomState(seed)
    images = []
    for i in range(n):
        kind = i % 4
        im = Image.new("L", (size, size), color=0)
        draw = ImageDraw.Draw(im)
        bg = int(rng.uniform(0, 60))
        draw.rectangle([0, 0, size, size], fill=bg)
        fg = int(rng.uniform(180, 255))

        if kind == 0:
            # circle
            r = rng.uniform(0.2, 0.4) * size
            cx, cy = rng.uniform(0.4, 0.6) * size, rng.uniform(0.4, 0.6) * size
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=fg)
        elif kind == 1:
            # rectangle / logo-like block shapes
            x0, y0 = rng.uniform(0.1, 0.3) * size, rng.uniform(0.1, 0.3) * size
            x1, y1 = rng.uniform(0.6, 0.9) * size, rng.uniform(0.6, 0.9) * size
            draw.rectangle([x0, y0, x1, y1], fill=fg)
            draw.ellipse([x0 + 10, y0 + 10, x1 - 10, y1 - 10], fill=bg)
        elif kind == 2:
            # gradient (smooth, low edge content -> tests uniform-fill risk)
            arr = np.zeros((size, size), dtype=np.float32)
            angle = rng.uniform(0, np.pi)
            yy, xx = np.mgrid[0:size, 0:size]
            proj = xx * np.cos(angle) + yy * np.sin(angle)
            proj = (proj - proj.min()) / (proj.max() - proj.min() + 1e-6)
            arr = bg + proj * (fg - bg)
            im = Image.fromarray(arr.astype(np.uint8))
        else:
            # text / line-art
            try:
                font = PILImageFont.load_default()
            except Exception:
                font = None
            draw.line([10, size // 2, size - 10, size // 2], fill=fg, width=3)
            draw.line([size // 2, 10, size // 2, size - 10], fill=fg, width=3)
            draw.text((size // 4, size // 4), "AI", fill=fg, font=font)

        images.append(im)
    return images


def run_bench(n: int = N_IMAGES, cols: int = COLS, anneal_steps: int = ANNEAL_STEPS, seed: int = 0):
    font = default_font()
    images = _make_images(n=n, seed=seed)

    rows_out = []
    mode_sums = {m: {"ssim": 0.0, "edge": 0.0, "reward": 0.0} for m in MODES}
    anneal_sum = {"ssim": 0.0, "edge": 0.0, "reward": 0.0}

    t_start = time.time()
    for i, img in enumerate(images):
        arr = np.asarray(img, dtype=np.float32) / 255.0
        row = {"image": i}
        for mode in MODES:
            text = render(arr, cols=cols, mode=mode, font=font)
            rows_n = len(text.split("\n"))
            result = score(text, arr, font=font, cols=cols, rows=rows_n)
            row[f"{mode}_ssim"] = result["ssim"]
            row[f"{mode}_edge"] = result["edge_score"]
            row[f"{mode}_reward"] = result["reward"]
            mode_sums[mode]["ssim"] += result["ssim"]
            mode_sums[mode]["edge"] += result["edge_score"]
            mode_sums[mode]["reward"] += result["reward"]

        anneal_text, _hist = anneal(arr, cols=cols, steps=anneal_steps, seed=seed + i, font=font)
        rows_n = len(anneal_text.split("\n"))
        a_result = score(anneal_text, arr, font=font, cols=cols, rows=rows_n)
        row["anneal_ssim"] = a_result["ssim"]
        row["anneal_edge"] = a_result["edge_score"]
        row["anneal_reward"] = a_result["reward"]
        anneal_sum["ssim"] += a_result["ssim"]
        anneal_sum["edge"] += a_result["edge_score"]
        anneal_sum["reward"] += a_result["reward"]

        rows_out.append(row)

    elapsed = time.time() - t_start

    means = {m: {k: v / n for k, v in mode_sums[m].items()} for m in MODES}
    anneal_means = {k: v / n for k, v in anneal_sum.items()}

    beats_structure = anneal_means["reward"] > means["structure"]["reward"]

    md_lines = []
    md_lines.append(f"# ASCII art bench ({n} images, cols={cols}, anneal_steps={anneal_steps})")
    md_lines.append("")
    md_lines.append(f"Wall time: {elapsed:.1f}s")
    md_lines.append("")
    md_lines.append("## Per-mode means")
    md_lines.append("")
    md_lines.append("| method | ssim | edge_score | reward |")
    md_lines.append("|---|---|---|---|")
    for mode in MODES:
        m = means[mode]
        md_lines.append(f"| {mode} | {m['ssim']:.4f} | {m['edge']:.4f} | {m['reward']:.4f} |")
    md_lines.append(
        f"| anneal ({anneal_steps} steps) | {anneal_means['ssim']:.4f} | "
        f"{anneal_means['edge']:.4f} | {anneal_means['reward']:.4f} |"
    )
    md_lines.append("")
    md_lines.append(f"**Anneal beats structure on mean reward: {beats_structure}** "
                     f"({anneal_means['reward']:.4f} vs {means['structure']['reward']:.4f})")
    md_lines.append("")
    md_lines.append("## Per-image detail")
    md_lines.append("")
    header = "| image |" + "".join(f" {m} ssim/edge/reward |" for m in MODES) + " anneal ssim/edge/reward |"
    sep = "|---|" + "---|" * (len(MODES) + 1)
    md_lines.append(header)
    md_lines.append(sep)
    for row in rows_out:
        cells = [str(row["image"])]
        for mode in MODES:
            cells.append(f"{row[f'{mode}_ssim']:.3f}/{row[f'{mode}_edge']:.3f}/{row[f'{mode}_reward']:.3f}")
        cells.append(f"{row['anneal_ssim']:.3f}/{row['anneal_edge']:.3f}/{row['anneal_reward']:.3f}")
        md_lines.append("| " + " | ".join(cells) + " |")

    md_text = "\n".join(md_lines) + "\n"
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "bench.md")
    with open(out_path, "w") as f:
        f.write(md_text)

    return {
        "means": means,
        "anneal_means": anneal_means,
        "beats_structure": beats_structure,
        "elapsed": elapsed,
        "out_path": out_path,
    }


if __name__ == "__main__":
    result = run_bench()
    print(f"wrote {result['out_path']} in {result['elapsed']:.1f}s")
    print(f"anneal beats structure: {result['beats_structure']}")
