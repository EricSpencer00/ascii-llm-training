"""CLI: python -m asciiart.cli image.png --cols 80 --mode structure"""

from __future__ import annotations

import argparse

from PIL import Image

from asciiart.render import MODES, render


def main():
    p = argparse.ArgumentParser(description="Render an image as ASCII art.")
    p.add_argument("image", help="path to input image")
    p.add_argument("--cols", type=int, default=80)
    p.add_argument("--rows", type=int, default=None)
    p.add_argument("--mode", choices=MODES, default="structure")
    p.add_argument("--invert", action="store_true")
    p.add_argument("--charset", default=None, help="restrict output to these characters")
    p.add_argument("-o", "--output", default=None, help="write to file instead of stdout")
    args = p.parse_args()

    img = Image.open(args.image)
    text = render(
        img,
        cols=args.cols,
        rows=args.rows,
        mode=args.mode,
        charset=args.charset,
        invert=args.invert,
    )
    if args.output:
        with open(args.output, "w") as f:
            f.write(text + "\n")
    else:
        print(text)


if __name__ == "__main__":
    main()
