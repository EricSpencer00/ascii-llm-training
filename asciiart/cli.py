"""CLI: python -m asciiart.cli image.png --cols 80 --mode structure"""

from __future__ import annotations

import argparse

from PIL import Image

from asciiart.render import MODES, render


def _add_render_args(p):
    p.add_argument("image", help="path to input image")
    p.add_argument("--cols", type=int, default=80)
    p.add_argument("--rows", type=int, default=None)
    p.add_argument("--invert", action="store_true")
    p.add_argument("--charset", default=None, help="restrict output to these characters")
    p.add_argument("-o", "--output", default=None, help="write to file instead of stdout")


def _write_or_print(text: str, output: str | None):
    if output:
        with open(output, "w") as f:
            f.write(text + "\n")
    else:
        print(text)


def _cmd_render(args):
    img = Image.open(args.image)
    text = render(
        img,
        cols=args.cols,
        rows=args.rows,
        mode=args.mode,
        charset=args.charset,
        invert=args.invert,
    )
    _write_or_print(text, args.output)


def _cmd_verify(args):
    from asciiart.verify import score

    img = Image.open(args.image)
    with open(args.text) as f:
        text = f.read().rstrip("\n")
    result = score(text, img, cols=args.cols, rows=args.rows)
    import json

    print(json.dumps(result, indent=2))


def _cmd_anneal(args):
    from asciiart.anneal import anneal

    img = Image.open(args.image)
    text, history = anneal(img, cols=args.cols, rows=args.rows, steps=args.steps, seed=args.seed)
    _write_or_print(text, args.output)
    if args.history:
        import json

        with open(args.history, "w") as f:
            json.dump(history, f)


def main():
    p = argparse.ArgumentParser(description="Render an image as ASCII art.")
    sub = p.add_subparsers(dest="command")

    p_render = sub.add_parser("render", help="render an image to ASCII (default command)")
    _add_render_args(p_render)
    p_render.add_argument("--mode", choices=MODES, default="structure")
    p_render.set_defaults(func=_cmd_render)

    p_verify = sub.add_parser("verify", help="score ASCII text against a target image")
    p_verify.add_argument("image", help="path to target image")
    p_verify.add_argument("text", help="path to ASCII text file")
    p_verify.add_argument("--cols", type=int, default=80)
    p_verify.add_argument("--rows", type=int, default=None)
    p_verify.set_defaults(func=_cmd_verify)

    p_anneal = sub.add_parser("anneal", help="simulated-annealing search for ASCII art")
    _add_render_args(p_anneal)
    p_anneal.add_argument("--steps", type=int, default=2000)
    p_anneal.add_argument("--seed", type=int, default=0)
    p_anneal.add_argument("--history", default=None, help="write step history JSON to this path")
    p_anneal.set_defaults(func=_cmd_anneal)

    # Backward compatible default: `python -m asciiart.cli image.png ...` with
    # no subcommand behaves like the original render-only CLI.
    known_commands = {"render", "verify", "anneal"}
    import sys

    argv = sys.argv[1:]
    if argv and argv[0] in ("-h", "--help"):
        p.parse_args(argv)
        return
    if not argv or argv[0] not in known_commands:
        p_render_default = argparse.ArgumentParser(description="Render an image as ASCII art.")
        _add_render_args(p_render_default)
        p_render_default.add_argument("--mode", choices=MODES, default="structure")
        args = p_render_default.parse_args(argv)
        _cmd_render(args)
        return

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
