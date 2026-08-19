# ASCII LLM Training

A testbed for two different problems that both produce ASCII art, and that get
conflated more often than they should:

1. **Image to ASCII is deterministic rendering.** Given an image and a
   target grid size, there is a well-defined mapping to characters. This is a
   solved problem with known techniques (luminance ramps, structure-aware
   glyph matching, edge detection). No model is required.
2. **Prompt to ASCII by an LLM is unsolved, and current LLMs are bad at it.**
   Byte-pair encoding destroys the character grid, transformers have no 2D
   receptive field over the output, autoregressive left-to-right generation
   is the wrong generation order for a raster, and there has been no
   differentiable or checkable reward to train against. Existing SFT corpora
   (e.g. the ~140k-example HF ASCII-art dataset) just imitate the output of
   `ascii-image-converter`, so a model trained on them learns to imitate a
   renderer instead of learning to render.

The correct decomposition of "turn a prompt into ASCII art" is: prompt to
generative image model, then image to ASCII via the deterministic converter
in problem 1. What is actually worth researching is the middle ground: ASCII
grids have checkable structure (fixed width, fixed charset, and a target
image to compare against), so they support a real verifier. That verifier
enables constrained decoding, RLVR, and non-neural search baselines. This
repo builds the converter, the verifier, and those baselines side by side so
they can be compared honestly.

## Repo layout

```
asciiart/           deterministic image -> ASCII converter, verifier, search baseline
  render.py          render(), rasterize(), CLI entry point
  verify.py          SSIM + edge-alignment verifier
  anneal.py          simulated-annealing baseline search
  cli.py             python -m asciiart.cli
docs/
  design.md           full argument, algorithms, reward-hacking risks, eval protocol
  roadmap.md          phased plan with acceptance criteria
scripts/             ALCF Sophia PBS job scripts
legacy word-OCR pipeline (see below): ascii_generator.py, data_prep.py,
  model.py, train.py, evaluate.py, config.py
```

## Quick start: converter, verifier, search

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Render an image to ASCII (three modes: luminance, structure, edge)
python -m asciiart.cli render path/to/image.png --cols 80 --mode structure

# Rasterize ASCII text back to an image and score it against the source
python -m asciiart.cli verify path/to/image.png path/to/output.txt

# Non-neural upper baseline: simulated annealing directly on the character grid
python -m asciiart.cli anneal path/to/image.png --cols 80 --rows 24
```

See `docs/design.md` for the SSIM-per-cell formula, the edge-alignment term,
the anneal move set and schedule, and the reward-hacking risks the verifier
is designed to close off. See `docs/roadmap.md` for the phased build order
(converter, verifier + anneal, constrained decoding, RLVR/GRPO on Sophia,
writeup) and the acceptance criteria for each phase.

## Legacy word-OCR pipeline

The original project in this repo: generate synthetic ASCII-art renderings
of random words with `pyfiglet`, and train a small Transformer encoder to
read the word back out of the ASCII art. Kept as a self-contained pipeline;
unrelated to the image/prompt-to-ASCII problems above beyond sharing a
repo and a general interest in ASCII as a discrete grid representation.

1. **Data generation** (`ascii_generator.py`): renders random words with a
   FIGlet font and writes `data/dataset.jsonl`.
2. **Dataset prep** (`data_prep.py`): builds glyph and word-character
   vocabularies, writes an `npz` archive plus `vocabs.json`.
3. **Model** (`model.py`): Transformer encoder over the flattened ASCII-art
   token sequence, with a per-position classifier over `MAX_WORD_LEN`
   character slots.
4. **Training** (`train.py`): cross-entropy over each character position;
   reports per-position and exact-match accuracy; saves the best checkpoint.
5. **Evaluation** (`evaluate.py`): loads a checkpoint, reports accuracy on
   held-out or freshly generated data, and can print qualitative examples.

### Quick start

```bash
# 1. Generate data (adjust --num-samples as desired)
python ascii_generator.py --num-samples 5000 --out-dir data

# 2. Prepare dataset
python data_prep.py --data-file data/dataset.jsonl --out npz/art_dataset.npz

# 3. Train model
python train.py --data npz/art_dataset.npz --epochs 15 --d-model 192

# 4. Evaluate
python evaluate.py --data npz/art_dataset.npz --checkpoint checkpoints/best.pt --samples 10
```

Customization: fonts via `--font` in `ascii_generator.py` (run
`pyfiglet --list_fonts` for options); word length and character set in
`config.py`; model depth/width via training arguments.

File formats: `data/dataset.jsonl` is one JSON object per line
(`{"word": ..., "art": ..., "font": ...}`); `npz/art_dataset.npz` holds `X`
and `Y` int64 arrays plus lengths; `vocabs.json` holds `input_vocab` and
`target_vocab`.

## Roadmap

Phased build order, acceptance criteria, and current status live in
`docs/roadmap.md`. Short version: P0 deterministic converter, P1 verifier
and annealing baseline, P2 constrained decoding, P3 RLVR/GRPO on a small
open model on ALCF Sophia, P4 writeup.

## License

MIT
