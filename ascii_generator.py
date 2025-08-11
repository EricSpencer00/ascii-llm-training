import argparse
import json
import random
from pathlib import Path
from typing import List

from pyfiglet import Figlet

from config import TARGET_CHARS, MIN_WORD_LEN, MAX_WORD_LEN, RANDOM_SEED

random.seed(RANDOM_SEED)


def random_word(min_len: int = MIN_WORD_LEN, max_len: int = MAX_WORD_LEN) -> str:
    L = random.randint(min_len, max_len)
    return ''.join(random.choice(TARGET_CHARS) for _ in range(L))


def gen_samples(num: int, font: str, fonts: List[str], multi_font: bool):
    samples = []
    if multi_font and not fonts:
        # Use a curated subset of fonts for variety
        fonts = [
            'standard', 'slant', '3-d', '3x5', '5lineoblique', 'alphabet', 'banner3-D',
            'doh', 'isometric1', 'letters', 'alligator', 'bubble', 'chunky', 'computer',
        ]
    for i in range(num):
        w = random_word()
        use_font = random.choice(fonts) if multi_font else font
        f = Figlet(font=use_font)
        art = f.renderText(w)
        samples.append({
            'word': w,
            'art': art.rstrip('\n'),
            'font': use_font
        })
    return samples


def main():
    p = argparse.ArgumentParser(description="Generate ASCII art dataset")
    p.add_argument('--num-samples', type=int, default=1000)
    p.add_argument('--out-dir', type=str, default='data')
    p.add_argument('--font', type=str, default='standard')
    p.add_argument('--multi-font', action='store_true')
    p.add_argument('--seed', type=int, default=RANDOM_SEED)
    args = p.parse_args()

    random.seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_file = out_dir / 'dataset.jsonl'

    samples = gen_samples(args.num_samples, args.font, [], args.multi_font)

    with open(dataset_file, 'w') as f:
        for s in samples:
            f.write(json.dumps(s) + '\n')

    print(f"Wrote {len(samples)} samples to {dataset_file}")

    # Optionally also save individual art files (disabled by default for space)


if __name__ == '__main__':
    main()
