# Central configuration constants
MAX_WORD_LEN = 12  # maximum target word length (pad/truncate)
MIN_WORD_LEN = 3

# Allowed characters for target words (lowercase letters only for now)
TARGET_CHARS = list("abcdefghijklmnopqrstuvwxyz")

# Additional tokens for target vocab
PAD_TOKEN = "<pad>"
EOS_TOKEN = "<eos>"  # optional (not strictly used in fixed-length prediction)

# Input tokens: we derive from observed ascii art plus newline '↵' placeholder.
NEWLINE_TOKEN = "<nl>"
PAD_INPUT_TOKEN = "<pad>"

RANDOM_SEED = 42
