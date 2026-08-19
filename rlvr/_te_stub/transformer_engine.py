# Shadow stub for `transformer_engine`.
#
# On sophia, the sophia-train venv is `--system-site-packages`, which leaks
# in the system conda environment's `transformer_engine` package. That
# package's compiled extension (`libtransformer_engine.so`) fails to
# `dlopen` on this system (undefined symbol against the loaded
# `libcublasLt.so`), which is irrelevant to us -- `peft` merely does an
# unconditional `import transformer_engine` at module load time (in
# `peft/tuners/lora/te.py`, to opportunistically support NVIDIA Transformer
# Engine LoRA variants) and has no try/except around the `dlopen` failure,
# so it takes the whole `import peft` down with it.
#
# Putting this directory ahead of the system site-packages on PYTHONPATH
# (see scripts/rlvr_sophia.pbs) makes `import transformer_engine` resolve to
# this empty stub instead: `peft.import_utils.is_te_pytorch_available()`
# checks `hasattr(transformer_engine, "pytorch")`, which is False here, so
# it correctly reports "not available" and peft proceeds without TE
# support -- which is all we need, since this project doesn't use TE.
