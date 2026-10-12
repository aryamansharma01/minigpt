# MiniGPT

A small PyTorch character-level GPT trained on the bundled `input.txt` corpus. The implementation includes causal self-attention, grouped-query attention, optional rotary positional embeddings, and a simple training loop.

## Setup

Requires Python and the packages listed in `requirements.txt`.

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Train

Run from the project root so the script can read `input.txt`:

```sh
python chargpt.py
```

The default run trains for 600 iterations and writes checkpoints, generated samples, and training logs to `out/chargpt/`. Configuration values can be overridden from the command line, for example:

```sh
python chargpt.py --trainer.max_iters=600 --trainer.batch_size=64
```

The default model is a six-layer, 192-dimensional character-level GPT. It uses learned positional embeddings and standard multi-head causal attention; GQA and RoPE can be enabled through the model configuration.