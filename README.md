# MiniGPT

A compact, decoder-only GPT language model implemented in PyTorch. It trains on raw text at the character level and generates continuations one character at a time. The implementation is based on the minGPT/GPT-2 style architecture, with optional Rotary Positional Embeddings (RoPE) and Grouped-Query Attention (GQA), plus a GPT-623 model variant.

The code and experiments are discussed in the accompanying article: [Coding a mini GPT: RoPE, causal attention, and GQA](https://aryaman01.vercel.app/blog/mini-gpt).

## Architecture

![Annotated MiniGPT transformer architecture](https://aryaman01.vercel.app/images/blog/mini-gpt/gpt623_architecture.png)

The model embeds each character, adds either learned position embeddings or RoPE, and passes the sequence through pre-normalized Transformer blocks. Each block has a causal attention layer and a feed-forward network, both wrapped in residual connections. A final linear head predicts a score for every character in the vocabulary at every position.

The default `gpt623` variant uses:

- Bias-free linear projections in attention and the feed-forward network.
- RMSNorm, including per-head RMSNorm on queries and keys (QK-Norm).
- A bias-free SwiGLU feed-forward network.

Other `model_type` values use LayerNorm and a GELU feed-forward network with projection biases. The model uses attention and residual dropout during training.

### Attention Options

With equal query and key/value head counts, the model uses standard multi-head causal self-attention. Set `n_kv_head` lower than `n_query_head` to use GQA: query heads are grouped, and each group shares a key/value head. GQA reduces the size of the key/value projections. This implementation repeats the key/value heads for attention and does not implement a KV cache during generation, so it is a simple reference implementation rather than a cache-optimized inference engine.

By default, learned absolute position embeddings are used. Set `rope=True` to apply rotary position embeddings to queries and keys instead. RoPE avoids learned parameters tied to individual positions, but the model's usable context is still limited by `data.block_size`, which determines the causal mask and maximum accepted input length.

## Data and Objective

[`input.txt`](input.txt) contains the bundled Tiny Shakespeare text corpus (about 1.12 million characters and 65 unique characters). The dataset builds a character-to-ID vocabulary from the complete file. Each training example is a window of `block_size` characters; the target is the same window shifted forward by one character.

Windows are sampled randomly with replacement. The script does not create a separate validation or test split, so the reported training loss is not a held-out evaluation metric.

## Quickstart

```sh
git clone https://github.com/aryamansharma01/minigpt.git
cd minigpt
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python chargpt.py
```

The script automatically selects CUDA, Apple MPS, or CPU when available. Run it from the project root because it reads `input.txt` relative to the current directory.

## Default Run

`chargpt.py` configures a small model and a short context for experimentation:

| Setting | Default |
| --- | ---: |
| Transformer layers | 6 |
| Embedding width | 192 |
| Query heads | 6 |
| Key/value heads | 6 (standard multi-head attention) |
| Context length (`block_size`) | 16 characters |
| RoPE | Off (learned position embeddings) |
| Batch size | 64 |
| Training iterations | 600 |
| Learning rate | `5e-4` |
| Optimizer | AdamW |

The trainer also uses AdamW betas `(0.9, 0.95)`, weight decay `0.1`, and gradient clipping at `1.0`. The model's embedding, attention, and residual dropout rates default to `0.1`.

Override configuration values from the command line using `--section.name=value` syntax. For example, this runs a 256-character context with RoPE and four KV heads shared across six query heads:

```sh
python chargpt.py --data.block_size=256 --model.rope=True --model.n_kv_head=2
```

For GQA, `n_query_head` must be divisible by `n_kv_head`, and the embedding width must be divisible by the query-head count. RoPE also requires an even per-head width.

## Sampling and Artifacts

Every 200 training iterations, the script switches to evaluation mode and generates a 500-character continuation from `O God, O God!`, sampling from the top 10 next-character candidates. It then resumes training.

Run artifacts are written to `out/chargpt/` by default:

- `model.pt`: latest model weights.
- `config.json` and `args.txt`: run configuration and command-line arguments.
- `train_losses.json`: recorded training loss.
- `attention_computation_time.json`: recorded attention timing.

The output directory can be changed with `--system.work_dir`, using a quoted string value, for example `--system.work_dir='"./out/rope"'`.

## Experiment: RoPE and GQA

The plot below is from the accompanying article, not from a single default `python chargpt.py` run. It compares four configurations (baseline, RoPE, GQA, and RoPE + GQA), trained for 600 iterations at context length 16 and then continued for 600 iterations at context length 256.

![Training loss for four configurations across a context-length increase from 16 to 256](https://aryaman01.vercel.app/images/blog/mini-gpt/results_four_configs.png)

In this experiment, the RoPE curves finish at lower training loss after the context-length increase. The learned-position baseline and GQA-only runs show a larger transient loss increase at the switch, while the GQA and non-GQA curves are otherwise close at this model scale. These are training-curve observations from this setup, not a general benchmark or a held-out quality comparison.

## Repository Layout

```text
chargpt.py       Character dataset, run configuration, sampling, and training entry point
input.txt        Tiny Shakespeare training corpus
mingpt/model.py  GPT, attention variants, positional embeddings, and optimizer setup
mingpt/trainer.py
				 Generic PyTorch training loop
mingpt/utils.py  Configuration, seeding, and run logging helpers
requirements.txt Python dependencies
```

For a detailed walkthrough of the implementation and figures, see the [supporting blog post](https://aryaman01.vercel.app/blog/mini-gpt).