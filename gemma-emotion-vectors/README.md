# Gemma emotion-vector trace

Run `google/gemma-4-E4B-it` locally and measure its layer-28 residual-stream
activation against the nine vectors published by
[`rain1955/emotion-vector-replication`](https://huggingface.co/rain1955/emotion-vector-replication).

The published directions are `afraid`, `angry`, `calm`, `desperate`, `guilty`,
`happy`, `loving`, `sad`, and `surprised`. All nine are scored after every
generated token; `--top-k` only controls how many are printed.

## Setup

```bash
uv sync --python 3.12
```

The first run downloads the Gemma checkpoint, approximately 15 GB, and the
published vector archive into the Hugging Face cache.

## Run once

```bash
uv run python scripts/emotion_trace.py \
  --top-k 9 \
  "A difficult test keeps failing and I am running out of time. What should I do?"
```

## Keep Gemma loaded

Interactive mode avoids reloading the checkpoint for every prompt:

```bash
uv run python scripts/emotion_trace.py \
  --interactive \
  --timings \
  --max-new-tokens 32 \
  --top-k 9
```

Enter `/quit` or press Ctrl-D to exit.

The output values are cosine alignments with published directions, not emotion
probabilities. Read [How it works](docs/HOW_IT_WORKS.md) for the complete data
flow, measured performance, and interpretation limits.
