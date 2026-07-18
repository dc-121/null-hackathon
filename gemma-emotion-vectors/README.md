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

## Speak the measured emotion

The trace can convert phrase-level vector changes into ElevenLabs v3 audio
tags and stream the speech immediately through an expressive character voice:

```bash
uv run python scripts/emotion_trace.py \
  --speak \
  --interactive \
  --timings \
  --max-new-tokens 32
```

Enter the ElevenLabs key at the secure prompt, or expose it to the process as
`ELEVENLABS_API_KEY`. The script never writes the key, and `.env` files are
git-ignored.

For the local hackathon setup, the key can live in the git-ignored `.env` file.
Start the visual demo with one command:

```bash
make web
```

After Gemma loads, the browser opens at
[http://127.0.0.1:8766](http://127.0.0.1:8766). Each response shows a clickable
phrase timeline, all nine raw cosine scores, locally calibrated neutral
z-scores, prompt-relative changes, normalized evidence shares, the exact v3
tags, and streamed audio.

The terminal version remains available:

```bash
make chat
```

The speech layer does not take the highest raw direction. At startup, it
measures all nine vectors across neutral assistant sentences to estimate a
local mean and standard deviation for each direction. During a response it:

1. averages every 3–6 token phrase;
2. converts raw cosines to neutral-calibrated z-scores;
3. combines absolute evidence with movement from the pre-response state;
4. gates out weak evidence and chooses one strongest direction; and
5. converts its magnitude into one intensity tag plus an optional matching
   performance cue for extreme evidence.

All nine normalized evidence shares stay visible for inspection, but they are
not emotion probabilities and are not blended into the voice. A direction needs
at least `1.25` evidence to speak. To prevent rapid tag flicker, a new winner
must beat the current direction by both 20% and `0.50` evidence.

Raw theatrical delivery uses three intensity bands. The emotion and intensity
come from Gemma's calibrated evidence; only the vocabulary used to express that
evidence is predefined:

| Gemma vector | Low | Medium | Extreme |
| --- | --- | --- | --- |
| `afraid` | `[uneasy]` | `[fearful]` | `[terrified] [voice trembling]` |
| `angry` | `[irritated]` | `[angry]` | `[furious] [shouts]` |
| `calm` | `[gentle]` | `[calm]` | `[deeply calm] [slow]` |
| `desperate` | `[urgent]` | `[desperate]` | `[desperate] [pleading]` |
| `guilty` | `[uneasy]` | `[guilty]` | `[remorseful] [voice breaking]` |
| `happy` | `[pleased]` | `[excited]` | `[ecstatic] [laughing]` |
| `loving` | `[warmly]` | `[affectionately]` | `[lovingly] [tenderly]` |
| `sad` | `[somber]` | `[sad]` | `[crying] [sobbing]` |
| `surprised` | `[curious]` | `[surprised]` | `[shocked] [gasps]` |

Use `--delivery-mode safe` for restrained assistant speech. Raise
`--tts-stability` toward `0.5` if raw mode becomes too unpredictable.

For demo latency, keep `--interactive` enabled so Gemma and the HTTP connection
stay warm. `--timings` reports generation, activation replay, time to first
audio, and total time. Add `--save-audio demo.mp3` to retain the streamed file.
