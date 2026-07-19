# How the Gemma emotion-vector trace works

This experiment applies published emotion directions directly to internal
activations from a locally running open model. It does not use sentiment
analysis, a second classifier model, or output-word matching.

## Inputs

- Model: `google/gemma-4-E4B-it`
- Internal location: residual-stream output of layer 28 out of 42
- Hidden width: 2,560
- Vector source: `rain1955/emotion-vector-replication`
- Emotions: `afraid`, `angry`, `calm`, `desperate`, `guilty`, `happy`,
  `loving`, `sad`, and `surprised`

The replication created each direction by generating emotion-specific stories,
averaging Gemma's layer-28 activations, subtracting the global mean across
emotions, and projecting out dominant components measured on neutral text.
This project downloads the resulting nine-vector archive as published; it does
not retrain, transfer, or expand those vectors.

## Runtime data flow

For each prompt, `scripts/emotion_trace.py` performs the following steps:

```text
prompt
  -> Gemma generates a deterministic response
  -> the complete prompt and response are replayed
  -> a hook captures layer-28 activation after every token
  -> each activation and published vector is normalized
  -> cosine alignment is calculated for all nine vectors
  -> the top-k directions are printed after each token
  -> 3-6 token phrases are standardized against local neutral calibration
  -> absolute and prompt-relative evidence are computed for all nine directions
  -> one strong, stable winner becomes an intensity-aware v3 audio tag
  -> audio is streamed to the local player as bytes arrive
```

`<before response>` is the activation after the final prompt token: the state
that predicts the first response token. Every later row is the state after the
displayed response token and helps predict what follows.

`--top-k` controls display only. The script calculates all nine scores at every
position.

## Selectable conversation conditioning

The conversation UI exposes two backend paths through a compact **Prompt /
Vector** toggle:

- **Prompt** preserves the original behavior: fused scores, reliable modality
  summaries, and a conservative response strategy are written into Gemma's
  instructions.
- **Vector** keeps emotion names, scores, modality summaries, and strategy
  labels out of the prompt. It combines the five shared layer-28 directions as
  `confidence * sum((score_i - 0.2) * vector_i)`, normalizes that direction,
  and adds a norm-relative residual edit to every token overlapping the user's
  final sentence.

Subtracting `0.2` makes a uniform five-way distribution produce no edit. The
fused confidence and distance from that uniform baseline reduce the applied
strength when evidence is weak or conflicting. The maximum configured edit is
20% of each selected token's residual norm; the effective ratio is usually
lower because it is multiplied by the blended direction's magnitude.

The intervention happens only during the initial prompt pass at layer 28. It is
not applied continuously to generated tokens, which reduces the risk of making
Gemma imitate an observed emotion. The response is still replayed afterward for
the normal nine-direction trace.

This is a real activation intervention but remains experimental: the published
vectors were derived as diagnostic directions, not validated causal controls.
The API therefore returns the selected mode, layer, target-token count,
direction magnitude, and applied residual ratio for each conversation.

## What a score means

A value such as `desperate=+0.0970` means that the current layer-28 activation
has positive cosine alignment with the published desperate direction. It does
not mean "9.7% desperate," and the nine values do not sum to one.

The replication was designed to recover emotion geometry from story-level
averages. It did not publish token-level neutral calibration. This project
therefore runs twelve neutral assistant sentences through the same model and
layer at startup and measures a local mean and standard deviation for each
direction. This makes an `angry` cosine comparable to its own neutral behavior,
without pretending all nine raw cosine scales are identical.

For a phrase, the compiler calculates two kinds of evidence:

- absolute evidence: how far its neutral z-score exceeds the activation floor;
- relative evidence: how much its z-score rose from the pre-response state.

The current deterministic transfer function is:

```text
absolute_i = max(z_i - 0.50, 0)
relative_i = 0.35 * max(delta_z_i - 0.25, 0)
evidence_i = absolute_i + relative_i
evidence_share_i = evidence_i / sum(evidence)
intensity_i = min(evidence_i / 6.0, 1.0)
```

Positive evidence is normalized across all nine directions for the diagnostic
display. These evidence shares are not calibrated emotion probabilities and are
not blended into the voice.

The speech compiler chooses exactly one direction per phrase. It emits no tag
when the strongest evidence is below `1.25`. Once a direction is active, a new
winner must beat its current-phrase evidence by both 20% and `0.50`; otherwise
the existing direction stays active. This hysteresis prevents tiny measurement
changes from making the voice flicker between tags while still allowing a real
emotional turn to switch the delivery.

Consequently:

- compare changes and spikes along one direction across nearby tokens;
- use controlled prompt contrasts when evaluating whether a direction responds;
- do not interpret the highest raw direction as a definitive emotional state;
- do not compare raw magnitudes across emotions as though they share a scale.

In local tests, `calm` had a large positive baseline across several prompts,
while `desperate` rose as a generated passage moved into time pressure. This
shows contextual movement along the direction; it is not a calibrated emotion
probability.

These are measurements of functional internal representations. They do not
establish consciousness, feelings, or subjective experience.

## Performance

The first run downloads roughly 15 GB into the Hugging Face cache. A measured
cached run on Apple Metal broke down as follows:

| Phase | Measured time |
| --- | ---: |
| Load Gemma into memory | 11.70 s |
| Generate eight tokens, including initial Metal warm-up | 5.52 s |
| Replay and score layer-28 activations | 0.09 s |

In persistent interactive mode, a warmed second prompt generated four tokens in
0.42 s and replayed/scored them in 0.07 s. Vector scoring is not the bottleneck;
model loading and autoregressive generation are.

With `--speak`, the selected direction chooses a low, medium, or extreme tag
from its evidence magnitude. An extreme direction can add one delivery action
such as `[shouts]`, `[laughing]`, or `[sobbing]`. Tags are inserted every few
tokens in one Eleven v3 request, allowing the voice to change within a single
response without stitching separate voices together.

The default raw delivery mode uses stability `0.25`. Lower stability widens
emotional range but also makes generations less predictable. The `safe` mode
keeps the same selected evidence and substitutes restrained tags; it does not
change or reclassify the underlying Gemma measurements.

## Commands

Run one prompt:

```bash
uv run python scripts/emotion_trace.py \
  --max-new-tokens 32 \
  --top-k 9 \
  "A difficult test keeps failing and I am running out of time. What should I do?"
```

Keep Gemma loaded for repeated prompts:

```bash
uv run python scripts/emotion_trace.py \
  --interactive \
  --timings \
  --max-new-tokens 32 \
  --top-k 9
```

## Attribution

- Emotion-vector code, data, and extracted vectors:
  `rain1955/emotion-vector-replication` (MIT license).
- Gemma model use remains subject to Google's Gemma terms.
