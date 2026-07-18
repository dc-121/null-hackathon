# Third-party notices

Nullhack combines original integration work with open-source software, model
artifacts, and hosted APIs. This file records the pieces that are material to
the demo and must not be presented as original work by this repository.

## Human crowd simulation

The frontend under `experience/` is derived from
[`dc-121/null-hackathon`](https://github.com/dc-121/null-hackathon), snapshot
`8255987ee6744081dee42c3b428ab050fff2beb4`.

Copyright (c) 2026 Daniel Cerasi. Licensed under the MIT License. The complete
upstream notice is retained at `experience/LICENSE`.

## Facial-expression inference

The integrated face path uses HSEmotion ONNX and OpenCV's bundled Haar face
detector. It does not contain or download DDAMFN source or checkpoints.

- [`av-savchenko/hsemotion-onnx`](https://github.com/av-savchenko/hsemotion-onnx)
  is distributed under the Apache License 2.0.
- [OpenCV](https://github.com/opencv/opencv) is distributed under the Apache
  License 2.0. Its packaged frontal-face Haar cascade is used only to crop the
  face before HSEmotion inference; no separate face-detector checkpoint is
  committed or downloaded by this project.

The `raph/facial-recognition` branch of `dc-121/null-hackathon` also contains
DDAMFN-derived files and an AffectNet checkpoint whose upstream repository does
not provide a license. Those files are intentionally excluded from the
distributable integration branch.

## Gemma emotion vectors

The response model is `google/gemma-4-E4B-it`, subject to Google's Gemma terms.
Emotion directions come from
[`rain1955/emotion-vector-replication`](https://huggingface.co/rain1955/emotion-vector-replication),
whose published project is MIT-licensed. Nullhack measures cosine alignment to
those directions; it does not claim the resulting values are calibrated
probabilities or evidence of subjective experience.

## ElevenLabs

Speech-to-text and text-to-speech are hosted ElevenLabs API services. No
ElevenLabs model weights are distributed with this repository. Use of the API
is governed by the account owner's ElevenLabs plan and terms.
