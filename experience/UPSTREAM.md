# Upstream provenance

This experience is based on the human crowd simulation from
[`dc-121/null-hackathon`](https://github.com/dc-121/null-hackathon) at commit
`8255987ee6744081dee42c3b428ab050fff2beb4`.

The upstream frontend and its visual language were created by Daniel Cerasi and
are used under the MIT License retained in [`LICENSE`](./LICENSE). Nullhack's
changes connect that simulation to camera, microphone, transcription, Gemma
emotion-vector tracing, and ElevenLabs speech; they do not erase or replace the
upstream authorship.

The separate upstream facial-recognition branch was reviewed but is not copied
into this distributable tree because its DDAMFN-derived source and checkpoint
do not include a usable upstream license. The integrated face path uses the
properly licensed HSEmotion ONNX and OpenCV stack instead.
