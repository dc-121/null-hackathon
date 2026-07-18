# null-hackathon

Project built for the null hackathon.

## Status

Just getting started.

## Getting started

```bash
git clone https://github.com/dc-121/null-hackathon.git
cd null-hackathon
```

## Real-time emotion recognition (`emotion_webcam.py`)

State-of-the-art real-time facial emotion recognition from the webcam:
YuNet face detection + landmark alignment → DDAMFN++ (AffectNet-8, 64.7%),
optionally ensembled with HSEmotion, run via ONNX Runtime.

Model weights are **not** committed — they download automatically on first
run (into this folder, gitignored). Use an isolated venv, not conda:

```bash
python3 -m venv venv
./venv/bin/python -m pip install "opencv-python<5" hsemotion-onnx onnxruntime torch numpy
./venv/bin/python emotion_webcam.py   # first run downloads weights; press 'q' to quit
```

Speed/accuracy knobs are at the top of `emotion_webcam.py`
(`USE_ENSEMBLE`, `USE_TTA`, `EMOTION_EVERY`, `SMOOTHING`).

## License

MIT — see [LICENSE](./LICENSE).
