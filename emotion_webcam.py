"""
State-of-the-art *real-time* facial emotion recognition from the webcam.

Pipeline:
  1. Face detection + 5-point landmarks .... YuNet (OpenCV DNN)
  2. Face alignment ........................ similarity transform to the
     ArcFace 112x112 template (matches the models' training distribution)
  3. Emotion recognition ................... DDAMFN++ (AffectNet-8, 64.7% val),
     optionally ensembled with HSEmotion enet_b2_8
  4. Temporal smoothing .................... exponential moving average

Both models run through ONNX Runtime on all CPU cores. DDAMFN++ is exported to
ONNX on first run (needs torch that once); after that torch isn't touched.
DDAMFN++ weights/code: https://github.com/SainingZhang/DDAMFN

Run:
    ./venv/bin/python emotion_webcam.py

Press 'q' to quit.

Speed knobs (top of file): USE_ENSEMBLE off -> ~2x faster (DDAMFN++ alone is
already SOTA-tier). USE_TTA adds a flip pass for a small accuracy bump.
"""

import os
import urllib.request
import cv2
import numpy as np
import onnxruntime as ort

# --- Config -----------------------------------------------------------------

CAMERA_INDEX = 0

HERE = os.path.dirname(os.path.abspath(__file__))
YUNET_MODEL = os.path.join(HERE, "face_detection_yunet_2023mar.onnx")
DDAMFN_CKPT = os.path.join(HERE, "affecnet8_epoch25_acc0.6469.pth")
DDAMFN_ONNX = os.path.join(HERE, "ddamfn8_affectnet.onnx")

USE_ENSEMBLE = False  # blend DDAMFN++ with HSEmotion enet_b2_8 (~2x slower, +~1% acc)
USE_TTA = False       # add a horizontal-flip pass per model (~2x slower)
DDAMFN_WEIGHT = 0.65  # ensemble weight for DDAMFN++ (the stronger model)
EMOTION_EVERY = 1     # run emotion nets every N frames (detection is every frame)
SMOOTHING = 0.55      # EMA factor for the primary face (0 = off, ->1 = sticky)

# Canonical label order == DDAMFN++'s output order.
CANON = ["Neutral", "Happiness", "Sadness", "Surprise",
         "Fear", "Disgust", "Anger", "Contempt"]

EMOTION_COLORS = {
    "Anger": (0, 0, 255), "Contempt": (0, 128, 255), "Disgust": (0, 200, 120),
    "Fear": (255, 0, 255), "Happiness": (0, 220, 0), "Neutral": (200, 200, 200),
    "Sadness": (255, 120, 0), "Surprise": (0, 255, 255),
}

# Weights are not committed to the repo; they download on first run.
ASSET_URLS = {
    YUNET_MODEL: ("https://github.com/opencv/opencv_zoo/raw/main/models/"
                  "face_detection_yunet/face_detection_yunet_2023mar.onnx"),
    DDAMFN_CKPT: ("https://raw.githubusercontent.com/SainingZhang/DDAMFN/main/"
                  "DDAMFN%2B%2B/checkpoints_ver2.0/"
                  "affecnet8_epoch25_acc0.6469.pth"),
}

ARCFACE_REF = np.array([
    [38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
    [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)


def ensure_assets():
    """Download model weights on first run (kept out of git)."""
    for path, url in ASSET_URLS.items():
        if os.path.isfile(path):
            continue
        print(f"Downloading {os.path.basename(path)} (one-time)...")
        tmp = path + ".part"
        urllib.request.urlretrieve(url, tmp)
        os.replace(tmp, path)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# --- DDAMFN++ (ONNX) --------------------------------------------------------

def export_ddamfn_onnx():
    """One-time export of the DDAMFN++ checkpoint to ONNX (needs torch)."""
    import torch
    from networks.DDAM import DDAMNet

    print("First run: exporting DDAMFN++ to ONNX (one-time)...")
    model = DDAMNet(num_class=8, num_head=2, pretrained=False)
    try:
        ckpt = torch.load(DDAMFN_CKPT, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(DDAMFN_CKPT, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    class Wrap(torch.nn.Module):
        def __init__(self, net):
            super().__init__()
            self.net = net

        def forward(self, x):
            return self.net(x)[0]  # logits only

    torch.onnx.export(
        Wrap(model), torch.randn(1, 3, 112, 112), DDAMFN_ONNX,
        input_names=["input"], output_names=["logits"], opset_version=17)
    print("Exported", DDAMFN_ONNX)


def make_session(path):
    so = ort.SessionOptions()
    so.intra_op_num_threads = os.cpu_count() or 4
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])


def load_ddamfn():
    if not os.path.isfile(DDAMFN_ONNX):
        export_ddamfn_onnx()
    sess = make_session(DDAMFN_ONNX)
    return sess, sess.get_inputs()[0].name


def load_hsemotion():
    try:
        from hsemotion_onnx.facial_emotions import HSEmotionRecognizer
    except Exception as e:
        print("HSEmotion unavailable, running DDAMFN++ only:", e)
        return None, None
    fer = HSEmotionRecognizer(model_name="enet_b2_8")
    hse_labels = [fer.idx_to_class[i] for i in range(len(fer.idx_to_class))]
    hse_to_canon = np.array([CANON.index(lbl) for lbl in hse_labels])
    return fer, hse_to_canon


def _softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def ddamfn_probs(sess, iname, aligned_bgr):
    rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    x = rgb.transpose(2, 0, 1)[None]  # 1x3x112x112
    logits = sess.run(None, {iname: x})[0][0]
    probs = _softmax(logits)
    if USE_TTA:
        xf = rgb[:, ::-1, :].transpose(2, 0, 1)[None].copy()
        probs = 0.5 * (probs + _softmax(sess.run(None, {iname: xf})[0][0]))
    return probs  # already in CANON order


def hse_probs(fer, hse_to_canon, rgb_crop):
    imgs = [rgb_crop]
    if USE_TTA:
        imgs.append(rgb_crop[:, ::-1, :].copy())
    acc = np.zeros(8, dtype=np.float32)
    for im in imgs:
        _, scores = fer.predict_emotions(im, logits=False)
        p = np.asarray(scores, dtype=np.float32)
        p = p / p.sum()
        canon = np.zeros(8, dtype=np.float32)
        canon[hse_to_canon] = p
        acc += canon
    return acc / len(imgs)


# --- Detection & alignment --------------------------------------------------

def make_detector():
    if os.path.isfile(YUNET_MODEL):
        det = cv2.FaceDetectorYN.create(YUNET_MODEL, "", (320, 320))
        det.setScoreThreshold(0.6)
        return ("yunet", det)
    print("YuNet model missing; using haar cascade (no alignment, weaker).")
    return ("haar", cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"))


def detect(detector, frame):
    kind, det = detector
    H, W = frame.shape[:2]
    out = []
    if kind == "yunet":
        det.setInputSize((W, H))
        _, faces = det.detect(frame)
        if faces is not None:
            for f in faces:
                x, y, w, h = f[:4].astype(int)
                re, le, nose = f[4:6], f[6:8], f[8:10]
                rm, lm = f[10:12], f[12:14]
                pts = np.array([le, re, nose, lm, rm], dtype=np.float32)
                out.append({"box": (x, y, w, h), "pts": pts})
    else:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        for (x, y, w, h) in det.detectMultiScale(gray, 1.1, 5, minSize=(80, 80)):
            out.append({"box": (int(x), int(y), int(w), int(h)), "pts": None})

    clean = []
    for d in out:
        x, y, w, h = d["box"]
        x, y = max(0, x), max(0, y)
        w, h = min(w, W - x), min(h, H - y)
        if w > 20 and h > 20:
            d["box"] = (x, y, w, h)
            clean.append(d)
    return clean


def align_112(frame, det):
    if det["pts"] is not None:
        M, _ = cv2.estimateAffinePartial2D(det["pts"], ARCFACE_REF, method=cv2.LMEDS)
        if M is not None:
            return cv2.warpAffine(frame, M, (112, 112), flags=cv2.INTER_LINEAR)
    x, y, w, h = det["box"]
    crop = frame[y:y + h, x:x + w]
    if crop.size == 0:
        return None
    return cv2.resize(crop, (112, 112), interpolation=cv2.INTER_LINEAR)


def bbox_crop(frame, det, margin=0.2):
    H, W = frame.shape[:2]
    x, y, w, h = det["box"]
    mx, my = int(w * margin), int(h * margin)
    x0, y0 = max(0, x - mx), max(0, y - my)
    x1, y1 = min(W, x + w + mx), min(H, y + h + my)
    crop = frame[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)


# --- Overlay ----------------------------------------------------------------

def draw(frame, box, emotion, conf, top3, color):
    x, y, w, h = box
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    label = f"{emotion} {conf * 100:.0f}%"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    cv2.rectangle(frame, (x, y - th - 12), (x + tw + 8, y), color, -1)
    cv2.putText(frame, label, (x + 4, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
    bx, by = x, y + h + 6
    for name, p in top3:
        c = EMOTION_COLORS.get(name, (200, 200, 200))
        cv2.rectangle(frame, (bx, by), (bx + int(90 * p), by + 12), c, -1)
        cv2.putText(frame, f"{name[:4]} {p * 100:.0f}%", (bx + 94, by + 11),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        by += 16


# --- Main -------------------------------------------------------------------

def main():
    ensure_assets()
    print("Loading models...")
    sess, iname = load_ddamfn()
    fer, hse_to_canon = (load_hsemotion() if USE_ENSEMBLE else (None, None))
    detector = make_detector()

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError(
            "Could not open the camera. Check macOS camera permissions "
            "(System Settings > Privacy & Security > Camera) and CAMERA_INDEX.")

    frame_count = 0
    smoothed = None
    cached = []
    fps, tprev = 0.0, cv2.getTickCount()

    print("Running... press 'q' to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_count += 1

        dets = detect(detector, frame)
        dets.sort(key=lambda d: d["box"][2] * d["box"][3], reverse=True)

        if (frame_count % EMOTION_EVERY == 0) or not cached:
            cached = []
            for i, d in enumerate(dets):
                aligned = align_112(frame, d)
                if aligned is None:
                    continue
                probs = ddamfn_probs(sess, iname, aligned)
                if fer is not None:
                    rgb = bbox_crop(frame, d)
                    if rgb is not None:
                        hp = hse_probs(fer, hse_to_canon, rgb)
                        probs = DDAMFN_WEIGHT * probs + (1 - DDAMFN_WEIGHT) * hp
                        probs = probs / probs.sum()

                if i == 0:
                    if smoothed is None or len(smoothed) != len(probs):
                        smoothed = probs
                    else:
                        smoothed = SMOOTHING * smoothed + (1 - SMOOTHING) * probs
                    probs = smoothed

                order = probs.argsort()[::-1]
                emotion = CANON[order[0]]
                top3 = [(CANON[j], float(probs[j])) for j in order[:3]]
                cached.append((d["box"], emotion, float(probs[order[0]]),
                               top3, EMOTION_COLORS.get(emotion, (0, 255, 0))))
        else:
            for i, d in enumerate(dets[:len(cached)]):
                _, e, c, t3, col = cached[i]
                cached[i] = (d["box"], e, c, t3, col)

        for (box, emotion, conf, top3, color) in cached:
            draw(frame, box, emotion, conf, top3, color)
        if not cached:
            cv2.putText(frame, "No face detected", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)

        tnow = cv2.getTickCount()
        fps = 0.9 * fps + 0.1 * (cv2.getTickFrequency() / (tnow - tprev))
        tprev = tnow
        cv2.putText(frame, f"{fps:.1f} FPS", (frame.shape[1] - 120, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow("SOTA emotion recognition (press q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
