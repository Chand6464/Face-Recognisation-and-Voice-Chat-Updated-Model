"""
NumPy port of face-api.js's age and gender network (TinyXception).
No TensorFlow, no PyTorch, no JavaScript: only numpy, opencv (and safetensors for .safetensors files).

Used by voice_bot_gui.py, but you can also test it alone:
    python faceapi_agegender.py --cam [weights]          live webcam test (q = quit)
        keys:  - / =  smaller / bigger margin around the face
               m      swap Man/Woman (if they come out reversed)
        an extra window shows the 112x112 picture the model really sees
    python faceapi_agegender.py --verify manifest.json [photo.jpg]
        compares the original face-api weights with the safetensors weights
    python faceapi_agegender.py photo.jpg [weights]      one photo

Weights (in models/ next to this file), first one found is used:
    age_gender_model-weights_manifest.json + age_gender_model-shard1   (original face-api.js)
    age_gender_model.safetensors                                        (third-party copy)

Optional better face detector: put face_detection_yunet_2023mar.onnx (OpenCV zoo)
into the same models/ folder and it is used automatically instead of the Haar detector.
"""
import os
import sys
import time
from collections import deque

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(HERE, "models")
SAFETENSORS_WEIGHTS = os.path.join(MODELS_DIR, "age_gender_model.safetensors")
TFJS_MANIFEST = os.path.join(MODELS_DIR, "age_gender_model-weights_manifest.json")
# Original face-api.js weights are preferred when present.
DEFAULT_WEIGHTS = TFJS_MANIFEST if os.path.isfile(TFJS_MANIFEST) else SAFETENSORS_WEIGHTS

WEIGHTS_URL = "https://huggingface.co/CognitiveOS/vision/resolve/main/age_gender_model.safetensors"
ORIGINAL_BASE = "https://raw.githubusercontent.com/justadudewhohacks/face-api.js/master/weights/"
ORIGINAL_FILES = ("age_gender_model-weights_manifest.json", "age_gender_model-shard1")
YUNET_FILE = "face_detection_yunet_2023mar.onnx"

INPUT_SIZE = 112
MEAN_RGB = np.array([122.782, 117.001, 104.298], dtype=np.float32)   # same numbers as face-api.js
# face-api.js reads output index 0 of the gender softmax as the probability of MALE.
# If your results come out with men and women swapped, change this to 1 (or press m in --cam).
MALE_INDEX = 0


# ----------------------------------------------------------------------------
# weights
# ----------------------------------------------------------------------------
def download_original(folder=MODELS_DIR):
    """Download the original face-api.js age/gender weights (about 420 KB) into models/."""
    import requests
    os.makedirs(folder, exist_ok=True)
    for name in ORIGINAL_FILES:
        dest = os.path.join(folder, name)
        if os.path.isfile(dest) and os.path.getsize(dest) > 1000:
            continue
        r = requests.get(ORIGINAL_BASE + name, timeout=60)
        r.raise_for_status()
        with open(dest, "wb") as f:
            f.write(r.content)
    return os.path.join(folder, ORIGINAL_FILES[0])


def download_weights(path=None):
    """Return a usable weights path, downloading if it is missing.
    Missing .json -> original face-api.js files. Missing .safetensors -> third-party copy."""
    path = path or DEFAULT_WEIGHTS
    if os.path.isfile(path):
        return path
    if path.lower().endswith(".json"):
        return download_original(os.path.dirname(path))
    import requests
    os.makedirs(os.path.dirname(path), exist_ok=True)
    r = requests.get(WEIGHTS_URL, timeout=60)
    r.raise_for_status()
    with open(path, "wb") as f:
        f.write(r.content)
    return path


def _put(params, name, arr):
    node = params
    parts = name.split("/")
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = np.ascontiguousarray(arr, dtype=np.float32)


def load_safetensors(path):
    """Read a .safetensors file into nested dicts: params['entry_flow']['conv_in']['filters']"""
    from safetensors import safe_open
    params = {}
    with safe_open(path, framework="np") as f:
        for key in f.keys():
            _put(params, key, f.get_tensor(key))
    return params


def load_tfjs(manifest_path):
    """Read the original TensorFlow.js weights (manifest .json + shard files), incl. uint8 compression."""
    import json
    base = os.path.dirname(os.path.abspath(manifest_path))
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    params = {}
    for group in manifest:
        blob = b""
        for name in group["paths"]:
            with open(os.path.join(base, name), "rb") as f:
                blob += f.read()
        offset = 0
        for spec in group["weights"]:
            shape = spec["shape"]
            count = int(np.prod(shape)) if shape else 1
            q = spec.get("quantization")
            if q:
                dt = np.dtype(q["dtype"]).newbyteorder("<")
                raw = np.frombuffer(blob, dtype=dt, count=count, offset=offset)
                if "scale" in q:
                    arr = raw.astype(np.float32) * np.float32(q["scale"]) + np.float32(q["min"])
                else:                                           # float16 compression: no scale
                    arr = raw.astype(np.float32)
            else:
                dt = np.dtype(spec.get("dtype", "float32")).newbyteorder("<")
                raw = np.frombuffer(blob, dtype=dt, count=count, offset=offset)
                arr = raw.astype(np.float32)
            offset += count * dt.itemsize
            _put(params, spec["name"], arr.reshape(shape))
        if offset != len(blob):
            print(f"Warning: manifest used {offset} bytes but shard files hold {len(blob)}.")
    return params


def load_params(path):
    if path.lower().endswith(".json"):
        return load_tfjs(path)
    return load_safetensors(path)


def _flatten(params, prefix=""):
    out = {}
    for k, v in params.items():
        name = f"{prefix}/{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten(v, name))
        else:
            out[name] = v
    return out


# ----------------------------------------------------------------------------
# layers (all arrays are height x width x channels, like TensorFlow.js)
# ----------------------------------------------------------------------------
def relu(x):
    return np.maximum(x, 0)


def conv_in(x, filters, bias):
    """3x3 convolution, stride 2, 'same' padding (TensorFlow puts the extra pad bottom/right)."""
    h, w, _ = x.shape
    oh, ow = h // 2, w // 2
    xp = np.pad(x, ((0, 1), (0, 1), (0, 0)))
    cols = [xp[i:i + 2 * oh:2, j:j + 2 * ow:2, :] for i in range(3) for j in range(3)]
    patches = np.concatenate(cols, axis=2)                          # (oh, ow, 27)
    kh, kw, cin, cout = filters.shape
    out = patches.reshape(-1, kh * kw * cin) @ filters.reshape(kh * kw * cin, cout)
    return out.reshape(oh, ow, cout) + bias


def conv1x1_stride2(x, filters, bias):
    xs = x[::2, ::2, :]
    h, w, c = xs.shape
    out = xs.reshape(-1, c) @ filters[0, 0]
    return out.reshape(h, w, -1) + bias


def separable_conv(x, p):
    """Depthwise 3x3 (stride 1, same) then pointwise 1x1, then bias."""
    h, w, c = x.shape
    k = p["depthwise_filter"][:, :, :, 0]                           # (3, 3, C)
    xp = np.pad(x, ((1, 1), (1, 1), (0, 0)))
    y = xp[0:h, 0:w] * k[0, 0]
    for i in range(3):
        for j in range(3):
            if i or j:
                y += xp[i:i + h, j:j + w] * k[i, j]
    pw = p["pointwise_filter"][0, 0]                                # (Cin, Cout)
    y = (y.reshape(-1, c) @ pw).reshape(h, w, -1)
    return y + p["bias"]


def max_pool_3x3_stride2(x):
    h, w, _ = x.shape
    oh, ow = (h + 1) // 2, (w + 1) // 2
    xp = np.pad(x, ((0, 1), (0, 1), (0, 0)), constant_values=-np.inf)
    out = None
    for i in range(3):
        for j in range(3):
            part = xp[i:i + 2 * oh:2, j:j + 2 * ow:2, :]
            out = part if out is None else np.maximum(out, part)
    return out


def reduction_block(x, p, activate_input=True):
    out = relu(x) if activate_input else x
    out = separable_conv(out, p["separable_conv0"])
    out = separable_conv(relu(out), p["separable_conv1"])
    out = max_pool_3x3_stride2(out)
    return out + conv1x1_stride2(x, p["expansion_conv"]["filters"], p["expansion_conv"]["bias"])


def main_block(x, p):
    out = separable_conv(relu(x), p["separable_conv0"])
    out = separable_conv(relu(out), p["separable_conv1"])
    out = separable_conv(relu(out), p["separable_conv2"])
    return out + x


class AgeGenderNet:
    def __init__(self, weights):
        """weights: a path (.json = original face-api.js, .safetensors) or an already loaded params dict."""
        self.p = weights if isinstance(weights, dict) else load_params(weights)
        self.n_main = sum(1 for k in self.p["middle_flow"] if k.startswith("main_block_"))
        self.last_input = None                                      # the 112x112 RGB picture of the last call

    def forward(self, face_rgb):
        """face_rgb: 112x112x3 float32, values 0..255, RGB. Returns (age_years, male_probability)."""
        p = self.p
        x = (face_rgb - MEAN_RGB) / 256.0
        x = relu(conv_in(x, p["entry_flow"]["conv_in"]["filters"], p["entry_flow"]["conv_in"]["bias"]))
        x = reduction_block(x, p["entry_flow"]["reduction_block_0"], activate_input=False)
        x = reduction_block(x, p["entry_flow"]["reduction_block_1"])
        for i in range(self.n_main):
            x = main_block(x, p["middle_flow"][f"main_block_{i}"])
        x = reduction_block(x, p["exit_flow"]["reduction_block"])
        x = relu(separable_conv(x, p["exit_flow"]["separable_conv"]))
        pooled = x.mean(axis=(0, 1))                                # 7x7 average pool -> 512 numbers
        age = float(pooled @ p["fc"]["age"]["weights"][:, 0] + p["fc"]["age"]["bias"][0])
        logits = pooled @ p["fc"]["gender"]["weights"] + p["fc"]["gender"]["bias"]
        e = np.exp(logits - logits.max())
        probs = e / e.sum()
        return age, float(probs[MALE_INDEX])

    def predict_crop(self, crop_bgr):
        """Prepare a face crop the way face-api.js does: square (centered, black padding), 112x112."""
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        m = max(h, w)
        square = np.zeros((m, m, 3), dtype=np.uint8)
        y0, x0 = (m - h) // 2, (m - w) // 2
        square[y0:y0 + h, x0:x0 + w] = rgb
        interp = cv2.INTER_AREA if m > INPUT_SIZE else cv2.INTER_LINEAR
        small = cv2.resize(square, (INPUT_SIZE, INPUT_SIZE), interpolation=interp)
        self.last_input = small
        return self.forward(small.astype(np.float32))


# ----------------------------------------------------------------------------
# face finder + analyzer used by the GUI
# ----------------------------------------------------------------------------
class FaceApiAnalyzer:
    """analyze(frame_bgr) returns a list shaped like DeepFace.analyze's output:
       [{'region': {x,y,w,h}, 'age': int, 'gender': {'Woman': %, 'Man': %}, 'dominant_emotion': str|None}]
       Raises ValueError('Face could not be detected.') when there is no face.

    Smoothing (like predict-age.com): for the biggest face, age and gender are the AVERAGE of the
    last `smooth` predictions. The average restarts after `reset_after` seconds without a face,
    or when you call reset() (do this on the "New person" button).
    Extra keys on every result: raw_age, raw_man (this frame only), samples (how many frames averaged)."""

    def __init__(self, weights_path=None, box_padding=0.1, emotion_fn=None, smooth=10, reset_after=2.0):
        weights_path = weights_path or DEFAULT_WEIGHTS
        self.weights_path = weights_path
        self.net = AgeGenderNet(weights_path)
        self.pad = float(box_padding)
        self.emotion_fn = emotion_fn
        self.smooth_n = int(smooth)
        self.reset_after = float(reset_after)
        self._ages = deque(maxlen=max(self.smooth_n, 1))
        self._men = deque(maxlen=max(self.smooth_n, 1))
        self._last_seen = 0.0
        self.detector_name = "Haar"
        self._yunet = None
        yunet_path = os.path.join(os.path.dirname(weights_path), YUNET_FILE)
        if os.path.isfile(yunet_path) and hasattr(cv2, "FaceDetectorYN"):
            try:
                self._yunet = cv2.FaceDetectorYN.create(yunet_path, "", (320, 320), 0.8, 0.3, 5000)
                self.detector_name = "YuNet"
            except Exception as e:
                print("YuNet could not be loaded, using Haar instead:", e)
        self._haar = cv2.CascadeClassifier(
            os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml"))
        if self._haar.empty():
            raise RuntimeError("OpenCV's Haar face file could not be loaded.")
        print(f"Face detector: {self.detector_name}   weights: {os.path.basename(weights_path)}")

    @property
    def samples(self):
        return len(self._ages)

    def reset(self):
        self._ages.clear()
        self._men.clear()

    def _detect(self, frame):
        if self._yunet is not None:
            h, w = frame.shape[:2]
            self._yunet.setInputSize((w, h))
            _, faces = self._yunet.detect(frame)
            if faces is None:
                return []
            return [tuple(int(v) for v in f[:4]) for f in faces]
        gray = cv2.equalizeHist(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        rects = self._haar.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
        return [tuple(int(v) for v in r) for r in rects]

    def analyze(self, frame_bgr):
        now = time.time()
        if now - self._last_seen > self.reset_after:
            self.reset()                                            # nobody for a while -> new person
        H, W = frame_bgr.shape[:2]
        results = []
        for (x, y, w, h) in self._detect(frame_bgr):
            x, y = max(x, 0), max(y, 0)
            w, h = min(w, W - x), min(h, H - y)
            if w < 20 or h < 20:
                continue
            px, py = int(w * self.pad), int(h * self.pad)
            crop = frame_bgr[max(y - py, 0):min(y + h + py, H), max(x - px, 0):min(x + w + px, W)]
            if crop.size == 0:
                continue
            age, male = self.net.predict_crop(crop)
            emotion = None
            if self.emotion_fn is not None:
                try:
                    emotion = self.emotion_fn(crop)
                except Exception:
                    emotion = None
            results.append({
                "region": {"x": x, "y": y, "w": w, "h": h},
                "age": int(round(max(age, 1.0))),
                "gender": {"Woman": (1.0 - male) * 100.0, "Man": male * 100.0},
                "dominant_emotion": emotion,
                "raw_age": max(age, 1.0),
                "raw_man": male * 100.0,
                "samples": 1,
            })
        if not results:
            raise ValueError("Face could not be detected.")
        self._last_seen = now
        if self.smooth_n > 1:
            main = max(results, key=lambda r: r["region"]["w"] * r["region"]["h"])
            self._ages.append(main["raw_age"])
            self._men.append(main["raw_man"])
            avg_age = sum(self._ages) / len(self._ages)
            avg_man = sum(self._men) / len(self._men)
            main["age"] = int(round(avg_age))
            main["gender"] = {"Woman": 100.0 - avg_man, "Man": avg_man}
            main["samples"] = len(self._ages)
        return results


# ----------------------------------------------------------------------------
# stand-alone tests
# ----------------------------------------------------------------------------
def _label(face):
    g = face["gender"]
    who = "Woman" if g["Woman"] > g["Man"] else "Man"
    return f"{who} {max(g.values()):.0f}%, age {face['age']}"


def _test_photo(path, weights=None):
    img = cv2.imread(path)
    if img is None:
        print("Could not read the image:", path)
        return
    an = FaceApiAnalyzer(download_weights(weights), smooth=0)
    try:
        for f in an.analyze(img):
            print(_label(f), f["region"])
    except ValueError as e:
        print(e)


def _verify(manifest, photo=None):
    """Are the safetensors weights the same as the original face-api.js weights?"""
    if not os.path.isfile(manifest):
        print("Manifest not found:", manifest)
        return
    a = _flatten(load_safetensors(download_weights(SAFETENSORS_WEIGHTS)))
    b = _flatten(load_tfjs(manifest))
    print(f"safetensors: {len(a)} tensors    original (tfjs): {len(b)} tensors")
    only_a, only_b = sorted(set(a) - set(b)), sorted(set(b) - set(a))
    if only_a or only_b:
        print("Names only in safetensors:", only_a[:5], "...")
        print("Names only in original   :", only_b[:5], "...")
    diffs, bad_shape = [], []
    for k in sorted(set(a) & set(b)):
        if a[k].shape != b[k].shape:
            bad_shape.append((k, a[k].shape, b[k].shape))
        else:
            diffs.append((float(np.abs(a[k] - b[k]).max()), k))
    for k, sa, sb in bad_shape[:5]:
        print("SHAPE DIFFERS:", k, sa, sb)
    diffs.sort(reverse=True)
    print("Biggest weight differences (tiny = just compression):")
    for d, k in diffs[:5]:
        print(f"   {d:.5f}  {k}")
    worst = diffs[0][0] if diffs else 999.0
    net_a, net_b = AgeGenderNet(load_safetensors(SAFETENSORS_WEIGHTS)), AgeGenderNet(load_tfjs(manifest))
    rng = np.random.default_rng(0)
    age_gap = male_gap = 0.0
    for _ in range(5):
        x = rng.uniform(0, 255, (INPUT_SIZE, INPUT_SIZE, 3)).astype(np.float32)
        (ag1, m1), (ag2, m2) = net_a.forward(x), net_b.forward(x)
        age_gap, male_gap = max(age_gap, abs(ag1 - ag2)), max(male_gap, abs(m1 - m2))
    print(f"Output gap on 5 random pictures: age {age_gap:.2f} years, male-probability {male_gap:.3f}")
    if not bad_shape and not only_a and not only_b and worst < 0.02 and age_gap < 2.0:
        print("RESULT: SAME weights. The third-party file is fine, so look at the crop / detector.")
    else:
        print("RESULT: DIFFERENT weights. Use the original (.json) weights.")
    if photo:
        print("\nOriginal weights on your photo:")
        _test_photo(photo, manifest)


def _show_model_input(an):
    img = an.net.last_input
    if img is not None:
        big = cv2.resize(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), (336, 336), interpolation=cv2.INTER_NEAREST)
        cv2.imshow("what the model sees (112x112)", big)


def _open_camera():
    """Try camera 0 and 1 with different Windows backends; return one that really gives a picture."""
    for index in (0, 1):
        for name, backend in (("DSHOW", cv2.CAP_DSHOW), ("MSMF", cv2.CAP_MSMF), ("default", cv2.CAP_ANY)):
            cap = cv2.VideoCapture(index, backend)
            if cap.isOpened():
                for _ in range(30):                      # cameras often need a moment to start
                    ok, _frame = cap.read()
                    if ok:
                        print(f"Camera {index} works with {name}.")
                        return cap
                    time.sleep(0.1)
            cap.release()
    print("No camera gave a picture. Check: another app (or an old python window) is using it, "
          "Windows Settings > Privacy > Camera allows desktop apps, or the lens cover is closed.")
    return None


def _test_camera(weights=None):
    global MALE_INDEX
    an = FaceApiAnalyzer(download_weights(weights), smooth=10)
    cap = _open_camera()
    if cap is None:
        return
    print("Live test running.  q = quit   - / = margin   m = swap Man/Woman")
    failed = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            failed += 1
            if failed > 50:
                print("The camera stopped giving pictures. Close other apps that use the camera and try again.")
                break
            time.sleep(0.05)
            continue
        failed = 0
        t0 = time.time()
        text, raw = "No face", ""
        try:
            faces = an.analyze(frame)
            f = max(faces, key=lambda r: r["region"]["w"] * r["region"]["h"])
            man = f["gender"]["Man"]
            who = "Man" if man > 50 else "Woman"
            text = f"{who} {max(man, 100 - man):.0f}%, age {f['age']}  (avg of {f['samples']})"
            raw = f"this frame: {'Man' if f['raw_man'] > 50 else 'Woman'} {f['raw_man']:.0f}% man, age {f['raw_age']:.0f}"
            r = f["region"]
            cv2.rectangle(frame, (r["x"], r["y"]), (r["x"] + r["w"], r["y"] + r["h"]), (0, 255, 0), 2)
        except ValueError:
            pass
        ms = (time.time() - t0) * 1000
        cv2.putText(frame, f"{text}   ({ms:.0f} ms)", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        cv2.putText(frame, raw, (10, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1)
        cv2.putText(frame, f"margin {an.pad:.2f}  ({an.detector_name})", (10, frame.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.imshow("faceapi test (q = quit)", frame)
        _show_model_input(an)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        elif k == ord("-"):
            an.pad = max(0.0, round(an.pad - 0.05, 2)); an.reset()
        elif k == ord("="):
            an.pad = min(0.8, round(an.pad + 0.05, 2)); an.reset()
        elif k == ord("m"):
            MALE_INDEX = 1 - MALE_INDEX; an.reset()
            print("Gender index is now", MALE_INDEX, "(set MALE_INDEX to this if it looks right)")
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        print(__doc__)
    elif args[0] == "--cam":
        _test_camera(args[1] if len(args) > 1 else None)
    elif args[0] == "--verify":
        if len(args) < 2:
            print("Usage: python faceapi_agegender.py --verify models\\age_gender_model-weights_manifest.json [photo.jpg]")
        else:
            _verify(args[1], args[2] if len(args) > 2 else None)
    else:
        _test_photo(args[0], args[1] if len(args) > 1 else None)