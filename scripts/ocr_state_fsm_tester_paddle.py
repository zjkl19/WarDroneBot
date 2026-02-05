# scripts/ocr_state_fsm_tester_paddle.py
"""
Test OCR state FSM using PaddleOCR on a single screenshot.
Example:
  python -m scripts.ocr_state_fsm_tester_paddle --image captures/settlement.png --cfg configs/ocr_states_fsm.json5
"""

import argparse
import os
import re
import importlib.util
from typing import List, Tuple, Dict, Any

import cv2
import json5
import paddle
from paddleocr import PaddleOCR

# Windows: add Paddle DLL search path
if os.name == "nt":
    spec = importlib.util.find_spec("paddle")
    if spec and spec.submodule_search_locations:
        libs_dir = os.path.join(spec.submodule_search_locations[0], "libs")
        if os.path.isdir(libs_dir):
            os.add_dll_directory(libs_dir)

# Patch missing set_optimization_level (older paddle builds)
try:
    from paddle.base import libpaddle
    if hasattr(libpaddle, "AnalysisConfig") and not hasattr(libpaddle.AnalysisConfig, "set_optimization_level"):
        libpaddle.AnalysisConfig.set_optimization_level = lambda self, level: None
except Exception:
    pass


def crop_rel(img, rel: List[float], wh: Tuple[int, int]):
    cx, cy, w, h = rel
    W, H = wh
    ww, hh = int(w * W), int(h * H)
    x1 = max(0, int(cx * W - ww / 2))
    y1 = max(0, int(cy * H - hh / 2))
    x2 = min(W, x1 + ww)
    y2 = min(H, y1 + hh)
    return img[y1:y2, x1:x2]


def _norm_text(s: str) -> str:
    if not isinstance(s, str):
        s = str(s)
    s = s.replace(" ", "").replace("\u3000", "")
    return s.strip()


def _build_ocr(use_gpu: bool, det_dir=None, rec_dir=None, cls_dir=None):
    # Params aligned with paddleocr5070.py
    ocr_kwargs = {
        "use_angle_cls": True,
        "lang": "ch",
        "det_db_thresh": 0.3,
        "det_db_box_thresh": 0.5,
        "det_db_unclip_ratio": 1.6,
        "max_text_length": 50,
        "rec_image_shape": "3, 48, 320",
        "show_log": False,
    }
    if use_gpu:
        ocr_kwargs["use_gpu"] = True
        ocr_kwargs["gpu_mem"] = 8000

    if det_dir:
        ocr_kwargs["det_model_dir"] = det_dir
    if rec_dir:
        ocr_kwargs["rec_model_dir"] = rec_dir
    if cls_dir:
        ocr_kwargs["cls_model_dir"] = cls_dir

    try:
        return PaddleOCR(**ocr_kwargs)
    except Exception as e:
        # PaddleOCR 3.x may reject use_gpu; retry without it
        if "use_gpu" in str(e):
            ocr_kwargs.pop("use_gpu", None)
            ocr_kwargs.pop("gpu_mem", None)
            return PaddleOCR(**ocr_kwargs)
        raise


class PaddleStateDetector:
    def __init__(self, cfg_path: str, det_dir=None, rec_dir=None, cls_dir=None):
        self.cfg = json5.load(open(cfg_path, "r", encoding="utf-8"))
        self.WH = (self.cfg["screen"]["width"], self.cfg["screen"]["height"])
        self.rois: Dict[str, List[float]] = self.cfg["rois"]
        self.states: List[Dict[str, Any]] = self.cfg["states"]

        # Prefer GPU, fallback to CPU
        use_gpu = False
        if paddle.device.is_compiled_with_cuda():
            try:
                paddle.device.set_device("gpu")
                use_gpu = True
                print("[INFO] Paddle set_device -> gpu")
            except Exception as e:
                print(f"[WARN] Paddle set_device('gpu') failed, fallback cpu: {e}")
                paddle.device.set_device("cpu")
        else:
            paddle.device.set_device("cpu")

        self.ocr_reader = _build_ocr(use_gpu, det_dir=det_dir, rec_dir=rec_dir, cls_dir=cls_dir)

    def _texts_in_roi(self, img, roi_key: str) -> List[Tuple[str, float]]:
        tile = crop_rel(img, self.rois[roi_key], self.WH)
        if tile.size == 0:
            return []
        h, w = tile.shape[:2]
        if h < 16 or w < 32:
            return []
        try:
            res = self.ocr_reader.ocr(tile, det=True, rec=True)
        except Exception:
            return []
        if not res or not res[0]:
            return []
        out: List[Tuple[str, float]] = []
        for line in res[0]:
            if not line or len(line) < 2:
                continue
            txt, conf = line[1][0], float(line[1][1])
            out.append((txt, conf))
        return out

    def _eval_rule(self, texts_norm: List[Tuple[str, float]], rule: Dict[str, Any]) -> bool:
        min_conf = float(rule.get("min_conf", 0.5))
        if "contains" in rule:
            kws = [_norm_text(k) for k in rule["contains"]]
            for t, c in texts_norm:
                if c >= min_conf and any(kw in t for kw in kws):
                    return True
            return False
        if "all_contains" in rule:
            kws = [_norm_text(k) for k in rule["all_contains"]]
            hit = {kw: False for kw in kws}
            for kw in kws:
                for t, c in texts_norm:
                    if c >= min_conf and kw in t:
                        hit[kw] = True
                        break
            return all(hit.values())
        if "regex" in rule:
            pat = re.compile(rule["regex"])
            for t, c in texts_norm:
                if c >= min_conf and pat.search(t):
                    return True
            return False
        return False

    def predict(self, img_bgr):
        scores: Dict[str, float] = {}
        dbg: Dict[str, Any] = {}
        for st in self.states:
            name = st["name"]
            s = 0.0
            details = {"ocr_hits": [], "ocr_raw": {}}
            for rule in st.get("ocr", []):
                roi = rule["roi"]
                texts = self._texts_in_roi(img_bgr, roi)
                normed = [(_norm_text(t), c) for t, c in texts]
                details["ocr_raw"].setdefault(roi, {"raw": texts, "norm": normed})
                if self._eval_rule(normed, rule):
                    hit_txt, hit_conf = "", 0.0
                    if "contains" in rule:
                        keys = [_norm_text(k) for k in rule["contains"]]
                        for (t_raw, c_raw), (t_norm, c_norm) in zip(texts, normed):
                            if c_norm >= float(rule.get("min_conf", 0.5)) and any(k in t_norm for k in keys):
                                hit_txt, hit_conf = t_raw, c_raw
                                break
                    elif "all_contains" in rule:
                        hit_txt, hit_conf = "ALL_CONTAINS_OK", max((c for _, c in normed), default=0.0)
                    elif "regex" in rule:
                        hit_txt, hit_conf = "REGEX_OK", max((c for _, c in normed), default=0.0)
                    s += 1.0
                    details["ocr_hits"].append((roi, hit_txt, hit_conf))
            scores[name] = s
            dbg[name] = details

        best = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        if not best or best[0][1] <= 0:
            return "unknown", {"scores": scores, "details": dbg}
        if len(best) >= 2 and best[0][1] == best[1][1]:
            return "unknown", {"scores": scores, "details": dbg}
        return best[0][0], {"scores": scores, "details": dbg}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="Image path (png/jpg)")
    ap.add_argument("--cfg", default="configs/ocr_states_fsm.json5", help="FSM config path")
    args = ap.parse_args()

    assert os.path.exists(args.image), f"Image not found: {args.image}"
    assert os.path.exists(args.cfg), f"Config not found: {args.cfg}"

    img = cv2.imread(args.image)
    assert img is not None, f"Failed to read image: {args.image}"

    det = PaddleStateDetector(args.cfg)
    state, dbg = det.predict(img)

    print(f"PRED: {state}")
    print("SCORES:", {k: round(v, 2) for k, v in dbg["scores"].items()})

    for name, detail in dbg["details"].items():
        hits = detail.get("ocr_hits", [])
        if hits:
            print(f"\n[{name}]")
            for roi, txt, conf in hits:
                print(f"  OCR_HIT roi={roi} txt={txt} conf={conf:.2f}")

    print("\nRAW OCR (per ROI):")
    for name, detail in dbg["details"].items():
        raw = detail.get("ocr_raw", {})
        for roi, rec in raw.items():
            print(f"  [{name}] roi={roi}")
            for txt, conf in rec.get("raw", []):
                print(f"    RAW : {txt} (conf={conf:.2f})")
            for txt, conf in rec.get("norm", []):
                print(f"    NORM: {txt} (conf={conf:.2f})")


if __name__ == "__main__":
    main()
