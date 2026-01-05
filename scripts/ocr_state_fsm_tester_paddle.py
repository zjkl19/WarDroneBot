# scripts/ocr_state_fsm_tester_paddle.py
"""
使用 PaddleOCR 按 configs/ocr_states_fsm.json5 测试状态识别。
环境：需已安装 paddlepaddle + paddleocr。

用法示例（在装好 PaddleOCR 的虚拟环境内）：
  python -m scripts.ocr_state_fsm_tester_paddle --image ready.png --cfg configs/ocr_states_fsm.json5
"""
import argparse
import json
import os
import re
from typing import List, Tuple, Dict, Any

import cv2
import json5
from paddleocr import PaddleOCR


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


class PaddleStateDetector:
    def __init__(self, cfg_path: str, det_dir=None, rec_dir=None, cls_dir=None):
        self.cfg = json5.load(open(cfg_path, "r", encoding="utf-8"))
        self.WH = (self.cfg["screen"]["width"], self.cfg["screen"]["height"])
        self.rois: Dict[str, List[float]] = self.cfg["rois"]
        self.states: List[Dict[str, Any]] = self.cfg["states"]
        ocr_kwargs = {"use_angle_cls": True, "lang": "ch"}
        if det_dir:
            ocr_kwargs["det_model_dir"] = det_dir
        if rec_dir:
            ocr_kwargs["rec_model_dir"] = rec_dir
        if cls_dir:
            ocr_kwargs["cls_model_dir"] = cls_dir
        self.ocr_reader = PaddleOCR(**ocr_kwargs)

    def _texts_in_roi(self, img, roi_key: str) -> List[Tuple[str, float]]:
        tile = crop_rel(img, self.rois[roi_key], self.WH)
        if tile.size == 0:
            return []
        try:
            res = self.ocr_reader.ocr(tile, det=True, rec=True)
        except Exception:
            return []
        if not res or res[0] is None:
            return []
        out: List[Tuple[str, float]] = []
        for line in res[0]:
            if line is None or len(line) < 2:
                continue
            txt, conf = line[1][0], float(line[1][1])
            out.append((txt, conf))
        return out

    def _eval_rule(self, texts_norm: List[Tuple[str, float]], rule: Dict[str, Any]) -> bool:
        min_conf = float(rule.get("min_conf", 0.5))
        # contains: 任意关键字命中
        if "contains" in rule:
            kws = [_norm_text(k) for k in rule["contains"]]
            for t, c in texts_norm:
                if c >= min_conf and any(kw in t for kw in kws):
                    return True
            return False
        # all_contains: 全部关键字都命中
        if "all_contains" in rule:
            kws = [_norm_text(k) for k in rule["all_contains"]]
            hit = {kw: False for kw in kws}
            for kw in kws:
                for t, c in texts_norm:
                    if c >= min_conf and kw in t:
                        hit[kw] = True
                        break
            return all(hit.values())
        # regex
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
                    hit_txt = ""
                    hit_conf = 0.0
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
    ap.add_argument("--image", required=True, help="待检测的截屏路径（png/jpg）")
    ap.add_argument("--cfg", default="configs/ocr_states_fsm.json5", help="OCR 状态机配置路径")
    ap.add_argument("--det_dir", default=None, help="PaddleOCR det 模型目录")
    ap.add_argument("--rec_dir", default=None, help="PaddleOCR rec 模型目录")
    ap.add_argument("--cls_dir", default=None, help="PaddleOCR cls 模型目录")
    args = ap.parse_args()

    assert os.path.exists(args.image), f"找不到图像：{args.image}"
    assert os.path.exists(args.cfg), f"找不到配置：{args.cfg}"

    img = cv2.imread(args.image)
    assert img is not None, f"无法读取图像：{args.image}"

    det = PaddleStateDetector(args.cfg,
                              det_dir=args.det_dir,
                              rec_dir=args.rec_dir,
                              cls_dir=args.cls_dir)
    state, dbg = det.predict(img)
    print(f"PRED: {state}")
    print("SCORES:", {k: round(v, 2) for k, v in dbg["scores"].items()})

    for name, detail in dbg["details"].items():
        hits = detail.get("ocr_hits", [])
        if hits:
            print(f"\n[{name}]")
            for roi, txt, conf in hits:
                print(f"  OCR_HIT roi={roi} txt={txt} conf={conf:.2f}")

    print("\nRAW OCR (按 ROI):")
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
