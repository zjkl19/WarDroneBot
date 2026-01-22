# -*- coding: utf-8 -*-
"""
基于 PaddleOCR 的简易状态判定器，按 configs/ocr_states_fsm.json5 读取 ROI/规则。
"""
from __future__ import annotations
import importlib.util
import os
import re
from typing import List, Tuple, Dict, Any

import cv2
import json5

if os.name == "nt":
    spec = importlib.util.find_spec("paddle")
    if spec and spec.submodule_search_locations:
        libs_dir = os.path.join(spec.submodule_search_locations[0], "libs")
        if os.path.isdir(libs_dir):
            os.add_dll_directory(libs_dir)

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
