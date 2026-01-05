# -*- coding: utf-8 -*-
from __future__ import annotations
import re, json5, cv2
import numpy as np
from typing import List, Dict, Any, Tuple
import easyocr

# ============== 工具 ==============
def crop_rel(img, rel: List[float], wh: Tuple[int, int]):
    """
    rel: [cx, cy, w, h] (相对坐标 0~1)，以中心点裁剪
    """
    cx, cy, w, h = rel
    W, H = wh
    ww, hh = int(w * W), int(h * H)
    x1 = max(0, int(cx * W - ww / 2)); y1 = max(0, int(cy * H - hh / 2))
    x2 = min(W, x1 + ww);             y2 = min(H, y1 + hh)
    return img[y1:y2, x1:x2]

def preprocess_for_ocr(tile_bgr, max_width=800, binarize=False):
    """
    轻处理：限制宽度、转灰、去噪、CLAHE；可选 OTSU 二值化。
    保持和 probe 一致，避免 conf 波动。
    """
    h, w = tile_bgr.shape[:2]
    if w > max_width:
        scale = max_width / float(w)
        tile_bgr = cv2.resize(tile_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    g = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.medianBlur(g, 3)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g = clahe.apply(g)
    if binarize:
        g = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    return g

def match_ncc(big: np.ndarray, small: np.ndarray) -> float:
    if small.shape[0] > big.shape[0] or small.shape[1] > big.shape[1]:
        return 0.0
    res = cv2.matchTemplate(big, small, cv2.TM_CCORR_NORMED)
    _, maxv, _, _ = cv2.minMaxLoc(res)
    return float(maxv)

def _norm_text(s: str) -> str:
    """
    文本规范化：去半角/全角空格、收尾空格；（中文不区分大小写，这里不做大小写转换也可）
    """
    if not isinstance(s, str):
        s = str(s)
    s = s.replace(" ", "").replace("\u3000", "")
    return s.strip()


# ============== 判定器 ==============
class OcrStateDetector:
    """
    OCR 为主的 UI 状态判定器。
    - OCR 命中一条规则 +1 分
    - 每个 aux_template 达阈值 +0.5 分
    - 最高分为最终状态；同分或最高分<=0 → "unknown"
    """
    def __init__(self, cfg_path="configs/ocr_states.json5"):
        self.cfg = json5.load(open(cfg_path, "r", encoding="utf-8"))
        self.WH = (self.cfg["screen"]["width"], self.cfg["screen"]["height"])
        self.rois: Dict[str, List[float]] = self.cfg["rois"]
        self.states: List[Dict[str, Any]] = self.cfg["states"]

        # 加载模板
        self.templates: Dict[str, np.ndarray] = {}
        for t in self.cfg.get("templates", []):
            img = cv2.imread(t["path"])
            if img is not None:
                self.templates[t["name"]] = img

        # OCR 引擎缓存（按语言复用）
        self._readers: Dict[str, easyocr.Reader] = {}

    # ---------- OCR ----------
    def _get_reader(self, lang: str):
        if lang not in self._readers:
            # 如需尝试 GPU，改成 gpu=True（但你当前环境是 CPU 版 torch）
            self._readers[lang] = easyocr.Reader([lang], gpu=False)
        return self._readers[lang]

    def _texts_in_roi(self, img, roi_key: str, lang: str) -> List[Tuple[str, float]]:
        tile = crop_rel(img, self.rois[roi_key], self.WH)
        tile = preprocess_for_ocr(tile, max_width=800, binarize=False)
        reader = self._get_reader(lang)
        res = reader.readtext(tile)
        out: List[Tuple[str, float]] = []
        for _, txt, conf in res:
            out.append((txt, float(conf)))
        return out

    # ---------- 模板 ----------
    def _aux_template_score(self, img, roi_key: str, tmpl_name: str) -> float:
        if tmpl_name not in self.templates:
            return 0.0
        tile = crop_rel(img, self.rois[roi_key], self.WH)
        gray_big = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
        gray_sml = cv2.cvtColor(self.templates[tmpl_name], cv2.COLOR_BGR2GRAY)
        return match_ncc(gray_big, gray_sml)

    # ---------- 规则评估（支持 contains / all_contains / regex） ----------
    def _eval_ocr_rule(self, texts_norm: List[Tuple[str, float]], rule: Dict[str, Any]) -> bool:
        """
        texts_norm: [(normalized_text, conf), ...]  —— 已规范化
        rule: {contains? | all_contains? | regex?, min_conf?}
        """
        min_conf = float(rule.get("min_conf", 0.5))

        # 任意包含：命中其中一个关键字为真
        if "contains" in rule:
            keywords = [ _norm_text(k) for k in rule["contains"] ]
            for t, c in texts_norm:
                if c >= min_conf and any(kw in t for kw in keywords):
                    return True
            return False

        # 且包含：必须全部关键字都命中（可分散在不同文本行）
        if "all_contains" in rule:
            kws = [ _norm_text(k) for k in rule["all_contains"] ]
            hit = {kw: False for kw in kws}
            for kw in kws:
                for t, c in texts_norm:
                    if c >= min_conf and (kw in t):
                        hit[kw] = True
                        break
            return all(hit.values())

        # 正则匹配：任意文本命中正则为真
        if "regex" in rule:
            pat = re.compile(rule["regex"])
            for t, c in texts_norm:
                if c >= min_conf and pat.search(t):
                    return True
            return False

        # 未配置任何条件 -> 不加分
        return False

    # ---------- 预测 ----------
    def predict(self, img_bgr: np.ndarray) -> Tuple[str, Dict[str, Any]]:
        """
        返回 (state_name, debug_info)
        打分策略：
          - OCR contains/regex/all_contains 命中 +1
          - 每个 aux_template 达阈值 +0.5
          - 取最高分；同分或<=0 返回 unknown
        """
        scores: Dict[str, float] = {}
        dbg: Dict[str, Any] = {}

        # 方便按模板名查 ROI
        tmpl_roi_map = { t["name"]: t["roi"] for t in self.cfg.get("templates", []) }

        for st in self.states:
            name = st["name"]
            s = 0.0
            details = {"ocr_hits": [], "tmpl_hits": [], "ocr_raw": {}}

            # OCR 规则
            for rule in st.get("ocr", []):
                roi = rule["roi"]
                lang = rule.get("lang", "ch_sim")
                texts = self._texts_in_roi(img_bgr, roi, lang)  # [(txt, conf), ...]

                # 记录原始与规范化文本，便于调试
                normed = [(_norm_text(t), c) for (t, c) in texts]
                details["ocr_raw"].setdefault(roi, {
                    "raw": [(t, c) for (t, c) in texts],
                    "norm": normed,
                })

                if self._eval_ocr_rule(normed, rule):
                    # 找到用于展示的第一条命中项（用于 dbg）
                    hit_txt = ""
                    hit_conf = 0.0
                    if "contains" in rule:
                        keys = [ _norm_text(k) for k in rule["contains"] ]
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

            # 模板加分
            for aux in st.get("aux_templates", []):
                tmpl = aux["template"]
                thr = float(aux.get("min_score", 0.7))
                roi_key = tmpl_roi_map.get(tmpl)
                if roi_key:
                    sc = self._aux_template_score(img_bgr, roi_key, tmpl)
                    if sc >= thr:
                        s += 0.5
                        details["tmpl_hits"].append((tmpl, sc))

            scores[name] = s
            dbg[name] = details

        # 取最高分
        best = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        if not best or best[0][1] <= 0:
            return "unknown", {"scores": scores, "details": dbg}
        if len(best) >= 2 and best[0][1] == best[1][1]:
            return "unknown", {"scores": scores, "details": dbg}
        return best[0][0], {"scores": scores, "details": dbg}
