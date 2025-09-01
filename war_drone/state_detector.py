# war_drone/state_detector.py
"""
配置驱动的模板状态识别器（War Drone）
- 掩码（mask）匹配：支持 PNG alpha 或 *_mask.png（白=比较，黑=忽略）
- 边缘预处理（Canny）：降低纯色块误匹配
- ROI 尺寸 & 偏移：逐状态可覆盖
- 多模板组合：max / and_min_top2
- 动态扩展状态：从 configs/config.json5.extra_states 读取

依赖：OpenCV(cv2), numpy, json5
"""

from __future__ import annotations

import os
import enum
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import json5


# --------------------- 数据结构 ---------------------

@dataclass
class DetectedState:
    name: str
    score: float
    loc: Tuple[int, int]  # 模板左上角在整屏中的像素坐标
    template: str         # 触发模板文件名


class States(str, enum.Enum):
    LIST = "list"
    PREBATTLE = "prebattle"
    COMBAT = "combat"
    SETTLEMENT = "settlement"
    UNKNOWN = "unknown"
# （扩展状态以字符串加入识别，不新增到枚举）


# --------------------- 工具函数 ---------------------

def _bytes_to_bgr(img_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

def _crop_roi(img: np.ndarray, center_xy: Tuple[int, int], half_w: int, half_h: int) -> Tuple[np.ndarray, Tuple[int, int]]:
    """以 center 为中心裁 ROI，返回 roi 以及其相对整屏的左上角偏移 (ox, oy)"""
    H, W = img.shape[:2]
    cx, cy = center_xy
    x1 = max(0, cx - half_w); y1 = max(0, cy - half_h)
    x2 = min(W, cx + half_w); y2 = min(H, cy + half_h)
    return img[y1:y2, x1:x2].copy(), (x1, y1)

def _load_template_and_mask(path: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    读取模板及可选掩码：
    - 若存在同名 *_mask.png，则读取为 mask（灰度>0→255；=0→0）
    - 否则若模板是 RGBA PNG，则 alpha>0 → 255 作为 mask
    - 否则 mask=None
    返回 (tmpl_bgr, mask_u8 或 None)
    """
    root, _ = os.path.splitext(path)
    mask_path = f"{root}_mask.png"
    mask = None
    if os.path.exists(mask_path):
        m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if m is not None:
            _, mask = cv2.threshold(m, 1, 255, cv2.THRESH_BINARY)

    tmpl = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if tmpl is None:
        return None, None

    # RGBA → BGR + alpha
    if tmpl.ndim == 3 and tmpl.shape[2] == 4:
        bgr = cv2.cvtColor(tmpl, cv2.COLOR_BGRA2BGR)
        if mask is None:
            alpha = tmpl[:, :, 3]
            _, mask = cv2.threshold(alpha, 1, 255, cv2.THRESH_BINARY)
        return bgr, mask

    # 普通 BGR 或灰度
    if tmpl.ndim == 3:
        return tmpl, mask
    if tmpl.ndim == 2:
        bgr = cv2.cvtColor(tmpl, cv2.COLOR_GRAY2BGR)
        return bgr, mask

    return None, None


# --------------------- 识别器 ---------------------

class TemplateStateDetector:
    def __init__(
        self,
        cfg_path: str = "configs/config.json5",
        templates_dir: str = "templates",
        roi_half_size: Optional[Dict[States, Tuple[int, int]]] = None,  # 基础四状态默认 ROI
        method: str = "CCORR_NORMED",   # CCORR_NORMED/SQDIFF(_NORMED)/CCOEFF_NORMED
        default_thresh: float = 0.85,   # 置信度阈值
        use_edges: bool = True,         # 全局：是否做边缘预处理
        use_mask: bool = True,          # 全局：是否使用 mask 匹配
    ):
        self.cfg = json5.load(open(cfg_path, "r", encoding="utf-8"))
        self.wh = (self.cfg["screen"]["width"], self.cfg["screen"]["height"])
        self.coords = self.cfg["coords"]
        self.extra_states_cfg = self.cfg.get("extra_states", [])
        self.templates_dir = templates_dir
        self.default_thresh = float(default_thresh)
        self.use_edges = bool(use_edges)
        self.use_mask = bool(use_mask)

        # OpenCV 模板匹配方法映射
        _mm = {
            "CCORR_NORMED": cv2.TM_CCORR_NORMED,
            "SQDIFF": cv2.TM_SQDIFF,
            "SQDIFF_NORMED": cv2.TM_SQDIFF_NORMED,
            "CCOEFF_NORMED": cv2.TM_CCOEFF_NORMED,
        }
        self.method = _mm.get(method.upper(), cv2.TM_CCORR_NORMED)

        # 基础四状态 ROI 缺省（半宽/半高，像素）
        if roi_half_size is None:
            roi_half_size = {
                States.LIST:       (300, 180),
                States.PREBATTLE:  (180, 120),  # 降低与 list 混淆
                States.COMBAT:     (260, 260),  # 稍大，容忍战斗内漂移
                States.SETTLEMENT: (280, 180),
            }
        self.roi_half_size = roi_half_size

        # 基础四状态：模板与锚点（字符串 key）
        base_states = ["list", "prebattle", "combat", "settlement"]
        self.templates: Dict[str, List[str]] = {
            "list":       [os.path.join(templates_dir, "btn_list_start.png")],
            "prebattle":  [os.path.join(templates_dir, "btn_pre_start.png")],
            "combat":     [os.path.join(templates_dir, "btn_support_icon.png")],
            "settlement": [os.path.join(templates_dir, "btn_collect.png")],
        }
        self.anchor_keys: Dict[str, str] = {
            "list":       "list_start",
            "prebattle":  "pre_start",
            "combat":     "support3",
            "settlement": "collect",
        }

        # 组合策略（多模板如何合成分数）
        # - "max": 取最高
        # - "and_min_top2": 取前两高中的较低者（需要至少两张模板同时高分）
        self.combine_mode: Dict[str, str] = {
            "list": "max", "prebattle": "max", "combat": "max", "settlement": "max"
        }

        # 每状态是否启用边缘预处理（None=继承全局）
        self.use_edges_per_state: Dict[str, Optional[bool]] = {
            "combat": False  # 经验：支援图标大色块，边缘会削弱信息
        }

        # 每状态 ROI 尺寸/偏移（字符串 key）
        self.roi_half_size_per_state: Dict[str, Tuple[int,int]] = {
            "list": self.roi_half_size[States.LIST],
            "prebattle": self.roi_half_size[States.PREBATTLE],
            "combat": self.roi_half_size[States.COMBAT],
            "settlement": self.roi_half_size[States.SETTLEMENT],
        }
        self.roi_offset_pct: Dict[str, Tuple[float,float]] = {
            "list": (0.0, 0.0),
            "prebattle": (0.0, 0.0),
            "combat": (0.0, 0.0),
            "settlement": (0.0, 0.0),
        }

        # 注入扩展状态（配置驱动）
        for s in self.extra_states_cfg:
            name = s["name"]
            self.anchor_keys[name] = s["anchor"]
            self.templates[name] = [os.path.join(templates_dir, t) for t in s.get("templates", [])]
            if "roi_half_size" in s:
                self.roi_half_size_per_state[name] = (int(s["roi_half_size"][0]), int(s["roi_half_size"][1]))
            if "roi_offset_pct" in s:
                self.roi_offset_pct[name] = (float(s["roi_offset_pct"][0]), float(s["roi_offset_pct"][1]))
            self.use_edges_per_state[name] = s.get("use_edges", None)
            self.combine_mode[name] = s.get("combine_mode", "max")

        # 最终识别顺序：基础四状态 + 扩展状态（按配置定义顺序）
        self.state_order: List[str] = base_states + [s["name"] for s in self.extra_states_cfg]

    # ---------- 预处理 ----------

    def _prep(self, bgr: np.ndarray) -> np.ndarray:
        """边缘预处理：Canny；关闭则直接返回原图。"""
        g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        g = cv2.GaussianBlur(g, (3, 3), 0)
        e = cv2.Canny(g, 60, 120)
        return e

    def _pct_to_px(self, p) -> Tuple[int, int]:
        return int(p[0] * self.wh[0]), int(p[1] * self.wh[1])

    # ---------- 匹配 ----------

    def _best_matches_in_state(self, img_bgr: np.ndarray, state: Union[States, str]) -> List[Tuple[float, Tuple[int, int], str]]:
        """
        返回该状态下各模板最佳匹配列表 [(score, loc, tpath), ...]
        - score：越大越好（若用 SQDIFF(_NORMED)，已转换为 1 - min）
        - loc：匹配到的左上角（整屏坐标）
        - tpath：模板路径
        """
        key = state.value if isinstance(state, States) else state

        # ROI 中心 = 锚点坐标 + 相对偏移
        anchor = self.anchor_keys[key]
        cx, cy = self._pct_to_px(self.coords[anchor])
        offx, offy = self.roi_offset_pct.get(key, (0.0, 0.0))
        cx += int(offx * self.wh[0]); cy += int(offy * self.wh[1])

        half_w, half_h = self.roi_half_size_per_state.get(key, (220, 180))
        roi_bgr, (ox, oy) = _crop_roi(img_bgr, (cx, cy), half_w, half_h)

        # 是否对本状态启用边缘
        ue = self.use_edges_per_state.get(key, None)
        use_edges_state = self.use_edges if ue is None else ue
        roiX = self._prep(roi_bgr) if use_edges_state else roi_bgr

        results: List[Tuple[float, Tuple[int, int], str]] = []
        for tpath in self.templates.get(key, []):
            if not os.path.exists(tpath):
                continue
            tmpl_bgr, mask = _load_template_and_mask(tpath)
            if tmpl_bgr is None:
                continue
            tmplX = self._prep(tmpl_bgr) if use_edges_state else tmpl_bgr

            # 尺寸检查
            if tmplX.shape[0] > roiX.shape[0] or tmplX.shape[1] > roiX.shape[1]:
                continue

            # 是否启用 mask（方法需支持）
            supports_mask = self.method in (cv2.TM_CCORR_NORMED, cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED)
            use_mask = (self.use_mask and (mask is not None) and supports_mask)

            if use_mask:
                res = cv2.matchTemplate(roiX, tmplX, self.method, mask=mask)
            else:
                res = cv2.matchTemplate(roiX, tmplX, self.method)

            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            if self.method in (cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED):
                score = 1.0 - float(min_val)
                loc = (min_loc[0] + ox, min_loc[1] + oy)
            else:
                score = float(max_val)
                loc = (max_loc[0] + ox, max_loc[1] + oy)

            results.append((score, loc, tpath))

        return results

    # 兼容旧测试：返回单个最佳匹配
    def _best_match_in_state(self, img_bgr: np.ndarray, state: Union[States, str]):
        matches = self._best_matches_in_state(img_bgr, state)
        if not matches:
            return None
        score, loc, tpath = max(matches, key=lambda x: x[0])
        return DetectedState(name=(state.value if isinstance(state, States) else state),
                             score=float(score), loc=loc, template=os.path.basename(tpath))

    # ---------- 预测 ----------

    def predict(self, img_bytes: Optional[bytes] = None, img_bgr: Optional[np.ndarray] = None, margin: float = 0.12) -> DetectedState:
        """
        返回最可能的状态；若最高分低于阈值或领先优势不足，则 UNKNOWN。
        """
        assert img_bytes is not None or img_bgr is not None
        if img_bgr is None:
            img_bgr = _bytes_to_bgr(img_bytes)

        candidates: List[DetectedState] = []

        for st_name in self.state_order:
            matches = self._best_matches_in_state(img_bgr, st_name)
            if not matches:
                continue

            mode = self.combine_mode.get(st_name, "max")
            if mode == "and_min_top2" and len(matches) >= 2:
                # 需要至少两张模板都高分：取前两高的较低者作为该状态分数
                top2 = sorted([s for (s, _, _) in matches], reverse=True)[:2]
                score = min(top2)
                score1, loc, tpath = max(matches, key=lambda x: x[0])
            else:
                score, loc, tpath = max(matches, key=lambda x: x[0])

            candidates.append(DetectedState(name=st_name, score=score, loc=loc, template=os.path.basename(tpath)))

        if not candidates:
            return DetectedState(name=States.UNKNOWN.value, score=0.0, loc=(0, 0), template="")

        top2 = sorted(candidates, key=lambda d: d.score, reverse=True)[:2]
        best = top2[0]

        if best.score < self.default_thresh:
            return DetectedState(name=States.UNKNOWN.value, score=best.score, loc=best.loc, template=best.template)

        if len(top2) >= 2 and (best.score - top2[1].score) < margin:
            return DetectedState(name=States.UNKNOWN.value, score=best.score, loc=best.loc, template=best.template)

        return best
