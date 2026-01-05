# -*- coding: utf-8 -*-
import os, glob, cv2, pytest
from war_drone.ocr_state_detector import OcrStateDetector

DATASETS = {
    "list":       "tests/dataset/list/*.jpg",
    "prebattle":  "tests/dataset/prebattle/*.jpg",
    "combat":     "tests/dataset/combat/*.jpg",
    "settlement": "tests/dataset/settlement/*.jpg",
    "splash":     "tests/dataset/splash/*.jpg",
    "upgrade":    "tests/dataset/upgrade/*.jpg",
}

@pytest.fixture(scope="module")
def det():
    return OcrStateDetector(cfg_path="configs/ocr_states.json5")

@pytest.mark.parametrize("state,pattern", list(DATASETS.items()))
def test_pick_best(det, state, pattern):
    paths = sorted(glob.glob(pattern))
    if not paths:
        pytest.skip(f"no samples in {pattern}")
    for p in paths:
        img = cv2.imread(p); assert img is not None
        pred, dbg = det.predict(img)
        # 冒烟标准：预测命中的状态分数应≥其它状态（同分算 unknown）
        # 起步阶段不强制100%全对，让你先把 ROI/规则跑通
        ok = (pred == state)
        if not ok:
            print("\n[DBG]", p, "=>", pred, dbg["scores"])
        assert ok, f"{p}: expect={state}, got={pred}, scores={dbg['scores']}"
