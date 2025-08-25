"""
目的：
- “随机抖动”后点击像素坐标仍在屏幕内，避免越界点击
说明：
- tap_jitter_px 表示点击点允许的 ±像素随机偏移
- 这里重复多次，确保任何抖动都不会把点带出屏幕
"""
import json5
from war_drone.simple_bot import _pct_to_px  # 复用你的换算函数

CFG = "configs/config.json5"

def test_jitter_within_screen():
    cfg = json5.load(open(CFG, "r", encoding="utf-8"))
    wh = (cfg["screen"]["width"], cfg["screen"]["height"])
    jitter = cfg["random"]["tap_jitter_px"]

    # 随机从多个关键点采样（支援+开始/收集按钮），连跑 200 次
    pts = [
        cfg["coords"]["list_start"],
        cfg["coords"]["pre_start"],
        cfg["coords"]["collect"],
        cfg["coords"]["support1"],
        cfg["coords"]["support3"],
        cfg["coords"]["support6"],
    ]
    for _ in range(200):
        for p in pts:
            x,y = _pct_to_px(p, wh, jitter)
            assert 0 <= x < wh[0] and 0 <= y < wh[1], f"越界：({x},{y}) not in {wh}"
