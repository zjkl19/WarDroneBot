# tests/test_vis_hotzones.py
"""
目的：
- 以“随机抖动最大区域”为准，在四个界面图上绘制热区矩形 + 中心点。
- 矩形尺寸 = 2 * tap_jitter_px（以配置中的随机抖动半径为边的一半，正方形）。
- 这样可直观看到：脚本所有“可能落点”的边界是否仍在按钮内。

输出（out_vis/）：
  hz_list_jitter.jpg
  hz_prebattle_jitter.jpg
  hz_combat_jitter.jpg
  hz_settlement_jitter.jpg

截图要求（JPG，分辨率与 config 一致：2670x1200 横屏）：
  tests/assets/list_screen.jpg
  tests/assets/prebattle_screen.jpg
  tests/assets/combat_screen.jpg
  tests/assets/settlement_screen.jpg

可调参数：
- EXTRA_MARGIN_PX：给可视化矩形额外膨胀/收缩（正数=更大，负数=更小），仅影响“图上的框”，不影响脚本逻辑。
"""

import os
import json5
import cv2
from tests.helpers.visual import draw_rect, draw_point, save_image

CFG = "configs/config.json5"
ASSETS = {
    "list":       "tests/assets/list_screen.jpg",
    "prebattle":  "tests/assets/prebattle_screen.jpg",
    "combat":     "tests/assets/combat_screen.jpg",
    "settlement": "tests/assets/settlement_screen.jpg",
}
OUTS = {
    "list":       "out_vis/hz_list_jitter.jpg",
    "prebattle":  "out_vis/hz_prebattle_jitter.jpg",
    "combat":     "out_vis/hz_combat_jitter.jpg",
    "settlement": "out_vis/hz_settlement_jitter.jpg",
}

# 仅用于可视化的额外边距（像素）。想看更宽松/更紧的包络可以改这里：
EXTRA_MARGIN_PX = 0  # 例如设为 6，会把矩形从 2*jitter 增大到 2*(jitter+6)


def pct_to_px(p, wh):
    return int(p[0] * wh[0]), int(p[1] * wh[1])


def _draw_centered_jitter_rect(img, x, y, jitter_px, label, color):
    """以 (x,y) 为中心，画边长 = 2*jitter_px 的正方形 + 中心点。"""
    side = max(2 * jitter_px + 2 * EXTRA_MARGIN_PX, 2)  # 最小2，避免0宽高
    draw_rect(img, x, y, side, side, text=f"{label} (±{jitter_px}px)", color=color)
    draw_point(img, x, y, text=None, color=(255, 255, 255))


def test_visualize_hotzones_rects():
    # 1) 配置
    assert os.path.exists(CFG), "缺少配置文件：configs/config.json5"
    cfg = json5.load(open(CFG, "r", encoding="utf-8"))

    wh = (cfg["screen"]["width"], cfg["screen"]["height"])
    W, H = wh

    # 读取“随机抖动像素”
    assert "random" in cfg and "tap_jitter_px" in cfg["random"], "config 缺少 random.tap_jitter_px"
    jitter_px = int(cfg["random"]["tap_jitter_px"])

    # ===== 任务列表页 =====
    assert os.path.exists(ASSETS["list"]), f"缺少截图：{ASSETS['list']}"
    img = cv2.imread(ASSETS["list"]); assert img is not None
    assert (img.shape[1], img.shape[0]) == wh, "list_screen 分辨率与 config 不一致"
    x, y = pct_to_px(cfg["coords"]["list_start"], wh)
    _draw_centered_jitter_rect(img, x, y, jitter_px, "list_start", (0, 255, 0))
    save_image(img, OUTS["list"])

    # ===== 战前页 =====
    assert os.path.exists(ASSETS["prebattle"]), f"缺少截图：{ASSETS['prebattle']}"
    img = cv2.imread(ASSETS["prebattle"]); assert img is not None
    assert (img.shape[1], img.shape[0]) == wh, "prebattle_screen 分辨率与 config 不一致"
    x, y = pct_to_px(cfg["coords"]["pre_start"], wh)
    _draw_centered_jitter_rect(img, x, y, jitter_px, "pre_start", (0, 255, 0))
    save_image(img, OUTS["prebattle"])

    # ===== 战斗页（自动 6 个 support*）=====
    assert os.path.exists(ASSETS["combat"]), f"缺少截图：{ASSETS['combat']}"
    img = cv2.imread(ASSETS["combat"]); assert img is not None
    assert (img.shape[1], img.shape[0]) == wh, "combat_screen 分辨率与 config 不一致"
    sup_keys = sorted([k for k in cfg["coords"] if k.startswith("support")])
    assert len(sup_keys) >= 1, f"未找到 support* 坐标：{sup_keys}"
    palette = [(0, 255, 255), (0, 230, 255), (0, 205, 255),
               (0, 180, 255), (0, 155, 255), (0, 130, 255)]
    for i, k in enumerate(sup_keys):
        x, y = pct_to_px(cfg["coords"][k], wh)
        _draw_centered_jitter_rect(img, x, y, jitter_px, k, palette[i % len(palette)])
    save_image(img, OUTS["combat"])

    # ===== 结算页 =====
    assert os.path.exists(ASSETS["settlement"]), f"缺少截图：{ASSETS['settlement']}"
    img = cv2.imread(ASSETS["settlement"]); assert img is not None
    assert (img.shape[1], img.shape[0]) == wh, "settlement_screen 分辨率与 config 不一致"
    x, y = pct_to_px(cfg["coords"]["collect"], wh)
    _draw_centered_jitter_rect(img, x, y, jitter_px, "collect", (0, 255, 0))
    save_image(img, OUTS["settlement"])
