"""
目的：
- 读取 configs/config.json5 里的相对坐标（0~1）
- 使用分辨率（2670x1200）换算为像素
- 在一张整屏 JPG（tests/assets/screen_sample.jpg）上画出这些点，并保存到 out_vis/mark_clicks.jpg
- 便于你“肉眼核查”按钮位置是否准确

使用：
pytest -q tests/test_pct_to_px_and_clicks_viz.py
生成：out_vis/mark_clicks.jpg
"""
import json5
import cv2
import os
from tests.helpers.visual import draw_point, save_image

CFG = "configs/config.json5"
ASSET = "tests/assets/screen_sample.jpg"
OUT = "out_vis/mark_clicks.jpg"

def pct_to_px(p, wh):
    return int(p[0]*wh[0]), int(p[1]*wh[1])

def test_visualize_click_points():
    assert os.path.exists(CFG), f"配置不存在：{CFG}"
    assert os.path.exists(ASSET), f"测试资产不存在：{ASSET}"

    cfg = json5.load(open(CFG, "r", encoding="utf-8"))
    img = cv2.imread(ASSET)
    assert img is not None, "无法读取测试图片"
    H, W = img.shape[:2]

    wh = (cfg["screen"]["width"], cfg["screen"]["height"])
    assert (W, H) == wh, f"图片分辨率 {W}x{H} 与配置 {wh} 不一致，请换一张对应分辨率的图"

    # 要标注的一组键
    keys = ["list_start","pre_start","collect","support1","support2","support3","support4"]
    for k in keys:
        assert k in cfg["coords"], f"coords 缺少键：{k}"
        x, y = pct_to_px(cfg["coords"][k], wh)
        draw_point(img, x, y, text=k)

    out = save_image(img, OUT)
    print(f"[VIS] 标注已保存：{out}")
