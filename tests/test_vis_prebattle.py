"""
测试目的（Pre-battle 屏）：
- 在 “战前配置页” 的整屏 JPG 上，标注 pre_start 坐标
- 生成 out_vis/vis_prebattle.jpg，核对是否落在右下“开始游戏”按钮上

应截图成什么样（prebattle_screen.jpg）：
- 进入任务后还未开始战斗，右下角有大“开始游戏”按钮
- 下方单位/载具选择条可见
- 无弹窗/礼包/广告
- 分辨率 2670×1200（横屏）

抓图命令：
  python scripts/grab_asset.py --name prebattle_screen
"""
import os, json5, cv2
from tests.helpers.visual import draw_point, save_image

CFG = "configs/config.json5"
SRC = "tests/assets/prebattle_screen.jpg"
OUT = "out_vis/vis_prebattle.jpg"

def pct_to_px(p, wh): return int(p[0]*wh[0]), int(p[1]*wh[1])

def test_visualize_pre_start():
    assert os.path.exists(CFG), "缺少配置 configs/config.json5"
    assert os.path.exists(SRC), f"缺少截图 {SRC}"
    cfg = json5.load(open(CFG, "r", encoding="utf-8"))

    img = cv2.imread(SRC); assert img is not None
    H, W = img.shape[:2]
    wh = (cfg["screen"]["width"], cfg["screen"]["height"])
    assert (W, H) == wh, f"图片分辨率 {W}x{H} 与配置 {wh} 不一致"

    x,y = pct_to_px(cfg["coords"]["pre_start"], wh)
    draw_point(img, x, y, "pre_start", (255,165,0))
    out = save_image(img, OUT)
    print("[VIS] Saved:", out)
