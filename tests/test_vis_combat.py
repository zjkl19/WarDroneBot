"""
测试目的（Combat 屏）：
- 在 “战斗进行中” 的整屏 JPG 上，标注右下 4 个支援按钮坐标 support1..4
- 生成 out_vis/vis_combat.jpg，核对 4 个圆圈是否分别落在 4 个支援按钮上

应截图成什么样（combat_screen.jpg）：
- 战斗画面：右下角能清楚看到 4 个支援按钮（坦克/狙击/范围炸弹×2）
- 上方血条、“目标：x%”等元素出现更好（非必须）
- 无弹窗/广告
- 分辨率 2670×1200（横屏）

抓图命令：
  python scripts/grab_asset.py --name combat_screen
"""
import os, json5, cv2
from tests.helpers.visual import draw_point, save_image

CFG = "configs/config.json5"
SRC = "tests/assets/combat_screen.jpg"
OUT = "out_vis/vis_combat.jpg"

def pct_to_px(p, wh): return int(p[0]*wh[0]), int(p[1]*wh[1])

def test_visualize_support_buttons():
    assert os.path.exists(CFG), "缺少配置 configs/config.json5"
    assert os.path.exists(SRC), f"缺少截图 {SRC}"

    cfg = json5.load(open(CFG, "r", encoding="utf-8"))
    img = cv2.imread(SRC); assert img is not None
    H, W = img.shape[:2]
    wh = (cfg["screen"]["width"], cfg["screen"]["height"])
    assert (W, H) == wh, f"图片分辨率 {W}x{H} 与配置 {wh} 不一致"

    # 自动收集并排序所有 support*
    sup_keys = sorted([k for k in cfg["coords"].keys() if k.startswith("support")])
    assert len(sup_keys) >= 6, "应有 6 个 support 坐标，请检查 config"

    # 颜色轮换：便于区分
    palette = [(0,255,255),(0,220,255),(0,190,255),(0,160,255),(0,130,255),(0,100,255)]
    for i, k in enumerate(sup_keys):
        x,y = pct_to_px(cfg["coords"][k], wh)
        draw_point(img, x, y, k, palette[i % len(palette)])

    out = save_image(img, OUT)
    print("[VIS] Saved:", out)
