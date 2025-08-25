"""
测试目的（Settlement 屏）：
- 在 “结算页（胜利或失败）” 的整屏 JPG 上，标注 collect 坐标
- 生成 out_vis/vis_settlement.jpg，核对是否落在左下“收集/继续/完成”按钮上
  *注意：我们默认不点“+50% 广告”，所以不要标注右下角那个按钮*

应截图成什么样（settlement_screen.jpg）：
- 失败：“友军被杀死”，左下有“收集”蓝色按钮
- 胜利：一般左下是“继续/完成/收集”之类按钮（我们点这个）
- 不要被“领取+50%”广告按钮遮挡左下按钮
- 分辨率 2670×1200（横屏）

抓图命令：
  python scripts/grab_asset.py --name settlement_screen
"""
import os, json5, cv2
from tests.helpers.visual import draw_point, save_image

CFG = "configs/config.json5"
SRC = "tests/assets/settlement_screen.jpg"
OUT = "out_vis/vis_settlement.jpg"

def pct_to_px(p, wh): return int(p[0]*wh[0]), int(p[1]*wh[1])

def test_visualize_collect_button():
    assert os.path.exists(CFG), "缺少配置 configs/config.json5"
    assert os.path.exists(SRC), f"缺少截图 {SRC}"
    cfg = json5.load(open(CFG, "r", encoding="utf-8"))

    img = cv2.imread(SRC); assert img is not None
    H, W = img.shape[:2]
    wh = (cfg["screen"]["width"], cfg["screen"]["height"])
    assert (W, H) == wh, f"图片分辨率 {W}x{H} 与配置 {wh} 不一致"

    x,y = pct_to_px(cfg["coords"]["collect"], wh)
    draw_point(img, x, y, "collect", (0,255,0))
    out = save_image(img, OUT)
    print("[VIS] Saved:", out)
