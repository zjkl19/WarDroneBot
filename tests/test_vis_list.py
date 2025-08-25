"""
测试目的（List 屏）：
- 在 “任务列表页” 的整屏 JPG 上，标注 config.json5 中的 list_start 坐标
- 生成 out_vis/vis_list.jpg，肉眼核对圆圈是否落在绿色“开始游戏”按钮上

应截图成什么样（list_screen.jpg）：
- 打开游戏停在“任务”列表界面
- 画面下方靠右能看到绿色“开始游戏”按钮
- 顶部资源栏（体力/现金/金币）可见
- 无弹窗/礼包/广告遮挡
- 分辨率必须是 2670×1200（横屏）

如何抓图（建议命令）：
  python scripts/grab_asset.py --name list_screen
"""
import os, json5, cv2
from tests.helpers.visual import draw_point, save_image

CFG = "configs/config.json5"
SRC = "tests/assets/list_screen.jpg"
OUT = "out_vis/vis_list.jpg"

def pct_to_px(p, wh): return int(p[0]*wh[0]), int(p[1]*wh[1])

def test_visualize_list_start():
    assert os.path.exists(CFG), "缺少配置 configs/config.json5"
    assert os.path.exists(SRC), f"缺少截图 {SRC}"
    cfg = json5.load(open(CFG, "r", encoding="utf-8"))

    img = cv2.imread(SRC); assert img is not None
    H, W = img.shape[:2]
    wh = (cfg["screen"]["width"], cfg["screen"]["height"])
    assert (W, H) == wh, f"图片分辨率 {W}x{H} 与配置 {wh} 不一致"

    x,y = pct_to_px(cfg["coords"]["list_start"], wh)
    draw_point(img, x, y, "list_start", (0,255,0))
    out = save_image(img, OUT)
    print("[VIS] Saved:", out)
