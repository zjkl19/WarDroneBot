# scripts/crop_template.py
"""
从整屏截图按 config.json5 的锚点裁出模板（可选自动生成 mask）。
示例（PowerShell）：
  python scripts/crop_template.py --screen tests/assets/combat_screen.jpg ^
    --anchor hp_bar --size 520x120 --out-dir templates --name combat_hp --auto-mask

参数：
  --screen    整屏截图路径（jpg/png）
  --anchor    configs/config.json5 里的 coords 锚点名
  --size      WxH 像素（例如 520x120）
  --name      输出模板的基名（不带扩展名）；会写出 <name>.png
  --out-dir   输出目录（默认 templates）
  --dx/--dy   相对屏幕的偏移（比例，正=右/下，负=左/上），用于微调锚点
  --auto-mask 生成 *_mask.png（用边缘膨胀得到的二值掩码）
"""
import os
import argparse
import json5
import cv2
import numpy as np

CFG_PATH = "configs/config.json5"

def _pct_to_px(p, wh): return int(p[0] * wh[0]), int(p[1] * wh[1])

def _crop_center(bgr, cx, cy, w, h):
    H, W = bgr.shape[:2]
    x1 = max(0, cx - w // 2); y1 = max(0, cy - h // 2)
    x2 = min(W, cx + w // 2); y2 = min(H, cy + h // 2)
    return bgr[y1:y2, x1:x2].copy(), (x1, y1, x2, y2)

def _auto_mask(tile_bgr):
    """基于边缘的简单自动掩码：Canny → 膨胀 → 二值"""
    g = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(g, (3, 3), 0)
    edges = cv2.Canny(g, 50, 120)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(edges, kernel, iterations=1)
    _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
    return mask

def parse_size(s: str):
    w, h = s.lower().split("x")
    return int(w), int(h)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--screen", required=True, help="整屏截图路径")
    ap.add_argument("--anchor", required=True, help="config.json5 里的 coords 锚点名")
    ap.add_argument("--size", required=True, help="WxH 像素，例如 520x120")
    ap.add_argument("--name", required=True, help="输出模板文件名（不带扩展名）")
    ap.add_argument("--out-dir", default="templates")
    ap.add_argument("--dx", type=float, default=0.0, help="相对屏幕宽度的偏移（正=右）")
    ap.add_argument("--dy", type=float, default=0.0, help="相对屏幕高度的偏移（正=下）")
    ap.add_argument("--auto-mask", action="store_true")
    args = ap.parse_args()

    cfg = json5.load(open(CFG_PATH, "r", encoding="utf-8"))
    wh = (cfg["screen"]["width"], cfg["screen"]["height"])
    coords = cfg["coords"]

    assert os.path.exists(args.screen), f"找不到整屏：{args.screen}"
    assert args.anchor in coords, f"coords 里没有锚点：{args.anchor}"

    W, H = wh
    cx, cy = _pct_to_px(coords[args.anchor], wh)
    if args.dx:
        cx += int(args.dx * W)
    if args.dy:
        cy += int(args.dy * H)

    w, h = parse_size(args.size)
    img = cv2.imread(args.screen)
    assert img is not None, f"无法读取：{args.screen}"

    tile, (x1, y1, x2, y2) = _crop_center(img, cx, cy, w, h)

    os.makedirs(args.out_dir, exist_ok=True)
    out_png = os.path.join(args.out_dir, f"{args.name}.png")
    cv2.imwrite(out_png, tile)

    if args.auto_mask:
        mask = _auto_mask(tile)
        out_mask = os.path.join(args.out_dir, f"{args.name}_mask.png")
        cv2.imwrite(out_mask, mask)

    print(f"Wrote: {out_png}  size={w}x{h}  roi=({x1},{y1})-({x2},{y2})  "
          f"offset=({args.dx:.3f},{args.dy:.3f})  mask={'yes' if args.auto_mask else 'no'}")

if __name__ == "__main__":
    main()
