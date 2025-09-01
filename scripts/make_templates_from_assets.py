# scripts/make_templates_from_assets.py
"""
从整屏截图裁剪按钮模板，用于 TemplateStateDetector。
- 以 configs/config.json5 中的 coords 锚点为中心，再叠加“相对偏移”(dx,dy)裁剪；
- 不修改 coords（点击逻辑不受影响）；
- 支持分状态尺寸与分状态偏移。
用法示例：
  python scripts/make_templates_from_assets.py
  python scripts/make_templates_from_assets.py --pre-size 150x60 --pre-dy -0.02
  python scripts/make_templates_from_assets.py --combat-size 100x100 --combat-dx 0.01
"""

import os
import argparse
import json5
import cv2

CFG = "configs/config.json5"
ASSETS = {
    "list":       "tests/assets/list_screen.jpg",
    "prebattle":  "tests/assets/prebattle_screen.jpg",
    "combat":     "tests/assets/combat_screen.jpg",
    "settlement": "tests/assets/settlement_screen.jpg",
}
OUTS = {
    "list":       "templates/btn_list_start.png",
    "prebattle":  "templates/btn_pre_start.png",
    "combat":     "templates/btn_support_icon.png",
    "settlement": "templates/btn_collect.png",
}
ANCHORS = {
    "list":       "list_start",
    "prebattle":  "pre_start",
    "combat":     "support3",
    "settlement": "collect",
}

def pct_to_px(p, wh):
    return int(p[0] * wh[0]), int(p[1] * wh[1])

def crop_center(bgr, cx, cy, w, h):
    H, W = bgr.shape[:2]
    x1 = max(0, cx - w // 2); y1 = max(0, cy - h // 2)
    x2 = min(W, cx + w // 2); y2 = min(H, cy + h // 2)
    return bgr[y1:y2, x1:x2].copy()

def parse_size(s: str):
    w, h = s.lower().split("x")
    return int(w), int(h)

def main():
    ap = argparse.ArgumentParser()
    # 分状态尺寸（像素）
    ap.add_argument("--list-size",   default="180x80")
    ap.add_argument("--pre-size",    default="180x80")
    ap.add_argument("--combat-size", default="120x120")
    ap.add_argument("--settle-size", default="180x80")
    # 全局相对偏移（比例，正右负左 / 正下负上）
    ap.add_argument("--dx", type=float, default=0.0, help="全局横向相对偏移（比例），正=右，负=左")
    ap.add_argument("--dy", type=float, default=0.0, help="全局纵向相对偏移（比例），正=下，负=上")
    # 分状态相对偏移（比例）—— 若提供则覆盖全局
    ap.add_argument("--list-dx",   type=float, default=None)
    ap.add_argument("--list-dy",   type=float, default=None)
    ap.add_argument("--pre-dx",    type=float, default=None)
    ap.add_argument("--pre-dy",    type=float, default=None)
    ap.add_argument("--combat-dx", type=float, default=None)
    ap.add_argument("--combat-dy", type=float, default=None)
    ap.add_argument("--settle-dx", type=float, default=None)
    ap.add_argument("--settle-dy", type=float, default=None)
    args = ap.parse_args()

    # 组装尺寸表
    sizes = {
        "list":       parse_size(args.list_size),
        "prebattle":  parse_size(args.pre_size),
        "combat":     parse_size(args.combat_size),
        "settlement": parse_size(args.settle_size),
    }

    # 偏移：若分状态未指定，则回退全局
    def ov(val, default): return default if val is None else val
    offsets_rel = {
        "list":       (ov(args.list_dx,   args.dx), ov(args.list_dy,   args.dy)),
        "prebattle":  (ov(args.pre_dx,    args.dx), ov(args.pre_dy,    args.dy)),
        "combat":     (ov(args.combat_dx, args.dx), ov(args.combat_dy, args.dy)),
        "settlement": (ov(args.settle_dx, args.dx), ov(args.settle_dy, args.dy)),
    }

    cfg = json5.load(open(CFG, "r", encoding="utf-8"))
    wh = (cfg["screen"]["width"], cfg["screen"]["height"])
    coords = cfg["coords"]

    os.makedirs("templates", exist_ok=True)

    for k in ["list", "prebattle", "combat", "settlement"]:
        src = ASSETS[k]; out = OUTS[k]
        assert os.path.exists(src), f"缺少整屏 {src}"
        img = cv2.imread(src); assert img is not None, f"无法读取 {src}"

        cx, cy = pct_to_px(coords[ANCHORS[k]], wh)
        rel_dx, rel_dy = offsets_rel[k]
        cx += int((rel_dx or 0.0) * wh[0])
        cy += int((rel_dy or 0.0) * wh[1])

        w, h = sizes[k]
        tile = crop_center(img, cx, cy, w, h)
        cv2.imwrite(out, tile)
        print(f"Wrote: {out}  size={w}x{h}  offset=({rel_dx or 0:.3f},{rel_dy or 0:.3f})")

if __name__ == "__main__":
    main()
