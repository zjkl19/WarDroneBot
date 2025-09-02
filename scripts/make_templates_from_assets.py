# scripts/make_templates_from_assets.py
"""
从整屏截图裁剪按钮/特征模板，用于 TemplateStateDetector。
- 以 configs/config.json5 中的 coords 锚点为中心，可叠加“相对偏移”(dx,dy)裁剪；
- 不修改 coords（点击逻辑不受影响）；
- 支持分状态尺寸与分状态偏移；
- 支持 splash。
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
    "splash":     "tests/assets/splash_screen.jpg",   # 放一张代表性的 splash 图
}
OUTS = {
    "list":       "templates/btn_list_start.png",
    "prebattle":  "templates/btn_pre_start.png",
    "combat":     "templates/btn_support_icon.png",
    "settlement": "templates/btn_collect.png",
    "splash":     "templates/logo_war_drone.png",
}
# 默认裁剪尺寸（像素）
DEFAULT_SIZES = {
    "list":       (180, 80),
    "prebattle":  (180, 80),
    "combat":     (120, 120),
    "settlement": (180, 80),
    "splash":     (400, 200),  # 覆盖“WAR DRONE”字样+底部黑条
}
ANCHORS = {
    "list":       "list_start",
    "prebattle":  "pre_start",
    "combat":     "support3",
    "settlement": "collect",
    "splash":     "splash_logo",
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
    ap.add_argument("--list-size",     default="180x80")
    ap.add_argument("--pre-size",      default="180x80")
    ap.add_argument("--combat-size",   default="120x120")
    ap.add_argument("--settle-size",   default="180x80")
    ap.add_argument("--splash-size",   default="400x200")
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
    ap.add_argument("--splash-dx", type=float, default=None)
    ap.add_argument("--splash-dy", type=float, default=None)
    args = ap.parse_args()

    # 组装尺寸表（允许通过参数覆盖默认值）
    sizes = dict(DEFAULT_SIZES)
    sizes.update({
        "list":       parse_size(args.list_size),
        "prebattle":  parse_size(args.pre_size),
        "combat":     parse_size(args.combat_size),
        "settlement": parse_size(args.settle_size),
        "splash":     parse_size(args.splash_size),
    })

    # 偏移：若分状态未指定，则回退全局
    def ov(val, default): return default if val is None else val
    offsets_rel = {
        "list":       (ov(args.list_dx,   args.dx), ov(args.list_dy,   args.dy)),
        "prebattle":  (ov(args.pre_dx,    args.dx), ov(args.pre_dy,    args.dy)),
        "combat":     (ov(args.combat_dx, args.dx), ov(args.combat_dy, args.dy)),
        "settlement": (ov(args.settle_dx, args.dx), ov(args.settle_dy, args.dy)),
        "splash":     (ov(args.splash_dx, args.dx), ov(args.splash_dy, args.dy)),
    }

    cfg = json5.load(open(CFG, "r", encoding="utf-8"))
    wh = (cfg["screen"]["width"], cfg["screen"]["height"])
    coords = cfg["coords"]

    os.makedirs("templates", exist_ok=True)

    for k in ["list", "prebattle", "combat", "settlement", "splash"]:
        src = ASSETS[k]
        out = OUTS[k]
        if not os.path.exists(src):
            print(f"[WARN] 跳过 {k}：缺少整屏 {src}")
            continue

        img = cv2.imread(src)
        if img is None:
            print(f"[WARN] 跳过 {k}：无法读取 {src}")
            continue

        if ANCHORS[k] not in coords:
            print(f"[WARN] 跳过 {k}：coords 里找不到锚点 '{ANCHORS[k]}'")
            continue

        cx, cy = pct_to_px(coords[ANCHORS[k]], wh)
        rel_dx, rel_dy = offsets_rel[k]
        if rel_dx: cx += int(rel_dx * wh[0])
        if rel_dy: cy += int(rel_dy * wh[1])

        w, h = sizes[k]
        tile = crop_center(img, cx, cy, w, h)
        cv2.imwrite(out, tile)
        print(f"Wrote: {out}  size={w}x{h}  offset=({rel_dx or 0:.3f},{rel_dy or 0:.3f})")

if __name__ == "__main__":
    main()
