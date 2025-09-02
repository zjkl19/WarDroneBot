# -*- coding: utf-8 -*-
"""
单次模板匹配测试（朴素版）
------------------------------------------------------
把模板在整屏（或 ROI）里 match 一下，输出分数与定位，
并生成一张可视化图片（画出命中的矩形）。

支持方法：
- NCC: 归一化互相关（OpenCV TM_CCORR_NORMED，越大越好）
- SSIM: 结构相似度（需安装 scikit-image）

用法示例：
  # 全图匹配（NCC）
  python scripts/test_template_match.py --screen tests/assets/combat_screen.jpg --template templates/combat_goal.png

  # 只在某 ROI 内匹配（相对坐标）
  python scripts/test_template_match.py --screen tests/assets/combat_screen.jpg --template templates/combat_goal.png `
     --roi 0.88,0.86,0.38,0.18

  # 用 SSIM 测
  python scripts/test_template_match.py --screen tests/assets/combat_screen.jpg --template templates/combat_goal.png --method ssim
"""
import os
import cv2
import numpy as np
import argparse

def parse_roi(s: str):
    parts = [float(x) for x in s.split(",")]
    assert len(parts) == 4
    return parts  # cx,cy,w,h (relative)

def rel_roi_to_rect(roi, W, H):
    cx, cy, rw, rh = roi
    w = int(rw * W)
    h = int(rh * H)
    x1 = int(cx*W - w/2); y1 = int(cy*H - h/2)
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(W, x1 + w); y2 = min(H, y1 + h)
    return (x1,y1,x2,y2)

def match_ncc(big, small):
    res = cv2.matchTemplate(big, small, cv2.TM_CCORR_NORMED)
    min_val,max_val,min_loc,max_loc = cv2.minMaxLoc(res)
    return float(max_val), max_loc

def match_ssim(big, small):
    from skimage.metrics import structural_similarity as ssim
    # 滑窗 SSIM 很慢，这里等比例缩放模板成 big 的多个尺度会更麻烦。
    # 简化：只在 NCC 的 best 附近做局部 SSIM 精修（够用）。
    ncc_score, loc = match_ncc(big, small)
    x,y = loc
    h,w = small.shape[:2]
    crop = big[y:y+h, x:x+w]
    if crop.shape[:2] != (h,w):
        return ncc_score, loc
    g1 = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    s = ssim(g1, g2)
    return float(s), loc

def draw_box(vis, top_left, wh, color=(0,255,255), text=None):
    x,y = top_left; w,h = wh
    cv2.rectangle(vis, (x,y), (x+w, y+h), color, 2)
    if text:
        cv2.putText(vis, text, (x, max(20,y-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--screen", required=True)
    ap.add_argument("--template", required=True)
    ap.add_argument("--roi", default=None, help="相对 ROI：cx,cy,w,h（0~1），仅在该区域匹配")
    ap.add_argument("--method", choices=["ncc","ssim"], default="ncc")
    ap.add_argument("--out", default=None, help="输出可视化路径，默认与屏幕图同目录加 _match.png")
    args = ap.parse_args()

    big = cv2.imread(args.screen); assert big is not None, f"读图失败 {args.screen}"
    small = cv2.imread(args.template); assert small is not None, f"读图失败 {args.template}"
    H,W = big.shape[:2]
    roi_rect = None
    big_roi = big

    if args.roi:
        roi = parse_roi(args.roi)
        x1,y1,x2,y2 = rel_roi_to_rect(roi, W, H)
        big_roi = big[y1:y2, x1:x2]
        roi_rect = (x1,y1,x2,y2)

    if args.method == "ncc":
        score, loc = match_ncc(big_roi, small)
    else:
        score, loc = match_ssim(big_roi, small)

    x,y = loc
    if roi_rect:
        x += roi_rect[0]; y += roi_rect[1]
    h,w = small.shape[:2]

    print(f"[结果] method={args.method}  score={score:.4f}  loc=(x={x}, y={y})  size=({w}x{h})")

    vis = big.copy()
    draw_box(vis, (x,y), (w,h), (0,255,255), f"{args.method}:{score:.3f}")
    if roi_rect:
        rx1,ry1,rx2,ry2 = roi_rect
        cv2.rectangle(vis, (rx1,ry1), (rx2,ry2), (255,0,0), 1)
        cv2.putText(vis, "ROI", (rx1+4, ry1+18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2, cv2.LINE_AA)

    out = args.out or os.path.splitext(args.screen)[0] + "_match.png"
    cv2.imwrite(out, vis)
    print(f"[保存可视化] {out}")

if __name__ == "__main__":
    main()
