# -*- coding: utf-8 -*-
"""
交互式裁剪 ROI 工具
------------------------------------------------------
用鼠标在整屏截图上按下拖拽画矩形，松开后：
- 控制台打印 ROI 的相对坐标 [cx, cy, w, h]（0~1）
- 打印像素坐标与尺寸（方便核对）
- 可按 'S' 保存裁剪图到 --out-dir 并指定文件名
- 可按 'C' 复制（打印）可直接粘进 json5 的行
- 按 'R' 重置矩形；'G' 开关网格；'ESC' 退出

用法示例：
  python scripts/interactive_crop.py --screen tests/assets/combat_screen.jpg --out-dir templates --name combat_goal

小贴士：
- 我们推荐“画框略大一点”，OCR 会更稳。
- 这个脚本只做选择/导出，不改你的 config。
"""
import os
import cv2
import numpy as np
import argparse
from datetime import datetime

def draw_grid(img, step=50, color=(80,80,80)):
    g = img.copy()
    H, W = g.shape[:2]
    for x in range(0, W, step):
        cv2.line(g, (x,0), (x,H), color, 1, cv2.LINE_AA)
    for y in range(0, H, step):
        cv2.line(g, (0,y), (W,y), color, 1, cv2.LINE_AA)
    return g

def norm_roi_from_rect(rect, wh):
    (x1,y1,x2,y2) = rect
    x1,x2 = sorted([max(0,x1), max(0,x2)])
    y1,y2 = sorted([max(0,y1), max(0,y2)])
    x2 = min(x2, wh[0]-1); y2 = min(y2, wh[1]-1)
    w = max(1, x2-x1); h = max(1, y2-y1)
    cx = x1 + w/2; cy = y1 + h/2
    # 归一化
    cx_r = round(cx/wh[0], 6)
    cy_r = round(cy/wh[1], 6)
    w_r  = round(w/wh[0], 6)
    h_r  = round(h/wh[1], 6)
    return (x1,y1,w,h,cx,cy), (cx_r, cy_r, w_r, h_r)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--screen", required=True, help="整屏截图路径（jpg/png）")
    ap.add_argument("--out-dir", default="templates", help="保存裁剪图目录")
    ap.add_argument("--name", default=None, help="裁剪文件名（不含扩展名），默认按时间戳")
    ap.add_argument("--max-width", type=int, default=1920, help="显示时窗口最大宽度（仅影响预览，不影响结果）")
    args = ap.parse_args()

    img = cv2.imread(args.screen)
    assert img is not None, f"无法读取 {args.screen}"
    H, W = img.shape[:2]

    # 展示时可能缩放
    scale = 1.0
    if W > args.max_width:
        scale = args.max_width / float(W)
    disp = cv2.resize(img, (int(W*scale), int(H*scale)), interpolation=cv2.INTER_AREA)

    dragging = False
    rect = None
    show_grid = False
    start_pt = (0,0)

    win = "interactive_crop"
    os.makedirs(args.out_dir, exist_ok=True)
    print("[提示] 鼠标拖拽框选，S=保存，C=打印json5行，R=重置，G=网格，ESC=退出")

    def on_mouse(event, x, y, flags, userdata):
        nonlocal dragging, rect, start_pt
        if event == cv2.EVENT_LBUTTONDOWN:
            dragging = True
            start_pt = (int(x/scale), int(y/scale))
            rect = (start_pt[0], start_pt[1], start_pt[0], start_pt[1])
        elif event == cv2.EVENT_MOUSEMOVE and dragging:
            cur = (int(x/scale), int(y/scale))
            rect = (start_pt[0], start_pt[1], cur[0], cur[1])
        elif event == cv2.EVENT_LBUTTONUP:
            dragging = False
            cur = (int(x/scale), int(y/scale))
            rect = (start_pt[0], start_pt[1], cur[0], cur[1])
            (x1,y1,w,h,cx,cy), (cx_r,cy_r,w_r,h_r) = norm_roi_from_rect(rect, (W,H))
            print(f"\n[ROI 完成] 像素：x={x1}, y={y1}, w={w}, h={h}, cx={cx:.1f}, cy={cy:.1f}")
            print(f"[ROI 归一化] [cx, cy, w, h] = [{cx_r}, {cy_r}, {w_r}, {h_r}]")

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)

    while True:
        base = disp.copy()
        if show_grid:
            base = draw_grid(base, step=50)
        if rect is not None:
            (x1,y1,x2,y2) = rect
            # 画到显示坐标
            p1 = (int(x1*scale), int(y1*scale))
            p2 = (int(x2*scale), int(y2*scale))
            cv2.rectangle(base, p1, p2, (0,255,255), 2)
        cv2.imshow(win, base)
        key = cv2.waitKey(16) & 0xFF
        if key == 27:  # ESC
            break
        elif key in (ord('r'), ord('R')):
            rect = None
        elif key in (ord('g'), ord('G')):
            show_grid = not show_grid
        elif key in (ord('c'), ord('C')):
            if rect is None:
                print("[提示] 请先框选 ROI")
                continue
            (_,_,_,_,_,_), (cx_r,cy_r,w_r,h_r) = norm_roi_from_rect(rect, (W,H))
            print(f'[json5] 例如：  some_roi: [{cx_r}, {cy_r}, {w_r}, {h_r}],')
        elif key in (ord('s'), ord('S')):
            if rect is None:
                print("[提示] 请先框选 ROI")
                continue
            (x1,y1,x2,y2) = rect
            x1,x2 = sorted([x1,x2]); y1,y2 = sorted([y1,y2])
            tile = img[y1:y2, x1:x2].copy()
            if tile.size == 0:
                print("[提示] 矩形无效")
                continue
            name = args.name or f"crop_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            out_path = os.path.join(args.out_dir, f"{name}.png")
            cv2.imwrite(out_path, tile)
            print(f"[保存] {out_path}")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
