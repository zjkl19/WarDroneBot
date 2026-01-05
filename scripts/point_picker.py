# -*- coding: utf-8 -*-
"""
交互式坐标拾取工具：在截屏上单击获取相对坐标 (0~1)。
操作：
  - 鼠标左键单击：打印像素坐标与归一化坐标 [x/W, y/H]
  - R：重置（清除标记）
  - G：显示/隐藏网格
  - ESC：退出

示例：
  python scripts/point_picker.py --screen main_menu.png
"""
import argparse
import cv2
import os


def draw_grid(img, step=50, color=(80, 80, 80)):
    g = img.copy()
    H, W = g.shape[:2]
    for x in range(0, W, step):
        cv2.line(g, (x, 0), (x, H), color, 1, cv2.LINE_AA)
    for y in range(0, H, step):
        cv2.line(g, (0, y), (W, y), color, 1, cv2.LINE_AA)
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--screen", required=True, help="截屏路径（png/jpg）")
    ap.add_argument("--max-width", type=int, default=1920, help="显示时的最大宽度（仅预览）")
    args = ap.parse_args()

    assert os.path.exists(args.screen), f"找不到文件：{args.screen}"
    img = cv2.imread(args.screen)
    assert img is not None, f"无法读取 {args.screen}"
    H, W = img.shape[:2]

    scale = 1.0
    if W > args.max_width:
        scale = args.max_width / float(W)
    disp = cv2.resize(img, (int(W * scale), int(H * scale)), interpolation=cv2.INTER_AREA)

    point = None
    show_grid = False
    win = "point_picker"
    print("[提示] 左键拾取坐标；R=重置；G=网格；ESC=退出")

    def on_mouse(event, x, y, flags, userdata):
        nonlocal point
        if event == cv2.EVENT_LBUTTONDOWN:
            px = int(x / scale)
            py = int(y / scale)
            rx = round(px / W, 6)
            ry = round(py / H, 6)
            point = (px, py)
            print(f"[POINT] px=({px},{py})  rel=({rx}, {ry})")

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)

    while True:
        base = disp.copy()
        if show_grid:
            base = draw_grid(base, step=50)
        if point is not None:
            p_disp = (int(point[0] * scale), int(point[1] * scale))
            cv2.drawMarker(base, p_disp, (0, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=18, thickness=2)
        cv2.imshow(win, base)
        key = cv2.waitKey(16) & 0xFF
        if key == 27:  # ESC
            break
        elif key in (ord("r"), ord("R")):
            point = None
        elif key in (ord("g"), ord("G")):
            show_grid = not show_grid

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
