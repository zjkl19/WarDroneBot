"""
tests/helpers/visual.py
用于在整屏截图上标注“点击点（像素）”和文本说明，帮助核查坐标是否准确。
"""
import cv2
import os

def draw_point(img_bgr, x, y, text=None, color=(0, 255, 0)):
    cv2.circle(img_bgr, (int(x), int(y)), 18, color, 3)
    if text:
        cv2.putText(img_bgr, text, (int(x)+10, int(y)-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
    return img_bgr

def save_image(img_bgr, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, img_bgr)
    return out_path
