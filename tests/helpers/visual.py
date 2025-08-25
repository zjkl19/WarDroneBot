# tests/helpers/visual.py
"""
在整屏截图上画出“点击点/热区矩形”与文字说明，便于核查坐标与热区。
"""
import cv2, os

def draw_point(img_bgr, x, y, text=None, color=(0,255,0)):
    cv2.circle(img_bgr, (int(x), int(y)), 18, color, 3)
    if text:
        cv2.putText(img_bgr, text, (int(x)+10, int(y)-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
    return img_bgr

def draw_rect(img_bgr, x, y, w, h, text=None, color=(255,0,0)):
    """
    在以 (x,y) 为中心的位置绘制宽 w、高 h 的矩形边框。
    """
    x1, y1 = int(x - w/2), int(y - h/2)
    x2, y2 = int(x + w/2), int(y + h/2)
    cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 3)
    if text:
        cv2.putText(img_bgr, text, (x1, max(0, y1-8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    return img_bgr

def save_image(img_bgr, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, img_bgr)
    return out_path

def draw_crosshair(img_bgr, x, y, size=20, color=(255,255,255), thickness=2):
    x, y = int(x), int(y)
    cv2.line(img_bgr, (x - size, y), (x + size, y), color, thickness, cv2.LINE_AA)
    cv2.line(img_bgr, (x, y - size), (x, y + size), color, thickness, cv2.LINE_AA)
    return img_bgr
