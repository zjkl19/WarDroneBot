import cv2
import numpy as np

def bytes_to_bgr(img_bytes: bytes):
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

def draw_crosshair(img_bgr, x, y, size=22, color=(255,255,255), thickness=2):
    x, y = int(x), int(y)
    cv2.line(img_bgr, (x - size, y), (x + size, y), color, thickness, cv2.LINE_AA)
    cv2.line(img_bgr, (x, y - size), (x, y + size), color, thickness, cv2.LINE_AA)
    return img_bgr

def put_label(img_bgr, text, org=(16,40), color=(0,255,0)):
    cv2.putText(img_bgr, text, org, cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
    return img_bgr

def annotate_click(img_bytes, x, y, label=None):
    """
    在截图上画十字并标注文字，返回 PNG 编码后的 bytes。
    """
    bgr = bytes_to_bgr(img_bytes)
    draw_crosshair(bgr, x, y, size=22, color=(255,255,255), thickness=2)
    if label:
        put_label(bgr, label, org=(16,40), color=(0,255,0))
    ok, png = cv2.imencode(".png", bgr)
    return png.tobytes() if ok else img_bytes
