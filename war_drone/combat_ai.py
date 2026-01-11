# war_drone/combat_ai.py
from __future__ import annotations


def _center_xyxy(xyxy):
    x1, y1, x2, y2 = xyxy
    return (x1 + x2) * 0.5, (y1 + y2) * 0.5


def _clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


def _center_in_mask(cx, cy, masks, screen_w, screen_h):
    for mx, my, mw, mh in masks:
        x1 = mx * screen_w
        y1 = my * screen_h
        x2 = x1 + mw * screen_w
        y2 = y1 + mh * screen_h
        if x1 <= cx <= x2 and y1 <= cy <= y2:
            return True
    return False


def filter_detections(detections, screen_w, screen_h, min_box_px, masks):
    filtered = []
    for det in detections:
        x1, y1, x2, y2 = det["xyxy"]
        w = x2 - x1
        h = y2 - y1
        if w < min_box_px or h < min_box_px:
            continue
        cx, cy = _center_xyxy(det["xyxy"])
        if _center_in_mask(cx, cy, masks, screen_w, screen_h):
            continue
        det = dict(det)
        det["center"] = (cx, cy)
        det["w"] = w
        det["h"] = h
        filtered.append(det)
    return filtered


def pick_target(detections, screen_w, screen_h):
    if not detections:
        return None
    cx = screen_w * 0.5
    cy = screen_h * 0.5
    scored = []
    for det in detections:
        dx = det["center"][0] - cx
        dy = det["center"][1] - cy
        dist2 = dx * dx + dy * dy
        scored.append((dist2, -det["conf"], det))
    scored.sort(key=lambda x: (x[0], x[1]))
    return scored[0][2]


def suggest_swipe(target_center, screen_w, screen_h, swipe_region, gain=1.0):
    if target_center is None:
        return None
    sx, sy, sw, sh = swipe_region
    start_x = sx + sw * 0.5
    start_y = sy + sh * 0.5
    dx = (target_center[0] - screen_w * 0.5) / screen_w
    dy = (target_center[1] - screen_h * 0.5) / screen_h
    max_dx = sw * 0.5
    max_dy = sh * 0.5
    end_x = start_x + _clamp(dx * gain, -max_dx, max_dx)
    end_y = start_y + _clamp(dy * gain, -max_dy, max_dy)
    return (start_x, start_y), (end_x, end_y)
