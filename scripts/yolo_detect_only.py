# scripts/yolo_detect_only.py
import argparse
import os
import time

import cv2
import json5

from war_drone.adb_client import AdbClient
from war_drone.combat_ai import filter_detections, pick_target, suggest_swipe


def _norm_to_px(p, wh):
    return int(p[0] * wh[0]), int(p[1] * wh[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--serial", default=None)
    ap.add_argument("--cfg", default="configs/yolo_combat.json5")
    ap.add_argument("--model", required=True, help="path to YOLO weights")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default=None, help="e.g. cpu or 0")
    ap.add_argument("--interval", type=float, default=0.3)
    ap.add_argument("--save-dir", default=None, help="save annotated frames")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--min-box-px", type=int, default=None, help="override min_box_px in cfg")
    args = ap.parse_args()

    try:
        from ultralytics import YOLO
    except Exception as e:
        raise SystemExit(f"ultralytics not available: {e}")

    cfg = json5.load(open(args.cfg, "r", encoding="utf-8"))
    screen_w = cfg["screen"]["width"]
    screen_h = cfg["screen"]["height"]
    masks = cfg.get("masks", [])
    min_box_px = int(cfg.get("min_box_px", 12))
    if args.min_box_px is not None:
        min_box_px = args.min_box_px
    swipe_region = cfg.get("swipe_region", [0.3, 0.25, 0.4, 0.5])
    swipe_gain = float(cfg.get("swipe_gain", 1.0))

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

    adb = AdbClient(serial=args.serial)
    model = YOLO(args.model)

    frame_idx = 0
    while True:
        img = adb.screencap()
        res = model.predict(
            img,
            conf=args.conf,
            imgsz=args.imgsz,
            device=args.device,
            verbose=False,
        )[0]
        dets = []
        if res.boxes is not None and len(res.boxes) > 0:
            xyxy = res.boxes.xyxy.cpu().numpy().tolist()
            confs = res.boxes.conf.cpu().numpy().tolist()
            clss = res.boxes.cls.cpu().numpy().tolist()
            for i in range(len(xyxy)):
                dets.append({
                    "xyxy": xyxy[i],
                    "conf": float(confs[i]),
                    "cls": int(clss[i]),
                })

        dets = filter_detections(dets, screen_w, screen_h, min_box_px, masks)
        target = pick_target(dets, screen_w, screen_h)
        swipe = None
        if target:
            swipe = suggest_swipe(target["center"], screen_w, screen_h, swipe_region, gain=swipe_gain)

        print(f"[DETECT] total={len(dets)} min_box_px={min_box_px}")
        if target:
            cx, cy = target["center"]
            print(f"         pick cls={target['cls']} conf={target['conf']:.2f} center=({int(cx)},{int(cy)})")
        else:
            print("         no target")

        if swipe:
            start, end = swipe
            sx, sy = _norm_to_px(start, (screen_w, screen_h))
            ex, ey = _norm_to_px(end, (screen_w, screen_h))
            print(f"[SWIPE] start={start} end={end} px=({sx},{sy})->({ex},{ey})")

        if args.save_dir:
            vis = res.plot()
            if target:
                cx, cy = int(target["center"][0]), int(target["center"][1])
                cv2.circle(vis, (cx, cy), 6, (0, 255, 255), 2)
            if swipe:
                cv2.line(vis, (sx, sy), (ex, ey), (255, 0, 0), 2)
            out_path = os.path.join(args.save_dir, f"frame_{frame_idx:06d}.jpg")
            cv2.imwrite(out_path, vis)

        frame_idx += 1
        if args.max_frames and frame_idx >= args.max_frames:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
