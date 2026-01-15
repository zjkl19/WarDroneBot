# AI combat runner: YOLO 检测 + 简易自瞄/开火（不转视角）
# 默认不影响原 paddle 宏流程，可单独运行。
#
# 用法示例（先激活安装好 ultralytics 的 venv）:
#   python -m scripts.ai_combat_runner ^
#     --serial e5081c2a ^
#     --model runs/detect/train4/weights/best.pt ^
#     --cfg configs/yolo_combat.json5 ^
#     --device cpu ^
#     --interval 0.2
#
# 关键参数来自 configs/yolo_combat.json5 内的 ai_runtime，可按需调整:
#   imgsz / conf / min_box_px / aim_tolerance / lead_k / cooldown / priority / fire_buttons
#
import argparse
import json5
import time
import math

import numpy as np
from ultralytics import YOLO

from war_drone.adb_client import AdbClient


def _pct_to_px(p, wh):
    return int(p[0] * wh[0]), int(p[1] * wh[1])


def clamp(v, lo=0.0, hi=1.0):
    return max(lo, min(hi, v))


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--serial", default=None, help="adb 序列号")
    ap.add_argument("--model", required=True, help="YOLO 权重路径")
    ap.add_argument("--cfg", default="configs/yolo_combat.json5", help="AI 运行配置")
    ap.add_argument("--device", default="cpu", help="YOLO device, e.g. cpu / 0")
    ap.add_argument("--interval", type=float, default=0.2, help="循环间隔（s），含推理耗时")
    ap.add_argument("--max-frames", type=int, default=0, help="最多循环帧数，0=不限制")
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = json5.load(open(args.cfg, "r", encoding="utf-8"))
    rt = cfg.get("ai_runtime", {})
    classes = cfg["classes"]
    W, H = cfg["screen"]["width"], cfg["screen"]["height"]

    imgsz = rt.get("imgsz", 1344)
    conf_th = rt.get("conf", 0.12)
    min_box_px = rt.get("min_box_px", 6)
    aim_tol = rt.get("aim_tolerance", 0.03)
    lead_k = rt.get("lead_k", 0.2)
    lock_enable = rt.get("lock_enable", False)
    lock_reacquire = rt.get("lock_reacquire_frames", 8)
    swipe_gain = rt.get("swipe_gain", 0.8)
    swipe_min_offset = rt.get("swipe_min_offset", 0.01)
    swipe_duration = rt.get("swipe_duration", 0.12)
    swipe_max_step = rt.get("swipe_max_step", 0.10)
    cooldown = rt.get("cooldown", {"hydra": 1.5, "hellfire": 2.0})
    priority = rt.get("priority", classes)
    fire_buttons = rt.get("fire_buttons", {})
    swipe_region = cfg.get("swipe_region", [0.30, 0.25, 0.40, 0.50])  # [x,y,w,h]

    prio_index = {n: i for i, n in enumerate(priority)}
    last_seen = {}  # name -> (cx, cy, t)
    last_fire = {"hydra": 0.0, "hellfire": 0.0}
    locked = None          # {"name": str, "pos": (cx,cy)}
    lock_miss = 0

    model = YOLO(args.model)
    adb = AdbClient(serial=args.serial)

    print(f"[INFO] AI combat runner start: device={args.device}, imgsz={imgsz}, conf={conf_th}")
    frame_idx = 0
    try:
        while True:
            t0 = time.time()
            if args.max_frames and frame_idx >= args.max_frames:
                print("[INFO] reach max frames, exit.")
                break
            frame_idx += 1

            bgr = adb.screencap()
            if bgr is None:
                print("[WARN] screencap failed")
                continue
            # YOLO 接受 BGR/RGB 均可，这里保持 BGR 直接传
            res = model.predict(
                bgr,
                imgsz=imgsz,
                conf=conf_th,
                device=args.device,
                verbose=False,
            )[0]

            targets = []
            for box, cls_idx, score in zip(
                res.boxes.xyxy.cpu().numpy(),
                res.boxes.cls.cpu().numpy(),
                res.boxes.conf.cpu().numpy(),
            ):
                w = box[2] - box[0]
                h = box[3] - box[1]
                if w < min_box_px or h < min_box_px:
                    continue
                cx = (box[0] + box[2]) / (2 * W)
                cy = (box[1] + box[3]) / (2 * H)
                name = classes[int(cls_idx)]
                targets.append({"name": name, "conf": score, "cx": cx, "cy": cy})

            if not targets:
                time.sleep(max(0.0, args.interval - (time.time() - t0)))
                continue

            def sort_key(t):
                p = prio_index.get(t["name"], 999)
                dc = abs(t["cx"] - 0.5) + abs(t["cy"] - 0.5)
                return (p, dc, -t["conf"])

            # 目标选择：若开启锁定则优先保持同一目标，否则按排序选第一个
            tgt = None
            if lock_enable and locked:
                # 找同名且距离上次位置最近的
                same = [t for t in targets if t["name"] == locked["name"]]
                if same:
                    same.sort(key=lambda t: (abs(t["cx"] - locked["pos"][0]) + abs(t["cy"] - locked["pos"][1])))
                    tgt = same[0]
                    lock_miss = 0
                    locked = {"name": tgt["name"], "pos": (tgt["cx"], tgt["cy"])}
                else:
                    lock_miss += 1
                    if lock_miss >= lock_reacquire:
                        locked = None
                        lock_miss = 0
            if tgt is None:
                targets.sort(key=sort_key)
                tgt = targets[0]
                if lock_enable:
                    locked = {"name": tgt["name"], "pos": (tgt["cx"], tgt["cy"])}
                    lock_miss = 0

            # 速度估计 + 提前量
            now = time.time()
            vx = vy = 0.0
            if tgt["name"] in last_seen:
                px, py, pt = last_seen[tgt["name"]]
                dt = max(1e-3, now - pt)
                vx = (tgt["cx"] - px) / dt
                vy = (tgt["cy"] - py) / dt
            last_seen[tgt["name"]] = (tgt["cx"], tgt["cy"], now)

            aim_x = clamp(tgt["cx"] + vx * lead_k)
            aim_y = clamp(tgt["cy"] + vy * lead_k)
            dist = math.hypot(aim_x - 0.5, aim_y - 0.5)
            if dist > aim_tol:
                # 先做一次小幅滑动，把准心拉向预瞄点
                dx = aim_x - 0.5
                dy = aim_y - 0.5
                if abs(dx) > swipe_min_offset or abs(dy) > swipe_min_offset:
                    sx, sy, sw, sh = swipe_region
                    start = (sx + sw / 2, sy + sh / 2)
                    # 限制一次滑动的最大步长，防止拉得过远
                    step_x = dx * swipe_gain
                    step_y = dy * swipe_gain
                    step_len = math.hypot(step_x, step_y)
                    if step_len > swipe_max_step > 0:
                        scale = swipe_max_step / step_len
                        step_x *= scale
                        step_y *= scale
                    move_x = clamp(start[0] + step_x)
                    move_y = clamp(start[1] + step_y)
                    x1, y1 = _pct_to_px(start, (W, H))
                    x2, y2 = _pct_to_px((move_x, move_y), (W, H))
                    dur_ms = int(max(1, swipe_duration * 1000))
                    adb._cmd(["shell", "input", "swipe", str(x1), str(y1), str(x2), str(y2), str(dur_ms)])
                # 本帧不射击，下一帧再判
                time.sleep(max(0.0, args.interval - (time.time() - t0)))
                continue

            # 武器选择（简化版）
            fire_btn = fire_buttons.get("gun")
            if tgt["name"] in ("CV90Tank", "AbramsTank"):
                if now - last_fire["hellfire"] >= cooldown.get("hellfire", 2.0):
                    fire_btn = fire_buttons.get("hellfire", fire_btn)
                    last_fire["hellfire"] = now
            elif tgt["name"] not in ("Infantry",):
                if now - last_fire["hellfire"] >= cooldown.get("hellfire", 2.0):
                    fire_btn = fire_buttons.get("hellfire", fire_btn)
                    last_fire["hellfire"] = now
            # Infantry 默认 gun；Hydra 暂不使用，后续可加群轰逻辑

            if fire_btn:
                x, y = _pct_to_px(fire_btn, (W, H))
                adb.tap(x, y)
                print(f"[FIRE] {tgt['name']} conf={tgt['conf']:.2f} btn={fire_btn}")

            # 控制循环节奏
            time.sleep(max(0.0, args.interval - (time.time() - t0)))

    except KeyboardInterrupt:
        print("\n[INFO] stopped by user")


if __name__ == "__main__":
    main()
