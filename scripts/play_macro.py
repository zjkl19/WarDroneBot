# scripts/play_macro.py
"""
回放录制好的点击序列（盲点，无状态判定）。
示例：
  python scripts/play_macro.py --file recordings/stage1.json --loops 3 --serial <adb-serial>
"""
import argparse
import json
import os
import time

import json5
import subprocess

from war_drone.adb_client import AdbClient


def _load_screen_wh(cfg_path: str = "configs/config.json5"):
    cfg = json5.load(open(cfg_path, "r", encoding="utf-8"))
    screen = cfg["screen"]
    return int(screen["width"]), int(screen["height"])


def _pct_to_px(pos, wh):
    return int(pos[0] * wh[0]), int(pos[1] * wh[1])


def _do_swipe(adb: AdbClient, x1, y1, x2, y2, duration_ms: int):
    base = [adb.adb]
    if adb.serial:
        base += ["-s", adb.serial]
    args = base + ["shell", "input", "swipe", str(x1), str(y1), str(x2), str(y2), str(max(1, duration_ms))]
    subprocess.check_call(args)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True, help="录制的 JSON 文件路径（如 recordings/stage1.json）")
    ap.add_argument("--serial", default=None, help="adb 设备序列号（可选）")
    ap.add_argument("--loops", type=int, default=1, help="回放循环次数，默认 1")
    ap.add_argument("--sleep-scale", type=float, default=1.0, help="时间缩放系数，<1 加速，>1 减速")
    ap.add_argument("--dry-run", action="store_true", help="仅打印不实际点击")
    args = ap.parse_args()

    assert os.path.exists(args.file), f"找不到录制文件：{args.file}"
    with open(args.file, "r", encoding="utf-8") as f:
        data = json.load(f)
    events = data.get("events", [])
    assert events, "录制文件 events 为空"

    W, H = _load_screen_wh()
    rec_dev = data.get("device_px")
    if rec_dev:
        rw, rh = int(rec_dev.get("width", W)), int(rec_dev.get("height", H))
        if (rw, rh) != (W, H):
            print(f"[WARN] 录制设备分辨率={rw}x{rh} 与当前配置 {W}x{H} 不同，将按相对坐标回放")
    adb = AdbClient(serial=args.serial)

    print(f"[INFO] loaded {len(events)} events from {args.file}")
    print(f"[INFO] config screen size={W}x{H} ; loops={args.loops} ; sleep_scale={args.sleep_scale}")
    if args.dry_run:
        print("[INFO] dry-run 模式，不会发送点击")

    for loop in range(1, args.loops + 1):
        print(f"[LOOP {loop}/{args.loops}] start")
        for idx, ev in enumerate(events, start=1):
            dt = float(ev.get("dt", 0.0)) * args.sleep_scale
            if dt > 0:
                time.sleep(dt)
            etype = ev.get("type", "tap")
            if etype == "tap":
                pos = ev["pos"]
                x, y = _pct_to_px(pos, (W, H))
                if not args.dry_run:
                    adb.tap(x, y)
                print(f"  [TAP {idx}] ({x},{y}) dt={dt:.2f}s")
            elif etype == "swipe":
                start = ev["start"]; end = ev["end"]
                dur_s = float(ev.get("duration", 0.3)) * args.sleep_scale
                x1, y1 = _pct_to_px(start, (W, H))
                x2, y2 = _pct_to_px(end, (W, H))
                dur_ms = int(max(1, dur_s * 1000))
                if not args.dry_run:
                    _do_swipe(adb, x1, y1, x2, y2, dur_ms)
                print(f"  [SWIPE {idx}] ({x1},{y1})->({x2},{y2}) dt={dt:.2f}s duration={dur_ms}ms")
            else:
                print(f"  [SKIP {idx}] unsupported type={etype}")
        print(f"[LOOP {loop}/{args.loops}] done")

    print("[DONE] 回放完成")


if __name__ == "__main__":
    main()
