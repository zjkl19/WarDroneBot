# scripts/paddle_runner.py
"""
基于 PaddleOCR 的状态循环 runner：
  - 截屏 -> OCR 判定状态（按 configs/ocr_states_fsm.json5）
  - 按状态执行点击（主菜单/准备/结算/弹窗），combat 可播放录制宏或支持连点

用法示例（在已安装 paddleocr 的 venv 内）：
  python -m scripts.paddle_runner --serial <adb-serial> --cfg configs/ocr_states_fsm.json5 --det-dir ... --rec-dir ... --cls-dir ...
参数：
  --interval   状态轮询间隔（秒，默认 1.5）
  --dry-run    仅打印状态，不发送点击
  --combat-macro  combat 状态播放的录制文件（JSON）
"""
import argparse
import json5
import time
import subprocess
import json

from war_drone.adb_client import AdbClient
from war_drone.paddle_state_detector import PaddleStateDetector


def _pct_to_px(p, wh):
    return int(p[0] * wh[0]), int(p[1] * wh[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--serial", default=None, help="adb 序列号")
    ap.add_argument("--cfg", default="configs/ocr_states_fsm.json5")
    ap.add_argument("--det-dir", default=None)
    ap.add_argument("--rec-dir", default=None)
    ap.add_argument("--cls-dir", default=None)
    ap.add_argument("--interval", type=float, default=1.5, help="轮询间隔秒")
    ap.add_argument("--dry-run", action="store_true", help="只打印状态，不点击")
    ap.add_argument("--combat-auto", action="store_true", help="combat 状态自动执行支持点击")
    ap.add_argument("--combat-sleep", type=float, default=0.8, help="combat 连续点击间隔秒")
    ap.add_argument("--combat-macro", default=None, help="combat 状态时播放的录制文件（JSON）")
    ap.add_argument("--combat-macro-loops", type=int, default=1, help="combat 宏循环次数")
    ap.add_argument("--macro-sleep-scale", type=float, default=1.0, help="宏事件间隔缩放系数")
    ap.add_argument("--max-combat", type=int, default=0, help="combat 状态执行的最大次数（0=不限制，按进入combat计数）")
    args = ap.parse_args()

    cfg = json5.load(open(args.cfg, "r", encoding="utf-8"))
    coords = cfg.get("coords", {})
    W, H = cfg["screen"]["width"], cfg["screen"]["height"]

    det = PaddleStateDetector(args.cfg, det_dir=args.det_dir, rec_dir=args.rec_dir, cls_dir=args.cls_dir)
    adb = AdbClient(serial=args.serial)

    # 映射：状态 -> 相对坐标（如需调整可改为 config.coords 内的键值）
    action_map = {
        "main_menu": (0.868165, 0.866667),
        "ready": (0.868165, 0.866667),
        "settlement": (0.150936, 0.85),
        "weapon": (0.098127, 0.543333),  # 武器界面点菜单返回
        # 弹窗关闭：默认点屏幕右上角，如需准确请改成具体 ROI 按钮
        "free_gift": (0.315356, 0.781667),
        "mission_hard": (0.325468, 0.618333),
        "piggy_full": (0.819101, 0.188333),
        "vip_ad": (0.820225, 0.111667),
        "ad_other": (0.95, 0.08),
    }

    # 预加载 combat 宏
    macro_events = []
    if args.combat_macro:
        try:
            data = json.load(open(args.combat_macro, "r", encoding="utf-8"))
            macro_events = data.get("events", [])
            print(f"[INFO] loaded combat macro {args.combat_macro}, events={len(macro_events)}")
        except Exception as e:
            print(f"[WARN] 无法读取 combat 宏 {args.combat_macro}: {e}")
            macro_events = []
    macro_running = False
    macro_idx = 0
    macro_loop = 0
    macro_last_ts = time.time()

    def tap_px(x, y, label=None):
        adb.tap(x, y)
        if label:
            print(f"[ACTION] {label} -> tap ({x},{y})")

    def tap_pct(pos, label=None):
        if not pos:
            return
        x, y = _pct_to_px(pos, (W, H))
        tap_px(x, y, label=label)

    def do_swipe_pct(start, end, duration_s=0.3):
        base = [adb.adb]
        if adb.serial:
            base += ["-s", adb.serial]
        x1, y1 = _pct_to_px(start, (W, H))
        x2, y2 = _pct_to_px(end, (W, H))
        dur_ms = int(max(1, duration_s * 1000))
        cmd = base + ["shell", "input", "swipe", str(x1), str(y1), str(x2), str(y2), str(dur_ms)]
        subprocess.check_call(cmd)

    def run_macro(events, loops=1, scale=1.0):
        nonlocal macro_idx, macro_loop, macro_last_ts, macro_running
        if not events or not macro_running:
            return
        now = time.time()
        while macro_idx < len(events):
            ev = events[macro_idx]
            dt = float(ev.get("dt", 0.0)) * scale
            if now - macro_last_ts < dt:
                break
            etype = ev.get("type", "tap")
            if etype == "tap":
                pos = ev["pos"]
                tap_pct(pos, label=f"macro[{macro_loop+1}:{macro_idx+1}]")
            elif etype == "swipe":
                start = ev.get("start"); end = ev.get("end")
                dur_s = float(ev.get("duration", 0.3)) * scale
                if start and end:
                    do_swipe_pct(start, end, dur_s)
                    print(f"[ACTION] macro[{macro_loop+1}:{macro_idx+1}] swipe dur={dur_s:.2f}s")
            macro_last_ts = time.time()
            macro_idx += 1
        if macro_idx >= len(events):
            macro_loop += 1
            if macro_loop >= loops:
                macro_running = False
                macro_idx = 0
                macro_loop = 0
                print("[INFO] combat 宏播放结束")
            else:
                macro_idx = 0
                macro_last_ts = time.time()

    print("[INFO] paddle runner 启动，按 Ctrl+C 退出")
    prev_state = None
    combat_count = 0
    try:
        while True:
            img = adb.screencap()
            state, dbg = det.predict(img)
            print(f"[STATE] {state} scores={ {k: round(v,2) for k,v in dbg['scores'].items()} }")

            # 离开 combat 时停止宏
            if state != "combat" and macro_running:
                print("[INFO] 离开 combat，终止宏播放")
                macro_running = False
                macro_idx = 0
                macro_loop = 0

            if args.dry_run:
                prev_state = state
                time.sleep(args.interval)
                continue

            pos = action_map.get(state)
            if pos:
                x, y = _pct_to_px(pos, (W, H))
                tap_px(x, y, label=state)
            elif state == "combat":
                if prev_state != "combat":
                    combat_count += 1
                    if args.max_combat and combat_count > args.max_combat:
                        print(f"[INFO] combat 次数 {combat_count} 超过限制 {args.max_combat}，结束循环")
                        break
                    # 进入 combat 时启动宏
                    if args.combat_macro and macro_events:
                        macro_running = True
                        macro_idx = 0
                        macro_loop = 0
                        macro_last_ts = time.time()
                        print(f"[INFO] combat 播放宏 {args.combat_macro} loops={args.combat_macro_loops}")
                if args.combat_macro and macro_events and macro_running:
                    run_macro(macro_events, loops=args.combat_macro_loops, scale=args.macro_sleep_scale)
                elif args.combat_auto:
                    print("[INFO] combat 自动执行支持点击")
                    for i in range(1, 7):
                        key = f"support{i}"
                        if key in coords:
                            sx, sy = _pct_to_px(coords[key], (W, H))
                            tap_px(sx, sy, label=f"combat->{key}")
                            time.sleep(args.combat_sleep)
                else:
                    print("[INFO] combat 状态，暂不自动操作")
            prev_state = state
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\n[INFO] 结束")


if __name__ == "__main__":
    main()
