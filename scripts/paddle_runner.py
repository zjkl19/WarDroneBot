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
import logging
import threading

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
    ap.add_argument("--prestart-macro", action="store_true", help="点击 ready 后延时播放宏，不等 OCR 判定 combat")
    ap.add_argument("--prestart-delay", type=float, default=1.0, help="ready 点击后延时多少秒启动宏")
    ap.add_argument("--quiet", action="store_true", help="减少日志输出（压低 paddleocr 日志）")
    args = ap.parse_args()

    if args.quiet:
        logging.getLogger("ppocr").setLevel(logging.ERROR)
        logging.getLogger("PIL").setLevel(logging.ERROR)
        logging.getLogger().setLevel(logging.WARNING)

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
    macro_thread = None
    macro_stop_event = threading.Event()
    macro_scheduled_ts = None
    exit_pending = False

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

    def stop_macro(reason=None):
        nonlocal macro_running, macro_thread, macro_stop_event
        if not macro_running:
            return
        macro_stop_event.set()
        if macro_thread:
            macro_thread.join(timeout=0.2)
        macro_thread = None
        macro_running = False
        if reason:
            print(reason)

    def macro_worker(events, loops, scale):
        nonlocal macro_running, macro_thread, macro_stop_event
        idx = 0
        loop_idx = 0
        last_ts = time.time()
        while not macro_stop_event.is_set():
            ev = events[idx]
            wait_s = float(ev.get("dt", 0.0)) * scale - (time.time() - last_ts)
            if wait_s > 0:
                macro_stop_event.wait(wait_s)
                if macro_stop_event.is_set():
                    break
            etype = ev.get("type", "tap")
            if etype == "tap":
                tap_pct(ev.get("pos"), label=f"macro[{loop_idx+1}:{idx+1}]")
            elif etype == "swipe":
                start = ev.get("start"); end = ev.get("end")
                dur_s = float(ev.get("duration", 0.3)) * scale
                if start and end:
                    do_swipe_pct(start, end, dur_s)
                    print(f"[ACTION] macro[{loop_idx+1}:{idx+1}] swipe dur={dur_s:.2f}s")
            last_ts = time.time()
            idx += 1
            if idx >= len(events):
                loop_idx += 1
                if loop_idx >= loops:
                    break
                idx = 0
                last_ts = time.time()
        macro_running = False
        macro_thread = None
        macro_stop_event.clear()
        print("[INFO] combat 宏播放结束")

    def start_macro_immediate():
        nonlocal macro_running, macro_thread, macro_stop_event
        if not macro_events:
            return
        stop_macro()
        macro_stop_event.clear()
        macro_running = True
        macro_thread = threading.Thread(
            target=macro_worker,
            args=(macro_events, args.combat_macro_loops, args.macro_sleep_scale),
            daemon=True,
        )
        macro_thread.start()
        print(f"[INFO] combat 播放宏 {args.combat_macro} loops={args.combat_macro_loops}")

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
                stop_macro("[INFO] 离开 combat，终止宏播放")

            if args.dry_run:
                prev_state = state
                time.sleep(args.interval)
                continue

            # 若已预约宏启动且时间到达，则启动
            if macro_scheduled_ts and time.time() >= macro_scheduled_ts and not macro_running and macro_events:
                start_macro_immediate()
                macro_scheduled_ts = None

            pos = action_map.get(state)
            if pos:
                x, y = _pct_to_px(pos, (W, H))
                tap_px(x, y, label=state)
                # 点击 ready 后，如果启用 prestart-macro，则预约宏
                if state == "ready" and args.prestart_macro and macro_events:
                    macro_scheduled_ts = time.time() + args.prestart_delay
                    print(f"[INFO] 已预约宏，将在 {args.prestart_delay:.2f}s 后启动")
            elif state == "combat":
                if prev_state != "combat":
                    combat_count += 1
                    limit_hit = args.max_combat and combat_count >= args.max_combat
                    # 进入 combat 时启动宏
                    if args.combat_macro and macro_events:
                        start_macro_immediate()
                    if limit_hit:
                        # 达到上限：如果有宏则等宏播完再退出，否则直接退出
                        if args.combat_macro and macro_events:
                            exit_pending = True
                        else:
                            print(f"[INFO] combat 次数 {combat_count} 已达/超限制 {args.max_combat}，结束循环")
                            break
                # 若开启 prestart-macro 且宏尚未启动，可以选择跳过 OCR 触发
                if args.prestart_macro and macro_events and macro_scheduled_ts:
                    # 已预约，确保运行
                    if not macro_running:
                        start_macro_immediate()
                        macro_scheduled_ts = None
                if args.combat_macro and macro_events and macro_running:
                    # 宏由后台线程执行，这里仅保持 OCR 循环
                    pass
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
            # 若达成退出条件且宏已停止，则结束循环
            if exit_pending and not macro_running:
                print(f"[INFO] combat 次数 {combat_count} 已达上限 {args.max_combat}，宏已结束，退出循环")
                break

            prev_state = state
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\n[INFO] 结束")


if __name__ == "__main__":
    main()
