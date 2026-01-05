# scripts/record_macro.py
"""
从设备截屏并用鼠标点击录制按键序列，保存为 JSON 以便回放。
操作说明：
  - 左键点击：记录一次 tap（相对坐标 + 与上次点击的时间间隔 dt）
  - r：刷新截屏
  - u：撤销最后一次点击
  - q：退出并保存
示例：
  python scripts/record_macro.py --name stage1 --serial <adb-serial>
"""
import argparse
import json
import os
import time
from datetime import datetime

import cv2
import json5

from war_drone.adb_client import AdbClient


def _load_screen_wh(cfg_path: str = "configs/config.json5"):
    cfg = json5.load(open(cfg_path, "r", encoding="utf-8"))
    screen = cfg["screen"]
    return int(screen["width"]), int(screen["height"])


def _write_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="录制文件名（不带扩展名），保存到 recordings/<name>.json")
    ap.add_argument("--serial", default=None, help="adb 设备序列号（可选）")
    ap.add_argument("--out-dir", default="recordings", help="录制输出目录，默认 recordings")
    args = ap.parse_args()

    adb = AdbClient(serial=args.serial)

    # 初次截屏以确定分辨率
    frame = adb.screencap()
    assert frame is not None, "无法从设备获取截图，请确认 adb devices 正常"
    H, W = frame.shape[:2]
    cfg_W, cfg_H = _load_screen_wh()
    print(f"[INFO] device screenshot size={W}x{H}, config screen={cfg_W}x{cfg_H}")

    events = []  # [{type, pos, dt, note?}]
    start_ts = time.time()
    last_ts = start_ts
    win = "record_macro"
    cv2.namedWindow(win)

    state = {"frame": frame, "events": events, "last_ts": last_ts, "start_ts": start_ts}

    def on_mouse(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        now = time.time()
        dt = now - state["last_ts"]
        state["last_ts"] = now
        rel = [x / float(W), y / float(H)]
        ev = {"type": "tap", "pos": rel, "dt": dt}
        state["events"].append(ev)
        print(f"[REC] tap #{len(state['events'])} pos=({rel[0]:.3f},{rel[1]:.3f}) dt={dt:.2f}s")

    cv2.setMouseCallback(win, on_mouse)

    print("操作：左键记录 / r 刷新 / u 撤销 / q 保存退出")
    while True:
        vis = state["frame"].copy()
        # 标记已录制的点击
        for idx, ev in enumerate(state["events"], start=1):
            x = int(ev["pos"][0] * W)
            y = int(ev["pos"][1] * H)
            cv2.circle(vis, (x, y), 16, (0, 255, 0), 2)
            cv2.putText(vis, str(idx), (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow(win, vis)
        key = cv2.waitKey(50) & 0xFF
        if key == ord("q"):
            break
        if key == ord("r"):
            state["frame"] = adb.screencap()
            if state["frame"] is None:
                print("[WARN] 截屏失败，保持上一帧")
            else:
                H, W = state["frame"].shape[:2]
                print("[INFO] 已刷新截屏")
        if key == ord("u"):
            if state["events"]:
                state["events"].pop()
                # 重算 last_ts
                elapsed = sum(ev["dt"] for ev in state["events"])
                state["last_ts"] = state["start_ts"] + elapsed
                print(f"[INFO] 撤销，剩余 {len(state['events'])} 条")
            else:
                print("[INFO] 无可撤销的事件")

    cv2.destroyAllWindows()

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"{args.name}.json")
    data = {
        "version": 1,
        "created_at": datetime.now().isoformat(),
        "device_px": {"width": W, "height": H},
        "config_px": {"width": cfg_W, "height": cfg_H},
        "events": state["events"],
    }
    _write_json(out_path, data)
    print(f"[DONE] 已保存 {len(state['events'])} 条点击到 {out_path}")


if __name__ == "__main__":
    main()
