# scripts/record_macro_device.py
"""
直接监听手机触摸事件（adb getevent），记录点击序列为 JSON，可被 play_macro.py 回放。
特性：
- 自动探测触摸设备（首选包含 "touch" 且具备 ABS_MT_POSITION_X/Y 的 /dev/input/eventX）
- 解析 DOWN -> X/Y -> SYN_REPORT 作为一次 tap，记录相对坐标 (0~1) 与时间间隔 dt
- Ctrl+C 结束录制并保存到 recordings/<name>.json

示例：
  python scripts/record_macro_device.py --name stage1 --serial <adb-serial>
"""
import argparse
import json
import os
import re
import subprocess
import time
from datetime import datetime
import sys

import json5

from war_drone.adb_client import AdbClient

try:
    import msvcrt  # Windows 下检测按键退出
except ImportError:
    msvcrt = None


def _is_stop_key(ch: str, stop_key: str) -> bool:
    """判断按键是否为退出键；支持单字符或 'ctrl+q'（0x11）。"""
    if not ch:
        return False
    if stop_key.lower() == "ctrl+q":
        return ch == "\x11"
    return ch.lower() == stop_key.lower()


def _load_screen_wh(cfg_path: str = "configs/config.json5"):
    cfg = json5.load(open(cfg_path, "r", encoding="utf-8"))
    sc = cfg["screen"]
    return int(sc["width"]), int(sc["height"])


def _run_cmd(cmd):
    return subprocess.check_output(cmd, text=True, errors="ignore")


def _parse_devices(getevent_lp_output: str):
    """
    粗略解析 getevent -lp 输出，按设备块收集 capability。
    返回 [{path, name, has_mt, max_x, max_y}, ...]
    """
    devices = []
    current = None
    for line in getevent_lp_output.splitlines():
        line = line.rstrip()
        if line.startswith("add device"):
            if current:
                devices.append(current)
            parts = line.split()
            dev = parts[-1] if parts else ""
            current = {"path": dev, "name": "", "max_x": None, "max_y": None}
        elif line.strip().startswith("name:") and current is not None:
            m = re.search(r'"(.+)"', line)
            if m:
                current["name"] = m.group(1)
        elif "ABS_MT_POSITION_X" in line and current is not None:
            m = re.search(r"max\s+(\d+)", line)
            if m:
                current["max_x"] = int(m.group(1))
        elif "ABS_MT_POSITION_Y" in line and current is not None:
            m = re.search(r"max\s+(\d+)", line)
            if m:
                current["max_y"] = int(m.group(1))
    if current:
        devices.append(current)
    # 标记是否有多点触控能力
    for d in devices:
        d["has_mt"] = (d.get("max_x") is not None) and (d.get("max_y") is not None)
    return devices


def _auto_pick_device(adb: AdbClient):
    base = [adb.adb]
    if adb.serial:
        base += ["-s", adb.serial]
    txt = _run_cmd(base + ["shell", "getevent", "-lp"])
    devices = _parse_devices(txt)
    # 优先匹配名称包含 touch 的多点设备
    for d in devices:
        if d["has_mt"] and ("touch" in d.get("name", "").lower()):
            return d
    # 其次任意具备 MT 的设备
    for d in devices:
        if d["has_mt"]:
            return d
    return None


def _probe_device_caps(adb: AdbClient, dev_path: str):
    """
    单独探测指定设备的 max_x/max_y（getevent -lp /dev/input/eventX）。
    """
    base = [adb.adb]
    if adb.serial:
        base += ["-s", adb.serial]
    try:
        txt = _run_cmd(base + ["shell", "getevent", "-lp", dev_path])
    except subprocess.CalledProcessError:
        return None, None
    devs = _parse_devices(txt)
    for d in devs:
        if d.get("path") == dev_path:
            return d.get("max_x"), d.get("max_y")
    return None, None


def _normalize(x, max_v):
    if max_v and max_v > 0:
        return max(0.0, min(1.0, x / float(max_v)))
    return 0.0


def _apply_rotation(nx, ny, mode: str):
    """
    旋转相对坐标：
      - none: 原样
      - cw:   顺时针90°，(x,y)->(y, 1-x)
      - ccw:  逆时针90°，(x,y)->(1-y, x)
    """
    if mode == "cw":
        return ny, 1.0 - nx
    if mode == "ccw":
        return 1.0 - ny, nx
    return nx, ny


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="录制文件名（不带扩展名），保存到 recordings/<name>.json")
    ap.add_argument("--serial", default=None, help="adb 设备序列号（可选）")
    ap.add_argument("--device", default=None, help="显式指定触摸设备路径，如 /dev/input/event7（可选）")
    ap.add_argument("--out-dir", default="recordings", help="输出目录，默认 recordings")
    ap.add_argument("--debug", action="store_true", help="打印解析到的事件，便于排查录制失败")
    ap.add_argument("--rotate", choices=["auto", "none", "cw", "ccw"], default="auto",
                    help="坐标旋转：auto=根据屏幕/触摸长短边判断；cw=顺时针90°；ccw=逆时针90°")
    ap.add_argument("--stop-key", default="ctrl+q", help="按该键停止录制（Windows 终端），默认 ctrl+q；Ctrl+C 亦可")
    args = ap.parse_args()

    adb = AdbClient(serial=args.serial)
    cfg_W, cfg_H = _load_screen_wh()

    # 选择触摸设备
    dev_info = None
    if args.device:
        mx, my = _probe_device_caps(adb, args.device)
        dev_info = {"path": args.device, "name": args.device, "max_x": mx, "max_y": my, "has_mt": True}
    else:
        dev_info = _auto_pick_device(adb)
    assert dev_info and dev_info.get("path"), "未找到触摸设备，请用 --device 指定 /dev/input/eventX"
    print(f"[INFO] 使用设备 {dev_info['path']} ({dev_info.get('name','')}) max=({dev_info.get('max_x')},{dev_info.get('max_y')})")

    # 读取触摸事件流
    base = [adb.adb]
    if adb.serial:
        base += ["-s", adb.serial]
    cmd = base + ["shell", "getevent", "-lt", dev_info["path"]]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True, errors="ignore")

    events = []
    start_ts = time.time()
    last_event_end_ts = start_ts  # 用于计算各手势的 dt
    in_touch = False
    cur_x = None
    cur_y = None
    max_x = dev_info.get("max_x")
    max_y = dev_info.get("max_y")
    observed_max_x = 0
    observed_max_y = 0
    # 当前手势缓存：{"start_ts":.., "pts":[(t,(x,y)), ...]}
    gesture = None

    def _finalize_gesture(final_ts=None):
        """将当前手势缓存写入 events，并重置缓存。"""
        nonlocal gesture, last_event_end_ts, events
        if not gesture or not gesture.get("pts"):
            gesture = None
            return
        g_start = gesture["start_ts"]
        g_end = gesture["pts"][-1][0] if final_ts is None else final_ts
        dt = g_start - last_event_end_ts
        duration = g_end - g_start
        xs = [p[1][0] for p in gesture["pts"]]
        ys = [p[1][1] for p in gesture["pts"]]
        dx = max(xs) - min(xs)
        dy = max(ys) - min(ys)
        is_swipe = len(gesture["pts"]) >= 2 and max(dx, dy) >= 0.01
        if is_swipe:
            ev = {
                "type": "swipe",
                "start": gesture["pts"][0][1],
                "end": gesture["pts"][-1][1],
                "duration": duration,
                "dt": dt,
            }
            events.append(ev)
            print(f"[REC] swipe #{len(events)} dt={dt:.2f}s duration={duration:.2f}s start={ev['start']} end={ev['end']}")
        else:
            ev = {"type": "tap", "pos": gesture["pts"][0][1], "dt": dt}
            events.append(ev)
            print(f"[REC] tap #{len(events)} pos=({ev['pos'][0]:.3f},{ev['pos'][1]:.3f}) dt={dt:.2f}s")
        last_event_end_ts = g_end
        gesture = None

    # 旋转模式判定
    rotate_mode = args.rotate
    if rotate_mode == "auto" and max_x and max_y:
        # 若触摸坐标长边在Y，屏幕长边在X（横屏），则假设顺时针90°
        if max_x < max_y and cfg_W > cfg_H:
            rotate_mode = "cw"
        elif max_x > max_y and cfg_W < cfg_H:
            rotate_mode = "cw"  # 竖屏场景同方向
        else:
            rotate_mode = "none"
    if args.debug:
        print(f"[INFO] rotate_mode={rotate_mode}")

    print(f"开始录制：在手机上点击；Ctrl+C 或 stop-key({args.stop_key}) 结束并保存")
    stop_requested = False
    try:
        for line in proc.stdout:
            # 允许按 stop-key 退出（Windows msvcrt）
            if msvcrt and msvcrt.kbhit():
                ch = msvcrt.getch()
                if isinstance(ch, bytes):
                    ch = ch.decode(errors="ignore")
                if _is_stop_key(ch, args.stop_key):
                    print(f"[INFO] 检测到停止键 '{args.stop_key}'，结束录制")
                    stop_requested = True
                    break

            line = line.strip()
            # 典型行（有/无时间戳，名称或数字）：
            #   /dev/input/event7: EV_ABS       ABS_MT_POSITION_X    00012ade
            #   [ 1234.567] /dev/input/event7: 0003 0035 00012ade
            if not line:
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            type_raw, code_raw, val_hex = parts[-3], parts[-2], parts[-1]
            try:
                val = int(val_hex, 16)
            except ValueError:
                continue

            # 将数字 code/type 映射为名称，方便统一处理
            EV_TYPE_MAP = {"0003": "EV_ABS", "0001": "EV_KEY", "0000": "EV_SYN"}
            ABS_CODE_MAP = {
                "0035": "ABS_MT_POSITION_X",
                "0036": "ABS_MT_POSITION_Y",
                "0039": "ABS_MT_TRACKING_ID",
                "0000": "ABS_X",
                "0001": "ABS_Y",
            }
            KEY_CODE_MAP = {"014a": "BTN_TOUCH", "0145": "BTN_TOOL_FINGER"}

            ev_type = type_raw if type_raw.startswith("EV_") else EV_TYPE_MAP.get(type_raw.lower(), type_raw)
            code = code_raw
            if ev_type == "EV_ABS":
                code = code_raw if code_raw.startswith("ABS_") else ABS_CODE_MAP.get(code_raw.lower(), code_raw)
            elif ev_type == "EV_KEY":
                code = code_raw if code_raw.startswith("BTN_") else KEY_CODE_MAP.get(code_raw.lower(), code_raw)
            elif ev_type == "EV_SYN" and not code_raw.startswith("SYN_"):
                code = "SYN_REPORT" if code_raw in ("0000", "0") else code_raw

            if args.debug:
                print(f"[DBG] raw={line} -> {ev_type} {code} {val}")

            if ev_type == "EV_KEY" and code in ("BTN_TOUCH", "BTN_TOOL_FINGER"):
                in_touch = (val != 0)
                if not in_touch:
                    _finalize_gesture()
                    cur_x = cur_y = None
                continue

            if ev_type == "EV_ABS":
                if code in ("ABS_MT_POSITION_X", "ABS_X"):
                    cur_x = val
                    observed_max_x = max(observed_max_x, val)
                    if max_x is None:
                        max_x = max(max_x or 0, val)
                    in_touch = True  # 有的设备不发 BTN_TOUCH，看到坐标变化即可认为按下
                elif code in ("ABS_MT_POSITION_Y", "ABS_Y"):
                    cur_y = val
                    observed_max_y = max(observed_max_y, val)
                    if max_y is None:
                        max_y = max(max_y or 0, val)
                    in_touch = True
                elif code == "ABS_MT_TRACKING_ID":
                    if val == 0xFFFFFFFF:
                        # 手指离开
                        in_touch = False
                        _finalize_gesture()
                    else:
                        # 新的触摸 id，重置手势
                        _finalize_gesture()
                continue

            if ev_type == "EV_SYN" and code == "SYN_REPORT":
                if (in_touch or (cur_x is not None and cur_y is not None)) and (cur_x is not None) and (cur_y is not None):
                    now = time.time()
                    nx = _normalize(cur_x, max_x)
                    ny = _normalize(cur_y, max_y)
                    nx, ny = _apply_rotation(nx, ny, rotate_mode)
                    if gesture is None:
                        gesture = {"start_ts": now, "pts": []}
                    gesture["pts"].append((now, (nx, ny)))
                continue
    except KeyboardInterrupt:
        print("\n[INFO] 捕获 Ctrl+C，结束录制并保存...")
    finally:
        # 循环外也尝试收尾，忽略 Ctrl+C
        try:
            _finalize_gesture()
        except KeyboardInterrupt:
            pass
        try:
            if proc and proc.poll() is None:
                proc.terminate()
        except KeyboardInterrupt:
            pass

    # 如果未能探测到设备 max，则用观测到的最大坐标填充
    if max_x is None and observed_max_x:
        max_x = observed_max_x
    if max_y is None and observed_max_y:
        max_y = observed_max_y

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"{args.name}.json")
    data = {
        "version": 1,
        "created_at": datetime.now().isoformat(),
        "device_px": {"width": max_x, "height": max_y},
        "config_px": {"width": cfg_W, "height": cfg_H},
        "rotate": rotate_mode,
        "source": "getevent",
        "device": dev_info.get("path"),
        "events": events,
    }
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[DONE] 已保存 {len(events)} 条点击到 {out_path}")
    except KeyboardInterrupt:
        # 即便 Ctrl+C，也尝试保存已有数据
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[WARN] Ctrl+C 中断但已尝试保存 {len(events)} 条到 {out_path}")


if __name__ == "__main__":
    main()
