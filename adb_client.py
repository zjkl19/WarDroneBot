import subprocess
import threading
from typing import Optional, Tuple


class AdbClient:
    def __init__(self, serial: Optional[str] = None, adb_path: str = "adb"):
        self.serial = serial
        self.adb = adb_path
        self._lock = threading.Lock()

    def _build_cmd(self, *parts: str):
        cmd = [self.adb]
        if self.serial:
            cmd += ["-s", self.serial]
        cmd += list(parts)
        return cmd

    def tap(self, x: int, y: int):
        """点击指定坐标"""
        with self._lock:
            cmd = self._build_cmd("shell", "input", "tap", str(int(x)), str(int(y)))
            try:
                subprocess.check_call(cmd, timeout=5)
            except subprocess.TimeoutExpired:
                print(f"[WARN] 点击命令超时 ({x},{y})")
            except Exception as e:
                print(f"[WARN] 点击失败 ({x},{y}): {e}")

    def swipe(self, start: Tuple[int, int], end: Tuple[int, int], duration_s: float = 0.3):
        """滑动"""
        with self._lock:
            x1, y1 = start
            x2, y2 = end
            dur_ms = int(max(1, float(duration_s) * 1000))
            cmd = self._build_cmd(
                "shell",
                "input",
                "swipe",
                str(int(x1)),
                str(int(y1)),
                str(int(x2)),
                str(int(y2)),
                str(dur_ms),
            )
            try:
                subprocess.check_call(cmd, timeout=max(3.0, float(duration_s) + 2.0))
                print(f"[ACTION] swipe ({x1},{y1})->({x2},{y2}) dur={duration_s:.2f}s")
            except subprocess.TimeoutExpired:
                print("[WARN] 滑动命令超时")
            except Exception as e:
                print(f"[WARN] 滑动命令失败: {e}")

    def screencap(self) -> bytes:
        """截屏，返回 PNG bytes。"""
        with self._lock:
            cmd = self._build_cmd("exec-out", "screencap", "-p")
            try:
                data = subprocess.check_output(cmd, timeout=10)
                # 某些设备通过 adb 传输时会把 PNG 的 LF 变成 CRLF
                if b"\r\n" in data:
                    data = data.replace(b"\r\n", b"\n")
                return data
            except subprocess.TimeoutExpired:
                print("[WARN] 截屏超时")
                raise
            except Exception as e:
                print(f"[WARN] 截屏失败: {e}")
                raise
