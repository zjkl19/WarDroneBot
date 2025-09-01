# war_drone/adb_client.py
import os
import subprocess
import random
import numpy as np
import cv2

class AdbClient:
    def __init__(self, serial=None):
        self.serial = serial
        self.adb = r"C:\Android\platform-tools\adb.exe"  # 你的 adb 路径
        if not os.path.exists(self.adb):
            # 也允许仅用 "adb"（已配到 PATH）
            self.adb = "adb"

    def _cmd(self, args, capture_output=False):
        base = [self.adb]
        if self.serial:
            base += ["-s", self.serial]
        base += args
        if capture_output:
            return subprocess.check_output(base)
        subprocess.check_call(base)

    def launch_package(self, pkg: str):
        # monkey 拉起桌面入口
        args = ["shell", "monkey", "-p", pkg, "-c", "android.intent.category.LAUNCHER", "1"]
        print("args:", args[2:])
        self._cmd(args)

    def screencap(self):
        # 返回 OpenCV BGR ndarray
        data = self._cmd(["exec-out", "screencap", "-p"], capture_output=True)
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        return img

    def tap(self, x: int, y: int):
        self._cmd(["shell", "input", "tap", str(x), str(y)])

    def rand_int(self, a, b):
        return random.randint(int(a), int(b))
