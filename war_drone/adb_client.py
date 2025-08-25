import subprocess, time, random

class ADBClient:
    def __init__(self, serial=None):
        self.serial = serial

    def _adb(self, *args) -> bytes:
        cmd = ["adb"]
        if self.serial: cmd += ["-s", self.serial]
        cmd += list(args)
        return subprocess.check_output(cmd)

    def tap(self, x, y):
        self._adb("shell","input","tap", str(int(x)), str(int(y)))

    def swipe(self, x1, y1, x2, y2, ms=300):
        self._adb("shell","input","swipe", str(int(x1)), str(int(y1)),
                  str(int(x2)), str(int(y2)), str(int(ms)))

    def key(self, code):
        self._adb("shell","input","keyevent", str(code))

    def launch(self, pkg):
        self._adb("shell","monkey","-p", pkg, "-c",
                  "android.intent.category.LAUNCHER", "1")

    def kill(self, pkg):
        self._adb("shell","am","force-stop", pkg)

    def screencap(self) -> bytes:
        # PNG bytes
        return self._adb("exec-out", "screencap", "-p")