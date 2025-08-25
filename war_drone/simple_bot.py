import time, random
import json5  # ← 新增
from .adb_client import ADBClient

def _pct_to_px(p, wh, jitter):
    x = int(p[0]*wh[0] + random.randint(-jitter, jitter))
    y = int(p[1]*wh[1] + random.randint(-jitter, jitter))
    return x, y

class SimpleSupportBot:
    def __init__(self, cfg_path="configs/config.json5", serial=None):  # ← 默认后缀改为 .json5
        with open(cfg_path, "r", encoding="utf-8") as f:
            self.cfg = json5.load(f)  # ← 用 json5 解析
        self.adb = ADBClient(serial)
        self.wh = (self.cfg["screen"]["width"], self.cfg["screen"]["height"])
        self.c = self.cfg["coords"]
        self.t = self.cfg["timing"]
        self.cd = self.cfg["support_cd"]
        self.rand = self.cfg["random"]
        self.pkg = self.cfg["package"]

    def _tap(self, name, base_delay=0.12):
        x, y = _pct_to_px(self.c[name], self.wh, self.rand["tap_jitter_px"])
        self.adb.tap(x, y)
        time.sleep(base_delay + random.random()*self.rand["delay_jitter_s"])

    def _support_loop(self, duration_s):
        next_time = { "support1": 0.0, "support2": 0.0, "support3": 0.0, "support4": 0.0 }
        end = time.time() + duration_s
        while time.time() < end:
            now = time.time()
            for key in next_time:
                if now >= next_time[key]:
                    try: self._tap(key, base_delay=0.08)
                    except Exception: pass
                    next_time[key] = now + self.cd[key]
            time.sleep(1.0)

    def run_one_round(self):
        print("[INFO] Launch game")
        self.adb.launch(self.pkg)
        time.sleep(self.t["launch_wait_s"])

        print("[INFO] On task list → Start")
        self._tap("list_start")
        time.sleep(1.0)

        print("[INFO] Pre-battle → Start")
        self._tap("pre_start")
        time.sleep(self.t["prebattle_wait_s"])

        print("[INFO] Combat (support-only) ...")
        self._support_loop(self.t["combat_seconds"])

        print("[INFO] Wait for settlement")
        time.sleep(self.t["settle_wait_s"])

        print("[INFO] Settlement → Collect")
        self._tap("collect")
        time.sleep(self.t["post_collect_wait_s"])

    def sleep_for_energy(self):
        secs = self.cfg["energy"]["recover_minutes_per_point"]*60 - self.cfg["energy"]["sleep_safety_seconds"]
        if secs < 60: secs = 60
        print(f"[INFO] Sleeping {secs}s to recover energy...")
        time.sleep(secs)
