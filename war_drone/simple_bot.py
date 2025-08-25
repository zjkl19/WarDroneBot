import time, random, json5, os
from .adb_client import ADBClient
from .vis import annotate_click

def _pct_to_px(p, wh, jitter):
    x = int(p[0]*wh[0] + (random.randint(-jitter, jitter) if jitter>0 else 0))
    y = int(p[1]*wh[1] + (random.randint(-jitter, jitter) if jitter>0 else 0))
    return x, y

class SimpleSupportBot:
    def __init__(self, cfg_path="configs/config.json5", serial=None, logger=None):
        with open(cfg_path, "r", encoding="utf-8") as f:
            self.cfg = json5.load(f)
        self.adb = ADBClient(serial)
        self.wh = (self.cfg["screen"]["width"], self.cfg["screen"]["height"])
        self.c  = self.cfg["coords"]
        self.t  = self.cfg["timing"]
        self.cd = self.cfg["support_cd"]
        self.pkg = self.cfg["package"]
        self.log = logger
        # 点击档
        cp = self.cfg.get("click_profiles", {})
        self.profile_relaxed = cp.get("relaxed", {"jitter_px": 10, "base_delay": 0.12, "delay_jitter_s": 0.20})
        self.profile_precise = cp.get("precise", {"jitter_px": 0,  "base_delay": 0.08, "delay_jitter_s": 0.00})
        # 文件名序号
        self.step_id = 0

    def _save_shot(self, tag, img_bytes=None, click_xy=None):
        """
        保存当前屏幕或传入的bytes；若提供 click_xy，则叠加十字+label。
        """
        if img_bytes is None:
            img_bytes = self.adb.screencap()
        label = f"{tag}" + (f" @({click_xy[0]},{click_xy[1]})" if click_xy else "")
        if click_xy:
            img_bytes = annotate_click(img_bytes, click_xy[0], click_xy[1], label)
        fname = f"{self.step_id:03d}_{tag.replace(' ','_')}.png"
        self.step_id += 1
        path = self.log.save_png_bytes(img_bytes, fname) if self.log else None
        if self.log: self.log.info("Saved", fname)
        return path

    def _tap(self, name, profile="relaxed", jitter_override=None, base_delay_override=None, delay_jitter_override=None):
        prof = self.profile_precise if profile=="precise" else self.profile_relaxed
        jitter_px = prof["jitter_px"] if jitter_override is None else int(jitter_override)
        base_delay = prof["base_delay"] if base_delay_override is None else float(base_delay_override)
        delay_jitter_s = prof["delay_jitter_s"] if delay_jitter_override is None else float(delay_jitter_override)
        x, y = _pct_to_px(self.c[name], self.wh, jitter_px)

        # 点击前截图
        if self.log: self._save_shot(f"before_tap_{name}")
        # 点击
        if self.log: self.log.info(f"TAP {name} at ({x},{y}) [profile={profile}]")
        self.adb.tap(x, y)
        # 点击后截图（叠加十字）
        after = self.adb.screencap()
        self._save_shot(f"after_tap_{name}_{x},{y}", img_bytes=after, click_xy=(x,y))

        time.sleep(base_delay + (random.random()*delay_jitter_s if delay_jitter_s>0 else 0.0))

    def _support_loop(self, duration_s):
        support_keys = sorted([k for k in self.c.keys() if k.startswith("support")])
        next_time = { k: 0.0 for k in support_keys }

        if self.log:
            self.log.info(f"[COMBAT] support-only for {duration_s}s :: keys={support_keys}")
            self._save_shot("combat_before")

        end = time.time() + duration_s
        while time.time() < end:
            now = time.time()
            for key in support_keys:
                cd = float(self.cd.get(key, 30.0))
                if now >= next_time[key]:
                    try:
                        self._tap(key, profile="relaxed")  # 支援用“带抖动”的档
                    except Exception as e:
                        if self.log: self.log.err("tap error:", e)
                    next_time[key] = now + cd
            time.sleep(1.0)

        if self.log:
            self._save_shot("combat_after_loop")

    def run_one_round(self):
        # 启动
        if self.log: self.log.info("[LAUNCH] starting", self.pkg)
        self.adb.launch(self.pkg)
        if self.log: self._save_shot("launch_after")
        time.sleep(self.t["launch_wait_s"])

        # 列表页
        if self.log: self.log.info("[LIST] Start")
        self._tap("list_start")
        time.sleep(1.0)

        # 战前
        if self.log: self.log.info("[PREBATTLE] Start")
        self._tap("pre_start")
        time.sleep(self.t["prebattle_wait_s"])

        # 战斗
        if self.log: self.log.info("[COMBAT] loop begin")
        self._support_loop(self.t["combat_seconds"])

        # 结算
        if self.log: self.log.info("[SETTLEMENT] Wait & Collect")
        time.sleep(self.t["settle_wait_s"])
        self._tap("collect")
        time.sleep(self.t["post_collect_wait_s"])

    def sleep_for_energy(self):
        mins = self.cfg["energy"]["recover_minutes_per_point"]
        safe = self.cfg["energy"]["sleep_safety_seconds"]
        secs = max(mins*60 - safe, 60)
        if self.log: self.log.info(f"[ENERGY] sleep {secs}s for recovery")
        time.sleep(secs)
