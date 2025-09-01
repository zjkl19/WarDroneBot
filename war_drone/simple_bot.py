# war_drone/simple_bot.py
import os
import time
import json5
import cv2
from datetime import datetime
from typing import Tuple, Union

from war_drone.adb_client import AdbClient
from war_drone.state_detector import TemplateStateDetector, States, DetectedState
from war_drone.logger import RunLogger

class SimpleSupportBot:
    """
    第一版：仅使用“支援”武器完成一局循环，包含：
      - 启动包名
      - 轮询识别状态：list / prebattle / combat / settlement（也可扩展）
      - 每个关键步骤：截图、落盘、日志（含点击点）
      - combat 阶段依次点击右下角 6 个支援位
    """
    def __init__(self, serial=None, use_edges=True, use_mask=True, debug=False):
        self.pkg = "com.miniclip.drone1"
        self.debug = debug

        self.cfg = json5.load(open("configs/config.json5", "r", encoding="utf-8"))
        self.W = self.cfg["screen"]["width"]
        self.H = self.cfg["screen"]["height"]
        self.coords = self.cfg["coords"]

        self.adb = AdbClient(serial=serial)
        self.det = TemplateStateDetector(
            cfg_path="configs/config.json5",
            templates_dir="templates",
            use_edges=use_edges,
            use_mask=use_mask,
            method="CCORR_NORMED",
            default_thresh=0.85,
        )

        # 运行会话日志目录
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join("runs", f"session_{stamp}")
        os.makedirs(self.run_dir, exist_ok=True)
        self.log = RunLogger(self.run_dir, verbose=debug)

        # 点击抖动（相对屏比例），非精确瞄准时使用
        self.jitter = 0.008

    # ---------- 基础工具 ----------

    def _pct_to_px(self, p):
        return int(p[0]*self.W), int(p[1]*self.H)

    def _screencap_bgr(self):
        img = self.adb.screencap()
        path = self.log.save_image(img)
        return img, path

    def _click_pct(self, p: Tuple[float,float], label: str, precise=False):
        x, y = self._pct_to_px(p)
        if not precise:
            dx = int(self.W * self.jitter)
            dy = int(self.H * self.jitter)
            x = max(0, min(self.W-1, self.adb.rand_int(x-dx, x+dx)))
            y = max(0, min(self.H-1, self.adb.rand_int(y-dy, y+dy)))

        self.adb.tap(x, y)
        self.log.info(f"[CLICK] {label} at ({x},{y})")

        img, _ = self._screencap_bgr()
        vis = img.copy()
        cv2.circle(vis, (x, y), 16, (0,255,0), 2)
        self.log.save_overlay(vis, suffix=f"click_{label}")

    def _wait_for_state(self, target: Union[States, str], timeout=30, poll=1.0) -> Tuple[bool, DetectedState]:
        """
        轮询识别直到目标状态或超时；target 可为 "list" / States.LIST 等。
        """
        target_name = target.value if isinstance(target, States) else str(target)
        end_ts = time.time() + timeout
        last = None
        while time.time() < end_ts:
            bgr, cap_path = self._screencap_bgr()
            r = self.det.predict(img_bgr=bgr, margin=0.12)
            self.log.info(f"[STATE] {os.path.basename(cap_path)} -> {r.name} score={r.score:.3f} by={r.template}")
            if r.name == target_name and r.score >= self.det.default_thresh:
                return True, r
            last = r
            time.sleep(poll)
        return False, last

    # ---------- 业务步骤（基础四状态流程） ----------

    def launch_game(self):
        self.log.info("[STEP] 启动游戏")
        self.adb.launch_package(self.pkg)
        time.sleep(8)  # 给足启动时间
        ok, _ = self._wait_for_state("list", timeout=40, poll=1.2)
        if not ok:
            self.log.warn("[WARN] 启动后未检测到列表页")

    def goto_prebattle(self):
        self.log.info("[STEP] 列表页 → 开始任务")
        self._click_pct(self.coords["list_start"], "list_start")
        ok, _ = self._wait_for_state("prebattle", timeout=25, poll=1.0)
        if not ok:
            self.log.warn("[WARN] 未到达 PREBATTLE，重试一次")
            self._click_pct(self.coords["list_start"], "list_start_retry")
            self._wait_for_state("prebattle", timeout=20, poll=1.0)

    def start_combat(self):
        self.log.info("[STEP] 战前 → 开始战斗")
        self._click_pct(self.coords["pre_start"], "pre_start")
        ok, _ = self._wait_for_state("combat", timeout=25, poll=1.0)
        if not ok:
            self.log.warn("[WARN] 未到达 COMBAT，重试一次")
            self._click_pct(self.coords["pre_start"], "pre_start_retry")
            self._wait_for_state("combat", timeout=20, poll=1.0)

    def combat_support_only(self):
        self.log.info("[STEP] 战斗阶段：依次释放 6 个支援")
        for i in range(1, 7):
            key = f"support{i}"
            if key not in self.coords:
                continue
            self._click_pct(self.coords[key], key)
            time.sleep(1.0)
        self.log.info("[STEP] 等待战斗结束 → 结算")
        ok, _ = self._wait_for_state("settlement", timeout=90, poll=1.5)
        if not ok:
            self.log.warn("[WARN] 90s 未识别到结算页，继续轮询 60s")
            self._wait_for_state("settlement", timeout=60, poll=1.5)

    def collect_and_back(self):
        self.log.info("[STEP] 结算页 → 领取/返回")
        self._click_pct(self.coords["collect"], "collect")
        self._wait_for_state("list", timeout=25, poll=1.0)

    # ---------- 主流程 ----------

    def run_one_cycle(self):
        self.log.section("RUN_ONE_CYCLE")
        self.launch_game()
        self.goto_prebattle()
        self.start_combat()
        self.combat_support_only()
        self.collect_and_back()
        self.log.section("CYCLE_DONE")
