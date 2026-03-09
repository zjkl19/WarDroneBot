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
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum

from war_drone.adb_client import AdbClient
from war_drone.paddle_state_detector import PaddleStateDetector


def _pct_to_px(p: Tuple[float, float], wh: Tuple[int, int]) -> Tuple[int, int]:
    """将相对坐标转换为绝对像素坐标"""
    return int(p[0] * wh[0]), int(p[1] * wh[1])


class MacroState(Enum):
    """宏状态枚举"""
    IDLE = "idle"
    RUNNING = "running"
    STOPPING = "stopping"
    SCHEDULED = "scheduled"


@dataclass
class MacroEvent:
    """宏事件数据类"""
    type: str  # 'tap' 或 'swipe'
    pos: Optional[Tuple[float, float]] = None
    start: Optional[Tuple[float, float]] = None
    end: Optional[Tuple[float, float]] = None
    dt: float = 0.0
    duration: float = 0.3


class MacroController:
    """线程安全的宏控制器"""
    
    def __init__(self, adb_client: AdbClient, screen_size: Tuple[int, int]):
        self.adb = adb_client
        self.W, self.H = screen_size
        
        # 线程同步
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        
        # 状态变量
        self._state = MacroState.IDLE
        self._thread: Optional[threading.Thread] = None
        self._events: List[MacroEvent] = []
        self._scheduled_time: Optional[float] = None
        
        # 配置参数
        self.loops: int = 1
        self.scale: float = 1.0
        
    def load_macro(self, filepath: str) -> bool:
        """加载宏文件"""
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            events_data = data.get("events", [])
            events = []
            for ev in events_data:
                event = MacroEvent(
                    type=ev.get("type", "tap"),
                    dt=float(ev.get("dt", 0.0))
                )
                if event.type == "tap":
                    pos = ev.get("pos")
                    if pos and len(pos) == 2:
                        event.pos = (float(pos[0]), float(pos[1]))
                elif event.type == "swipe":
                    start = ev.get("start")
                    end = ev.get("end")
                    if start and len(start) == 2 and end and len(end) == 2:
                        event.start = (float(start[0]), float(start[1]))
                        event.end = (float(end[0]), float(end[1]))
                        event.duration = float(ev.get("duration", 0.3))
                events.append(event)
            
            with self._lock:
                self._events = events
            print(f"[INFO] loaded combat macro {filepath}, events={len(events)}")
            return True
            
        except Exception as e:
            print(f"[WARN] 无法读取 combat 宏 {filepath}: {e}")
            return False
    
    def configure(self, loops: int, scale: float):
        """配置宏参数"""
        with self._lock:
            self.loops = loops
            self.scale = scale
    
    def start(self, reason: str = ""):
        """启动宏（非阻塞）"""
        with self._lock:
            if self._state == MacroState.RUNNING:
                return
            
            # 停止当前宏
            self._stop()
            
            # 准备启动
            self._stop_event.clear()
            self._state = MacroState.RUNNING
            self._scheduled_time = None
            
            # 启动线程
            self._thread = threading.Thread(
                target=self._worker,
                name="MacroWorker",
                daemon=True
            )
            self._thread.start()
            
            if reason:
                print(f"[INFO] 宏启动: {reason}")
    
    def stop(self, reason: str = ""):
        """停止宏（非阻塞）"""
        with self._lock:
            self._stop()
        if reason:
            print(f"[INFO] 宏停止: {reason}")
    
    def _stop(self):
        """内部停止方法（必须在锁内调用）"""
        if self._state == MacroState.RUNNING or self._state == MacroState.SCHEDULED:
            self._state = MacroState.STOPPING
            self._stop_event.set()
            self._scheduled_time = None
    
    def schedule(self, delay: float):
        """预约启动宏"""
        with self._lock:
            if not self._events:
                return
            self._scheduled_time = time.time() + delay
            self._state = MacroState.SCHEDULED
            print(f"[INFO] 已预约宏，将在 {delay:.2f}s 后启动")
    
    def check_scheduled(self) -> bool:
        """检查并执行预约的宏"""
        with self._lock:
            if (self._state == MacroState.SCHEDULED and 
                self._scheduled_time and 
                time.time() >= self._scheduled_time):
                self.start("预约执行")
                return True
        return False
    
    @property
    def is_running(self) -> bool:
        """检查宏是否在运行"""
        with self._lock:
            return self._state == MacroState.RUNNING
    
    @property
    def has_events(self) -> bool:
        """检查是否有宏事件"""
        with self._lock:
            return len(self._events) > 0
    
    def wait_for_completion(self, timeout: Optional[float] = None) -> bool:
        """等待宏完成（可选，用于退出时）"""
        thread = None
        with self._lock:
            if self._thread and self._thread.is_alive():
                thread = self._thread
        
        if thread:
            thread.join(timeout=timeout)
            return not thread.is_alive()
        return True
    
    def _worker(self):
        """宏工作线程"""
        # 复制配置到本地，避免锁竞争
        with self._lock:
            events = self._events.copy()
            loops = self.loops
            scale = self.scale
        
        if not events:
            with self._lock:
                self._state = MacroState.IDLE
            return
        
        try:
            idx = 0
            loop_idx = 0
            last_ts = time.time()
            stop_event = self._stop_event
            
            while loop_idx < loops and not stop_event.is_set():
                ev = events[idx]
                
                # 计算等待时间（使用绝对时间，避免累积误差）
                if idx == 0 and loop_idx == 0:
                    # 第一个事件不等待
                    wait_s = 0
                else:
                    expected_dt = ev.dt * scale
                    elapsed = time.time() - last_ts
                    wait_s = max(0, expected_dt - elapsed)
                
                if wait_s > 0.01:  # 只有显著大于0才等待
                    if stop_event.wait(wait_s):
                        break  # 被停止
                
                # 再次检查停止信号
                if stop_event.is_set():
                    break
                
                # 执行事件
                if ev.type == "tap" and ev.pos:
                    self._tap_pct(ev.pos, f"macro[{loop_idx+1}:{idx+1}]")
                elif ev.type == "swipe" and ev.start and ev.end:
                    self._swipe_pct(ev.start, ev.end, ev.duration * scale)
                    print(f"[ACTION] macro[{loop_idx+1}:{idx+1}] swipe dur={ev.duration*scale:.2f}s")
                
                last_ts = time.time()
                idx += 1
                
                if idx >= len(events):
                    loop_idx += 1
                    idx = 0
                    last_ts = time.time()
        
        finally:
            # 清理状态
            with self._lock:
                self._state = MacroState.IDLE
                self._thread = None
                self._stop_event.clear()
            print("[INFO] combat 宏播放结束")
    
    def _tap_pct(self, pos: Tuple[float, float], label: str = None):
        """点击相对坐标"""
        x, y = _pct_to_px(pos, (self.W, self.H))
        self.adb.tap(x, y)
        if label:
            print(f"[ACTION] {label} -> tap ({x},{y})")
    
    def _swipe_pct(self, start: Tuple[float, float], end: Tuple[float, float], duration_s: float):
        """滑动相对坐标"""
        base = [self.adb.adb]
        if self.adb.serial:
            base += ["-s", self.adb.serial]
        
        x1, y1 = _pct_to_px(start, (self.W, self.H))
        x2, y2 = _pct_to_px(end, (self.W, self.H))
        dur_ms = int(max(1, duration_s * 1000))
        
        cmd = base + ["shell", "input", "swipe", str(x1), str(y1), str(x2), str(y2), str(dur_ms)]
        try:
            subprocess.check_call(cmd, timeout=duration_s + 1)
        except subprocess.TimeoutExpired:
            print(f"[WARN] 滑动命令超时")
        except Exception as e:
            print(f"[WARN] 滑动命令失败: {e}")


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
    ap.add_argument("--combat-macro", default="recordings/mission12_01.json", help="combat 状态时播放的录制文件（JSON）")
    ap.add_argument("--combat-macro-loops", type=int, default=1, help="combat 宏循环次数")
    ap.add_argument("--macro-sleep-scale", type=float, default=1.0, help="宏事件间隔缩放系数")
    ap.add_argument("--max-combat", type=int, default=0, help="combat 状态执行的最大次数（0=不限制，按进入combat计数）")
    ap.add_argument("--prestart-macro", action="store_true", help="点击 ready 后延时播放宏，不等 OCR 判定 combat")
    ap.add_argument("--prestart-delay", type=float, default=0.0, help="ready 点击后延时多少秒启动宏")
    ap.add_argument("--quiet", action="store_true", help="减少日志输出（压低 paddleocr 日志）")
    args = ap.parse_args()

    # 配置日志
    if args.quiet:
        logging.getLogger("ppocr").setLevel(logging.ERROR)
        logging.getLogger("PIL").setLevel(logging.ERROR)
        logging.getLogger().setLevel(logging.WARNING)

    # 加载配置
    cfg = json5.load(open(args.cfg, "r", encoding="utf-8"))
    coords = cfg.get("coords", {})
    W, H = cfg["screen"]["width"], cfg["screen"]["height"]

    # 初始化组件
    det = PaddleStateDetector(args.cfg, det_dir=args.det_dir, rec_dir=args.rec_dir, cls_dir=args.cls_dir)
    adb = AdbClient(serial=args.serial)
    
    # 初始化宏控制器
    macro_ctrl = MacroController(adb, (W, H))
    
    # 加载宏
    if args.combat_macro:
        if macro_ctrl.load_macro(args.combat_macro):
            macro_ctrl.configure(args.combat_macro_loops, args.macro_sleep_scale)

    # 映射：状态 -> 相对坐标
    action_map = {
        "main_menu": (0.868165, 0.866667),
        "ready": (0.868165, 0.866667),
        "settlement": (0.150936, 0.85),
        "weapon": (0.098127, 0.543333),
        "free_gift": (0.315356, 0.781667),
        "mission_hard": (0.325468, 0.618333),
        "piggy_full": (0.819101, 0.188333),
        "vip_ad": (0.820225, 0.111667),
        "ad_other": (0.95, 0.08),
    }

    # 状态变量
    prev_state = None
    combat_count = 0
    exit_pending = False
    last_support_click = 0  # 用于combat自动点击的节流

    def tap_px(x, y, label=None):
        """点击绝对坐标"""
        adb.tap(x, y)
        if label:
            print(f"[ACTION] {label} -> tap ({x},{y})")

    def tap_pct(pos, label=None):
        """点击相对坐标"""
        if not pos:
            return
        x, y = _pct_to_px(pos, (W, H))
        tap_px(x, y, label=label)

    print("[INFO] paddle runner 启动，按 Ctrl+C 退出")
    
    try:
        while True:
            # 检查预约宏
            macro_ctrl.check_scheduled()
            
            # 截屏并识别状态
            img = adb.screencap()
            state, dbg = det.predict(img)
            
            # 打印状态（限制小数位数）
            scores_str = {k: round(v, 2) for k, v in dbg['scores'].items()}
            print(f"[STATE] {state} scores={scores_str}")

            # 离开 combat 时停止宏
            if state != "combat" and macro_ctrl.is_running:
                macro_ctrl.stop("离开 combat")

            if args.dry_run:
                prev_state = state
                time.sleep(args.interval)
                continue

            # 处理状态对应的操作
            pos = action_map.get(state)
            
            if pos:
                # 有对应的点击位置
                if state == "settlement":
                    # 结算界面：尝试多个位置
                    candidates = [pos, (0.92, 0.93)]
                    seen = set()
                    for c in candidates:
                        if not c:
                            continue
                        key = tuple(c)
                        if key in seen:
                            continue
                        seen.add(key)
                        x, y = _pct_to_px(c, (W, H))
                        tap_px(x, y, label=state)
                        time.sleep(0.05)  # 短暂延迟避免点击过快
                else:
                    x, y = _pct_to_px(pos, (W, H))
                    tap_px(x, y, label=state)
                
                # 点击 ready 后预约宏
                if state == "ready" and args.prestart_macro and macro_ctrl.has_events:
                    macro_ctrl.schedule(args.prestart_delay)
            
            elif state == "combat":
                # 处理 combat 状态
                if prev_state != "combat":
                    # 首次进入 combat
                    combat_count += 1
                    print(f"[INFO] 进入 combat #{combat_count}")
                    
                    # 检查次数限制
                    if args.max_combat > 0 and combat_count >= args.max_combat:
                        if macro_ctrl.has_events:
                            exit_pending = True
                            print(f"[INFO] combat 次数 {combat_count} 已达上限，等待宏结束")
                        else:
                            print(f"[INFO] combat 次数 {combat_count} 已达上限，结束循环")
                            break
                    
                    # 启动宏（如果有）
                    if macro_ctrl.has_events:
                        macro_ctrl.start("进入 combat")
                
                # 处理 combat 内的操作
                if macro_ctrl.is_running:
                    # 宏正在运行，不执行其他操作
                    pass
                elif args.combat_auto:
                    # 自动点击支持（带节流，避免点击过快）
                    now = time.time()
                    if now - last_support_click >= args.combat_sleep:
                        print("[INFO] combat 自动执行支持点击")
                        clicked = False
                        for i in range(1, 7):
                            key = f"support{i}"
                            if key in coords:
                                sx, sy = _pct_to_px(coords[key], (W, H))
                                tap_px(sx, sy, label=f"combat->{key}")
                                clicked = True
                                time.sleep(0.1)  # 短暂延迟
                        if clicked:
                            last_support_click = now
                else:
                    # 无操作
                    pass
            
            # 检查退出条件
            if exit_pending and not macro_ctrl.is_running:
                print(f"[INFO] combat 次数 {combat_count} 已达上限 {args.max_combat}，宏已结束，退出循环")
                break

            prev_state = state
            time.sleep(args.interval)
            
    except KeyboardInterrupt:
        print("\n[INFO] 收到中断信号，正在停止...")
        # 停止宏（非阻塞）
        if macro_ctrl.is_running:
            macro_ctrl.stop("用户中断")
            # 等待最多2秒让宏退出
            macro_ctrl.wait_for_completion(timeout=2.0)
        print("[INFO] 已退出")
    except Exception as e:
        print(f"[ERROR] 运行时错误: {e}")
        # 确保宏被停止
        if macro_ctrl.is_running:
            macro_ctrl.stop("错误退出")
    finally:
        # 清理资源
        print("[INFO] 清理资源...")


if __name__ == "__main__":
    main()