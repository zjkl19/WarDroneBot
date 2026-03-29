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
import traceback
import os
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import cv2

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
        old_thread = None
        with self._lock:
            if self._state == MacroState.RUNNING:
                return

            if self._thread and self._thread.is_alive():
                self._stop()
                old_thread = self._thread

        if old_thread and old_thread is not threading.current_thread():
            old_thread.join(timeout=2.0)

        with self._lock:
            self._stop_event.clear()
            self._state = MacroState.RUNNING
            self._scheduled_time = None
            self._thread = threading.Thread(
                target=self._worker,
                name="MacroWorker",
                daemon=False
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

    def cancel_scheduled(self, reason: str = ""):
        """取消尚未开始的预约宏"""
        with self._lock:
            if self._state == MacroState.SCHEDULED:
                self._scheduled_time = None
                self._state = MacroState.IDLE
                self._stop_event.clear()
                if reason:
                    print(f"[INFO] 已取消预约宏: {reason}")
    
    def _stop(self):
        """内部停止方法（必须在锁内调用）"""
        if self._state == MacroState.RUNNING or self._state == MacroState.SCHEDULED:
            self._state = MacroState.STOPPING
            self._stop_event.set()
            self._scheduled_time = None
    
    def schedule(self, delay: float):
        """预约启动宏"""
        with self._lock:
            if not self._events or self._state == MacroState.RUNNING:
                return
            self._stop_event.clear()
            self._scheduled_time = time.time() + max(0.0, float(delay))
            self._state = MacroState.SCHEDULED
            print(f"[INFO] 已预约宏，将在 {delay:.2f}s 后启动")
    
    def check_scheduled(self) -> bool:
        """检查并执行预约的宏"""
        should_start = False
        with self._lock:
            if (self._state == MacroState.SCHEDULED and
                self._scheduled_time and
                time.time() >= self._scheduled_time):
                should_start = True
        if should_start:
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

        except Exception as e:
            print(f"[ERROR] MacroWorker 异常: {e}")
            traceback.print_exc()

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
        x1, y1 = _pct_to_px(start, (self.W, self.H))
        x2, y2 = _pct_to_px(end, (self.W, self.H))
        if hasattr(self.adb, "swipe"):
            self.adb.swipe((x1, y1), (x2, y2), duration_s)
            return

        base = [self.adb.adb]
        if self.adb.serial:
            base += ["-s", self.adb.serial]

        dur_ms = int(max(1, duration_s * 1000))
        cmd = base + ["shell", "input", "swipe", str(x1), str(y1), str(x2), str(y2), str(dur_ms)]
        try:
            subprocess.check_call(cmd, timeout=duration_s + 1)
        except subprocess.TimeoutExpired:
            print(f"[WARN] 滑动命令超时")
        except Exception as e:
            print(f"[WARN] 滑动命令失败: {e}")


class CombatVideoRecorder:
    """使用安卓端 screenrecord 录制 combat，并自动拉回电脑。"""

    def __init__(
        self,
        adb_client: AdbClient,
        output_dir: str,
        size: str,
        bitrate: int,
        keep_device_video: bool,
        overlay: bool,
        reverse_video: bool,
        remote_dir: str = "/sdcard/Movies",
    ):
        self.adb = adb_client
        self.output_dir = output_dir
        self.size = size
        self.bitrate = max(100_000, int(bitrate))
        self.keep_device_video = keep_device_video
        self.overlay = overlay
        self.reverse_video = reverse_video
        self.remote_dir = remote_dir.rstrip("/") or "/sdcard"

        self._lock = threading.RLock()
        self._running = False
        self._proc: Optional[subprocess.Popen] = None
        self._last_path: Optional[str] = None
        self._remote_path: Optional[str] = None

    def _adb_base(self) -> List[str]:
        base = [self.adb.adb]
        if self.adb.serial:
            base += ["-s", self.adb.serial]
        return base

    def _run(self, args: List[str], capture_output: bool = False, timeout: Optional[float] = None):
        cmd = self._adb_base() + args
        if capture_output:
            return subprocess.check_output(cmd, timeout=timeout)
        subprocess.check_call(cmd, timeout=timeout)
        return None

    def _build_paths(self) -> Tuple[str, str]:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"combat_{stamp}.mp4"
        local_path = os.path.join(self.output_dir, filename)
        remote_path = f"{self.remote_dir}/{filename}"
        return local_path, remote_path

    def _build_reverse_path(self, local_path: str) -> str:
        stem, ext = os.path.splitext(local_path)
        return f"{stem}_reverse{ext}"

    @property
    def is_running(self) -> bool:
        with self._lock:
            if self._proc and self._proc.poll() is not None:
                self._running = False
            return self._running

    @property
    def last_path(self) -> Optional[str]:
        with self._lock:
            return self._last_path

    def update_overlay(self, state: str, macro_state: str, combat_count: int, scores: Dict[str, float]):
        # 安卓端 screenrecord 不支持实时叠加，这里保留接口以兼容主循环调用。
        return

    def _write_reverse_copy(self, local_path: str) -> Optional[str]:
        cap = cv2.VideoCapture(local_path)
        if not cap.isOpened():
            print(f"[WARN] 无法打开录像，跳过倒放生成: {local_path}")
            return None

        try:
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            if frame_count <= 0 or width <= 0 or height <= 0:
                print(f"[WARN] 录像元数据无效，跳过倒放生成: {local_path}")
                return None
            if fps <= 0:
                fps = 10.0

            reverse_path = self._build_reverse_path(local_path)
            writer = cv2.VideoWriter(
                reverse_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (width, height),
            )
            if not writer.isOpened():
                print(f"[WARN] 无法创建倒放文件，跳过: {reverse_path}")
                return None

            try:
                for idx in range(frame_count - 1, -1, -1):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ok, frame = cap.read()
                    if not ok:
                        print(f"[WARN] 倒放生成时读取第 {idx} 帧失败，已提前结束")
                        break
                    writer.write(frame)
            finally:
                writer.release()

            return reverse_path
        finally:
            cap.release()

    def start(self, reason: str = ""):
        local_path = None
        remote_path = None
        with self._lock:
            if self._running:
                return

            if self.overlay:
                print("[WARN] 安卓端 screenrecord 不支持实时叠加调试信息，已忽略 --record-video-overlay")

            os.makedirs(self.output_dir, exist_ok=True)
            local_path, remote_path = self._build_paths()
            self._last_path = local_path
            self._remote_path = remote_path

            try:
                self._run(["shell", "mkdir", "-p", self.remote_dir], timeout=10)
            except Exception as e:
                print(f"[WARN] 创建手机录像目录失败，将继续尝试录制: {e}")

            cmd = self._adb_base() + [
                "shell",
                "screenrecord",
                "--bit-rate", str(self.bitrate),
                "--size", self.size,
                remote_path,
            ]
            self._proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            time.sleep(0.5)
            if self._proc.poll() is not None:
                self._running = False
                self._proc = None
                print(f"[WARN] 安卓端录像启动失败: {remote_path}")
                return
            self._running = True
        if reason:
            print(f"[INFO] 录像启动: {reason} -> {local_path}")

    def stop(self, reason: str = ""):
        proc = None
        remote_path = None
        local_path = None
        with self._lock:
            if not self._running:
                return
            proc = self._proc
            remote_path = self._remote_path
            local_path = self._last_path
            self._running = False
            self._proc = None

        try:
            if proc and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=2.0)
        except Exception as e:
            print(f"[WARN] 停止安卓端录像进程失败: {e}")

        # 给设备一点时间完成 mp4 封装
        time.sleep(1.0)

        pulled = False
        if remote_path and local_path:
            try:
                self._run(["pull", remote_path, local_path], timeout=180)
                pulled = True
            except Exception as e:
                print(f"[WARN] 拉回录像失败: {e}")

            if not self.keep_device_video:
                try:
                    self._run(["shell", "rm", "-f", remote_path], timeout=10)
                except Exception as e:
                    print(f"[WARN] 删除手机临时录像失败: {e}")

        if reason:
            print(f"[INFO] 录像停止: {reason}")
        if pulled and local_path:
            print(f"[INFO] combat 录像已保存: {local_path}")
            if self.reverse_video:
                reverse_path = self._write_reverse_copy(local_path)
                if reverse_path:
                    print(f"[INFO] combat 倒放已保存: {reverse_path}")


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
    ap.add_argument("--record-combat-video", action="store_true", help="仅在 combat 状态时录制视频")
    ap.add_argument("--record-video-dir", default="recordings/videos", help="combat 录像输出目录")
    ap.add_argument("--record-video-size", default="1280x576", help="安卓端录像分辨率，例如 1280x576")
    ap.add_argument("--record-video-bitrate", type=int, default=3000000, help="安卓端录像码率，单位 bps")
    ap.add_argument("--record-video-remote-dir", default="/sdcard/Movies", help="安卓端临时录像目录")
    ap.add_argument("--keep-device-video", action="store_true", help="保留手机上的临时录像文件")
    ap.add_argument("--record-video-overlay", action="store_true", help="在录像中叠加状态/得分等调试信息")
    ap.add_argument("--record-video-reverse", action="store_true", help="额外生成一个本地倒放视频")
    ap.add_argument("--quiet", action="store_true", help="减少日志输出（压低 paddleocr 日志）")
    args = ap.parse_args()

    # 配置日志
    if args.quiet:
        logging.getLogger("ppocr").setLevel(logging.ERROR)
        logging.getLogger("PIL").setLevel(logging.ERROR)
        logging.getLogger().setLevel(logging.WARNING)

    # 加载配置
    with open(args.cfg, "r", encoding="utf-8") as f:
        cfg = json5.load(f)
    coords = cfg.get("coords", {})
    W, H = cfg["screen"]["width"], cfg["screen"]["height"]

    # 初始化组件
    det = PaddleStateDetector(args.cfg, det_dir=args.det_dir, rec_dir=args.rec_dir, cls_dir=args.cls_dir)
    adb = AdbClient(serial=args.serial)
    
    # 初始化宏控制器
    macro_ctrl = MacroController(adb, (W, H))

    # 初始化录像器
    video_recorder = None
    if args.record_combat_video:
        video_recorder = CombatVideoRecorder(
            adb_client=adb,
            output_dir=args.record_video_dir,
            size=args.record_video_size,
            bitrate=args.record_video_bitrate,
            keep_device_video=args.keep_device_video,
            overlay=args.record_video_overlay,
            reverse_video=args.record_video_reverse,
            remote_dir=args.record_video_remote_dir,
        )
    
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
        "bankrupt_sale": (0.261, 0.798333),
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
            scores_str = {k: round(v, 2) for k, v in dbg.get('scores', {}).items()}
            print(f"[STATE] {state} scores={scores_str}")

            if video_recorder:
                video_recorder.update_overlay(
                    state=state,
                    macro_state="running" if macro_ctrl.is_running else "idle",
                    combat_count=combat_count,
                    scores=scores_str,
                )

            # 离开 ready/combat 流程时取消预约；离开 combat 时停止宏
            if state not in ("ready", "combat"):
                macro_ctrl.cancel_scheduled("离开 ready/combat 流程")
            if state != "combat" and macro_ctrl.is_running:
                macro_ctrl.stop("离开 combat")
            if state != "combat" and video_recorder and video_recorder.is_running:
                video_recorder.stop("离开 combat")

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
                    candidates = [pos, (0.86, 0.86)]
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
                if state == "ready" and prev_state != "ready" and args.prestart_macro and macro_ctrl.has_events:
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
                    if video_recorder:
                        video_recorder.start("进入 combat")
                
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
        if video_recorder and video_recorder.is_running:
            video_recorder.stop("用户中断")
        print("[INFO] 已退出")
    except Exception as e:
        print(f"[ERROR] 运行时错误: {e}")
        # 确保宏被停止
        if macro_ctrl.is_running:
            macro_ctrl.stop("错误退出")
        if video_recorder and video_recorder.is_running:
            video_recorder.stop("错误退出")
    finally:
        print("[INFO] 清理资源...")
        macro_ctrl.cancel_scheduled("程序结束")
        if macro_ctrl.is_running:
            macro_ctrl.stop("程序结束")
        macro_ctrl.wait_for_completion(timeout=3.0)
        if video_recorder and video_recorder.is_running:
            video_recorder.stop("程序结束")


if __name__ == "__main__":
    main()
