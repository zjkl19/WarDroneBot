"""
目标锁定测试脚本 - 异步优化版（带滑动箭头可视化）
功能：
1. 多线程截图（持续截屏，不阻塞）
2. YOLO检测线程
3. 主线程处理显示和控制
4. 使用队列传递数据
5. 可视化ADB滑动箭头

使用方法：
python -m scripts.aim_test --serial e5081c2a --model runs/detect/train4/weights/best.pt --show-preview
"""

import argparse
import csv
import json5
import math
import time
import os
import threading
import queue
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from collections import deque
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple

import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont

from war_drone.adb_client import AdbClient


# ==================== 性能监控 ====================
class PerformanceMonitor:
    """性能监控器"""
    def __init__(self, window_size=30):
        self.window_size = window_size
        self.metrics = {
            'screenshot': deque(maxlen=window_size),
            'yolo': deque(maxlen=window_size),
            'total': deque(maxlen=window_size),
            'adb': deque(maxlen=window_size),
        }
        self.lock = threading.Lock()
    
    def add_time(self, metric: str, duration: float):
        with self.lock:
            if metric in self.metrics:
                self.metrics[metric].append(duration)
    
    def get_fps(self, metric: str) -> float:
        with self.lock:
            times = list(self.metrics.get(metric, []))
            if not times:
                return 0.0
            avg_time = sum(times) / len(times)
            return 1.0 / avg_time if avg_time > 0 else 0.0
    
    def get_avg_time(self, metric: str) -> float:
        with self.lock:
            times = list(self.metrics.get(metric, []))
            if not times:
                return 0.0
            return sum(times) / len(times)
    
    def get_stats(self) -> Dict[str, float]:
        stats = {}
        for metric in self.metrics:
            stats[f'{metric}_fps'] = self.get_fps(metric)
            stats[f'{metric}_ms'] = self.get_avg_time(metric) * 1000
        return stats


# ==================== 数据类 ====================
@dataclass
class DetectionResult:
    """检测结果数据类"""
    frame_id: int
    timestamp: float
    image: Optional[np.ndarray]
    detections: List[Dict]
    img_w: int
    img_h: int
    center_x: int
    center_y: int


@dataclass
class ControlCommand:
    """控制命令数据类"""
    dx: float
    dy: float
    dist: float
    slide_x: float
    slide_y: float
    slide_dist: float
    target_id: int
    target_name: str
    start_x_px: int = 0  # 滑动起点像素坐标
    start_y_px: int = 0
    end_x_px: int = 0    # 滑动终点像素坐标
    end_y_px: int = 0


# ==================== 截图线程 ====================
class ScreenshotThread(threading.Thread):
    """专用截图线程"""
    def __init__(self, adb_client: AdbClient, output_queue: queue.Queue, 
                 stop_event: threading.Event, perf_monitor: PerformanceMonitor):
        super().__init__(name="ScreenshotThread", daemon=True)
        self.adb = adb_client
        self.output_queue = output_queue
        self.stop_event = stop_event
        self.perf_monitor = perf_monitor
        self.frame_id = 0
        self.fail_count = 0
        
    def run(self):
        print("[截图线程] 已启动")
        while not self.stop_event.is_set():
            try:
                t0 = time.time()
                
                # 截屏
                bgr = self.adb.screencap()
                
                if bgr is None:
                    self.fail_count += 1
                    sleep_time = min(0.1 * self.fail_count, 1.0)
                    time.sleep(sleep_time)
                    continue
                
                self.fail_count = 0
                self.frame_id += 1
                
                # 记录截图时间
                screenshot_time = time.time() - t0
                self.perf_monitor.add_time('screenshot', screenshot_time)
                
                # 将截图放入队列（如果队列满了就丢弃旧数据）
                if self.output_queue.qsize() < 5:  # 限制队列大小
                    self.output_queue.put({
                        'frame_id': self.frame_id,
                        'timestamp': t0,
                        'image': bgr,
                        'img_h': bgr.shape[0],
                        'img_w': bgr.shape[1],
                    })
                else:
                    # 队列满了，尝试清空一个旧数据
                    try:
                        self.output_queue.get_nowait()
                        self.output_queue.put({
                            'frame_id': self.frame_id,
                            'timestamp': t0,
                            'image': bgr,
                            'img_h': bgr.shape[0],
                            'img_w': bgr.shape[1],
                        })
                    except queue.Empty:
                        pass
                
            except Exception as e:
                print(f"[截图线程] 错误: {e}")
                time.sleep(0.1)
        
        print("[截图线程] 已停止")


# ==================== YOLO检测线程 ====================
class YOLOThread(threading.Thread):
    """专用YOLO检测线程"""
    def __init__(self, model_path: str, classes: List[str], input_queue: queue.Queue,
                 output_queue: queue.Queue, stop_event: threading.Event,
                 perf_monitor: PerformanceMonitor, device: str = "0", 
                 imgsz: int = 2670, conf: float = 0.35, max_det: int = 20):
        super().__init__(name="YOLOThread", daemon=True)
        self.model_path = model_path
        self.classes = classes
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.stop_event = stop_event
        self.perf_monitor = perf_monitor
        self.device = device
        self.imgsz = imgsz
        self.conf = conf
        self.max_det = max_det
        self.use_half = str(device).lower() != "cpu"
        
    def run(self):
        print("[YOLO线程] 正在加载模型...")
        try:
            self.model = YOLO(self.model_path)
            print(f"[YOLO线程] 模型加载完成，设备: {self.device}")
        except Exception as e:
            print(f"[YOLO线程] 模型加载失败: {e}")
            return
        
        while not self.stop_event.is_set():
            try:
                # 从队列获取截图数据（最多等待0.1秒）
                try:
                    data = self.input_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                
                t0 = time.time()
                
                # 执行YOLO检测
                results = self.model.predict(
                    data['image'],
                    imgsz=self.imgsz,
                    conf=self.conf,
                    device=self.device,
                    verbose=False,
                    half=self.use_half,
                    augment=False,
                    max_det=self.max_det,
                )[0]
                
                # 解析检测结果
                detections = []
                if results.boxes is not None and len(results.boxes) > 0:
                    boxes = results.boxes.xyxy.detach().cpu().numpy()
                    cls_list = results.boxes.cls.detach().cpu().numpy().astype(int)
                    conf_list = results.boxes.conf.detach().cpu().numpy()
                    
                    for box, cls_idx, score in zip(boxes, cls_list, conf_list):
                        if cls_idx < 0 or cls_idx >= len(self.classes):
                            continue
                        
                        x1, y1, x2, y2 = box
                        w = x2 - x1
                        h = y2 - y1
                        if w < 2 or h < 2:
                            continue
                        
                        cx = (x1 + x2) / 2.0
                        cy = (y1 + y2) / 2.0
                        
                        detections.append({
                            "name": self.classes[cls_idx],
                            "conf": float(score),
                            "cx": float(cx),
                            "cy": float(cy),
                            "box": np.array([x1, y1, x2, y2], dtype=float),
                            "track_id": None,
                        })
                
                # 记录YOLO时间
                yolo_time = time.time() - t0
                self.perf_monitor.add_time('yolo', yolo_time)
                
                # 创建结果对象
                result = DetectionResult(
                    frame_id=data['frame_id'],
                    timestamp=data['timestamp'],
                    image=data['image'],
                    detections=detections,
                    img_w=data['img_w'],
                    img_h=data['img_h'],
                    center_x=data['img_w'] // 2,
                    center_y=data['img_h'] // 2
                )
                
                # 将结果放入输出队列
                if self.output_queue.qsize() < 3:  # 限制队列大小
                    self.output_queue.put(result)
                
            except Exception as e:
                print(f"[YOLO线程] 错误: {e}")
                time.sleep(0.01)
        
        print("[YOLO线程] 已停止")


# ==================== ADB控制线程 ====================
class ADBControlThread(threading.Thread):
    """专用ADB控制线程"""
    def __init__(self, adb_client: AdbClient, input_queue: queue.Queue,
                 stop_event: threading.Event, perf_monitor: PerformanceMonitor,
                 args, img_w: int, img_h: int):
        super().__init__(name="ADBThread", daemon=True)
        self.adb = adb_client
        self.input_queue = input_queue
        self.stop_event = stop_event
        self.perf_monitor = perf_monitor
        self.args = args
        self.img_w = img_w
        self.img_h = img_h
        self.last_command_time = 0
        self.min_command_interval = 0.03  # 最小命令间隔（约33Hz）
        self.last_cmd = None  # 保存最后一次命令用于可视化
        
    def run(self):
        print("[ADB线程] 已启动")
        while not self.stop_event.is_set():
            try:
                # 从队列获取控制命令
                try:
                    cmd = self.input_queue.get(timeout=0.01)
                except queue.Empty:
                    continue
                
                # 限流控制
                now = time.time()
                if now - self.last_command_time < self.min_command_interval:
                    time.sleep(self.min_command_interval - (now - self.last_command_time))
                
                t0 = time.time()
                
                # 执行滑动
                if cmd.slide_dist > 0 and not self.args.dry_run:
                    self._execute_swipe(cmd)
                
                # 保存最后执行的命令用于可视化
                self.last_cmd = cmd
                
                # 记录ADB时间
                adb_time = time.time() - t0
                self.perf_monitor.add_time('adb', adb_time)
                self.last_command_time = time.time()
                
            except Exception as e:
                print(f"[ADB线程] 错误: {e}")
        
        print("[ADB线程] 已停止")
    
    def _execute_swipe(self, cmd: ControlCommand):
        """执行滑动命令"""
        # 计算滑动方向和长度
        slide_dist = cmd.slide_dist
        if slide_dist <= 0:
            return
        
        # 直接使用 cmd 中的 slide_x, slide_y 作为方向向量
        # 注意：这里已经是计算好的滑动向量，包含了方向和距离
        dir_x = cmd.slide_x / slide_dist
        dir_y = cmd.slide_y / slide_dist
        
        # 滑动起点（屏幕中央偏下）- 使用固定位置，更容易观察
        start_x_px = self.img_w // 2
        start_y_px = self.img_h // 2   # 中央偏下0像素
        
        # 滑动长度直接使用 slide_dist，但限制范围
        swipe_distance = min(max(slide_dist, 20), 150)  # 20-150像素范围
        
        # 计算终点 - 使用相同的方向
        end_x_px = int(start_x_px + dir_x * swipe_distance)
        end_y_px = int(start_y_px + dir_y * swipe_distance)
        
        # 确保终点在屏幕范围内
        end_x_px = max(0, min(self.img_w, end_x_px))
        end_y_px = max(0, min(self.img_h, end_y_px))
        
        # 保存坐标到命令对象
        cmd.start_x_px = start_x_px
        cmd.start_y_px = start_y_px
        cmd.end_x_px = end_x_px
        cmd.end_y_px = end_y_px
        
        # 执行滑动
        dur_ms = int(self.args.swipe_duration * 500)
        self.adb._cmd(["shell", "input", "swipe", 
                    str(start_x_px), str(start_y_px), 
                    str(end_x_px), str(end_y_px), 
                    str(dur_ms)])
        
        # 调试输出
        if self.args.debug:
            print(f"\n[ADB] 滑动: ({cmd.slide_x:.1f}, {cmd.slide_y:.1f})px, "
                f"方向: ({dir_x:.2f}, {dir_y:.2f}), 距离: {swipe_distance:.1f}px")
    
    def get_last_command(self):
        """获取最后一次执行的命令（用于可视化）"""
        return self.last_cmd


# ==================== 辅助函数 ====================
def _pct_to_px(p, wh):
    """将相对坐标转换为像素坐标"""
    return int(p[0] * wh[0]), int(p[1] * wh[1])


def clamp(v, lo=0.0, hi=1.0):
    return max(lo, min(hi, v))


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--serial", default=None, help="adb 序列号")
    ap.add_argument("--model", required=True, help="YOLO 权重路径")
    ap.add_argument("--cfg", default="configs/yolo_combat.json5", help="配置文件")
    ap.add_argument("--device", default="0", help="YOLO device (0 for GPU, cpu for CPU)")
    ap.add_argument("--show-preview", action="store_true", help="显示检测预览窗口")
    ap.add_argument("--debug", action="store_true", help="显示详细调试信息")
    ap.add_argument("--save-csv", action="store_true", help="保存 CSV 日志")
    ap.add_argument("--log-dir", default="logs", help="日志输出目录")
    ap.add_argument("--max-lost", type=int, default=10, help="锁定目标最大允许丢失帧数")
    ap.add_argument("--max-match-dist", type=float, default=100.0, help="目标匹配最大距离")
    ap.add_argument("--imgsz", type=int, default=2670, help="YOLO 推理尺寸")
    ap.add_argument("--conf", type=float, default=0.35, help="YOLO 置信度阈值")
    ap.add_argument("--max-det", type=int, default=20, help="单帧最大检测数")
    ap.add_argument("--aim-tol", type=float, default=15.0, help="瞄准容差（像素）")
    ap.add_argument("--max-step", type=float, default=50.0, help="最大移动量（像素/帧）")
    ap.add_argument("--min-step", type=float, default=5.0, help="最小移动量（像素/帧）")
    ap.add_argument("--swipe-duration", type=float, default=1, help="滑动持续时间（秒）")
    ap.add_argument("--swipe-length", type=float, default=0.08, help="滑动长度比例")
    ap.add_argument("--dry-run", action="store_true", help="试运行（不发送实际ADB指令）")
    ap.add_argument("--max-lock-dist", type=float, default=500.0, help="最大锁定距离（像素），超过此距离不锁定")
    return ap.parse_args()


# ==================== 目标跟踪器 ====================
class SimpleTracker:
    """基于类别+最近距离的目标实例匹配器"""

    def __init__(self, max_match_dist=100.0, max_missed=15):
        self.max_match_dist = max_match_dist
        self.max_missed = max_missed
        self.next_track_id = 1
        self.tracks = {}  # track_id -> dict
        self.lock = threading.Lock()

    def _new_track(self, det):
        track_id = self.next_track_id
        self.next_track_id += 1
        self.tracks[track_id] = {
            "track_id": track_id,
            "name": det["name"],
            "cx": det["cx"],
            "cy": det["cy"],
            "box": det["box"].copy(),
            "conf": det["conf"],
            "missed": 0,
            "history": [(int(det["cx"]), int(det["cy"]))],
            "vx": 0.0,
            "vy": 0.0,
            "last_time": time.time(),
        }
        return track_id

    def update(self, detections):
        with self.lock:
            now = time.time()

            # 标记所有已有轨迹先+1 missed
            for track in self.tracks.values():
                track["missed"] += 1

            unmatched_det_indices = set(range(len(detections)))
            track_ids = list(self.tracks.keys())

            # 贪心匹配
            candidate_pairs = []
            for det_idx, det in enumerate(detections):
                for track_id in track_ids:
                    tr = self.tracks[track_id]
                    if det["name"] != tr["name"]:
                        continue
                    dist = math.hypot(det["cx"] - tr["cx"], det["cy"] - tr["cy"])
                    if dist <= self.max_match_dist:
                        candidate_pairs.append((dist, det_idx, track_id))

            candidate_pairs.sort(key=lambda x: x[0])

            used_tracks = set()
            matched_dets = set()

            for dist, det_idx, track_id in candidate_pairs:
                if det_idx in matched_dets or track_id in used_tracks:
                    continue

                det = detections[det_idx]
                tr = self.tracks[track_id]

                dt = max(1e-3, now - tr["last_time"])
                vx = (det["cx"] - tr["cx"]) / dt
                vy = (det["cy"] - tr["cy"]) / dt

                tr["cx"] = det["cx"]
                tr["cy"] = det["cy"]
                tr["box"] = det["box"].copy()
                tr["conf"] = det["conf"]
                tr["missed"] = 0
                tr["vx"] = vx
                tr["vy"] = vy
                tr["last_time"] = now
                tr["history"].append((int(det["cx"]), int(det["cy"])))
                if len(tr["history"]) > 30:
                    tr["history"] = tr["history"][-30:]

                det["track_id"] = track_id
                matched_dets.add(det_idx)
                unmatched_det_indices.discard(det_idx)
                used_tracks.add(track_id)

            # 未匹配检测，新建轨迹
            for det_idx in unmatched_det_indices:
                det = detections[det_idx]
                track_id = self._new_track(det)
                det["track_id"] = track_id

            # 删除长时间丢失的轨迹
            dead_ids = [tid for tid, tr in self.tracks.items() if tr["missed"] > self.max_missed]
            for tid in dead_ids:
                del self.tracks[tid]

            return detections

    def get_track(self, track_id):
        with self.lock:
            return self.tracks.get(track_id)


# ==================== 中文显示支持 ====================
FONT_CACHE = None
FONT_LOCK = threading.Lock()

def get_chinese_font(size=20):
    """获取中文字体"""
    global FONT_CACHE
    with FONT_LOCK:
        if FONT_CACHE is not None:
            return FONT_CACHE
        
        # Windows系统字体路径
        font_paths = [
            "C:/Windows/Fonts/msyh.ttc",  # 微软雅黑
            "C:/Windows/Fonts/simhei.ttf",  # 黑体
            "C:/Windows/Fonts/simsun.ttc",  # 宋体
            "C:/Windows/Fonts/msyhbd.ttc",  # 微软雅黑粗体
            "/System/Library/Fonts/PingFang.ttc",  # macOS
            "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",  # Linux
        ]
        
        for path in font_paths:
            if os.path.exists(path):
                try:
                    FONT_CACHE = ImageFont.truetype(path, size)
                    print(f"[字体] 加载成功: {path}")
                    return FONT_CACHE
                except Exception as e:
                    continue
        
        print("[字体] 警告: 未找到中文字体，中文将显示为乱码")
        return None


def draw_chinese_text(img, text, pos, font_size=20, color=(255, 255, 0)):
    """在图像上绘制中文文本"""
    if not text:
        return img
    
    font = get_chinese_font(font_size)
    if font is None:
        # 如果没有中文字体，用OpenCV绘制（可能乱码）
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return img
    
    # 转换为PIL图像
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)
    
    # 绘制文本
    draw.text(pos, text, font=font, fill=color[::-1])  # PIL用RGB，OpenCV用BGR，所以反转颜色
    
    # 转回OpenCV图像
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def draw_crosshair(img, center_x, center_y, color=(0, 0, 255)):
    cv2.circle(img, (center_x, center_y), 15, color, 2)
    cv2.line(img, (center_x - 25, center_y), (center_x + 25, center_y), color, 2)
    cv2.line(img, (center_x, center_y - 25), (center_x, center_y + 25), color, 2)


def draw_trajectory(img, points, color=(0, 255, 255), thickness=2):
    if len(points) < 2:
        return
    for i in range(len(points) - 1):
        cv2.line(img, points[i], points[i + 1], color, thickness)


def draw_arrow(img, start, end, color, thickness=2, tip_length=20, tip_angle=30):
    """绘制箭头"""
    # 计算角度
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    
    # 绘制主线
    cv2.line(img, start, end, color, thickness)
    
    # 计算箭头两个分支
    tip1_x = end[0] - tip_length * math.cos(angle - math.radians(tip_angle))
    tip1_y = end[1] - tip_length * math.sin(angle - math.radians(tip_angle))
    tip2_x = end[0] - tip_length * math.cos(angle + math.radians(tip_angle))
    tip2_y = end[1] - tip_length * math.sin(angle + math.radians(tip_angle))
    
    # 绘制箭头
    cv2.line(img, end, (int(tip1_x), int(tip1_y)), color, thickness)
    cv2.line(img, end, (int(tip2_x), int(tip2_y)), color, thickness)


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def open_csv_writer(enabled, log_dir):
    if not enabled:
        return None, None

    ensure_dir(log_dir)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = Path(log_dir) / f"aim_test_async_{ts}.csv"
    f = open(csv_path, "w", newline="", encoding="utf-8-sig")
    writer = csv.writer(f)
    writer.writerow([
        "timestamp",
        "frame_id",
        "track_id",
        "class_name",
        "conf",
        "cx",
        "cy",
        "box_x1",
        "box_y1",
        "box_x2",
        "box_y2",
        "dx_to_center",
        "dy_to_center",
        "dist_to_center",
        "is_locked_target",
        "vx",
        "vy",
        "slide_x",
        "slide_y",
        "slide_dist",
        "screenshot_fps",
        "yolo_fps",
        "adb_fps",
    ])
    return f, writer


def log_detections(csv_writer, frame_data, locked_info, perf_stats):
    if csv_writer is None or frame_data is None:
        return

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    detections = frame_data['detections']
    locked_track_id = frame_data['locked_track_id']
    center_x = frame_data['center_x']
    center_y = frame_data['center_y']
    slide_info = frame_data.get('slide_info')
    frame_id = frame_data.get('frame_id', 0)

    for det in detections:
        x1, y1, x2, y2 = det["box"].astype(int)
        dx = det["cx"] - center_x
        dy = det["cy"] - center_y
        dist = math.hypot(dx, dy)

        vx = det.get("vx", 0.0)
        vy = det.get("vy", 0.0)

        slide_x = slide_info["slide_x"] if slide_info and det["track_id"] == locked_track_id else 0.0
        slide_y = slide_info["slide_y"] if slide_info and det["track_id"] == locked_track_id else 0.0
        slide_dist = slide_info["slide_dist"] if slide_info and det["track_id"] == locked_track_id else 0.0

        csv_writer.writerow([
            ts,
            frame_id,
            det["track_id"],
            det["name"],
            f"{det['conf']:.4f}",
            f"{det['cx']:.2f}",
            f"{det['cy']:.2f}",
            f"{x1:.2f}",
            f"{y1:.2f}",
            f"{x2:.2f}",
            f"{y2:.2f}",
            f"{dx:.2f}",
            f"{dy:.2f}",
            f"{dist:.2f}",
            1 if det["track_id"] == locked_track_id else 0,
            f"{vx:.2f}",
            f"{vy:.2f}",
            f"{slide_x:.2f}",
            f"{slide_y:.2f}",
            f"{slide_dist:.2f}",
            f"{perf_stats.get('screenshot_fps', 0):.1f}",
            f"{perf_stats.get('yolo_fps', 0):.1f}",
            f"{perf_stats.get('adb_fps', 0):.1f}",
        ])


def load_config(cfg_path):
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json5.load(f)
    return cfg


def select_locked_target(detections, locked_track_id, tracker, center_x, center_y, max_lock_dist):
    """
    锁定策略：
    1. 若当前已锁定目标仍存在，优先保持
    2. 否则按距离准星距离选择最近的（且距离小于max_lock_dist）
    """
    if locked_track_id is not None:
        same = [d for d in detections if d["track_id"] == locked_track_id]
        if same:
            return same[0]["track_id"], same[0]

    if not detections or center_x is None or center_y is None:
        return None, None

    # 为每个检测计算到准星的距离，并过滤掉太远的
    valid_detections = []
    for det in detections:
        dist = math.hypot(det["cx"] - center_x, det["cy"] - center_y)
        det["dist_to_center"] = dist
        if dist <= max_lock_dist:
            valid_detections.append(det)
    
    if not valid_detections:
        return None, None
    
    # 按距离排序（最近的优先）
    detections_sorted = sorted(valid_detections, key=lambda d: d["dist_to_center"])
    
    best = detections_sorted[0]
    return best["track_id"], best


def calculate_swipe(dx, dy, args):
    """
    计算滑动向量
    简化版：手指滑动方向 = 目标偏移方向
    要让准星对准目标，镜头需要向目标方向移动，所以手指也向同方向滑动
    """
    # 手指滑动方向 = 目标偏移方向
    slide_x = dx
    slide_y = dy
    
    # 计算滑动距离
    slide_dist = math.hypot(slide_x, slide_y)
    
    # 限制滑动量
    if slide_dist > args.max_step:
        scale = args.max_step / slide_dist
        slide_x *= scale
        slide_y *= scale
        slide_dist = args.max_step
    elif slide_dist < args.min_step and slide_dist > 0:
        scale = args.min_step / slide_dist
        slide_x *= scale
        slide_y *= scale
        slide_dist = args.min_step
    
    return slide_x, slide_y, slide_dist


# ==================== 主函数 ====================
def main():
    args = parse_args()

    cfg = load_config(args.cfg)
    classes = cfg["classes"]
    swipe_region = cfg.get("swipe_region", [0.30, 0.25, 0.40, 0.50])
    
    # 将滑动区域添加到args中方便传递
    args.swipe_region = swipe_region

    print("=" * 70)
    print("目标锁定测试脚本 - 异步优化版（带滑动箭头）")
    print("=" * 70)
    print(f"设备: {args.device}")
    print(f"模型: {args.model}")
    print(f"配置: {args.cfg}")
    print(f"imgsz: {args.imgsz}")
    print(f"conf: {args.conf}")
    print(f"瞄准容差: {args.aim_tol}px")
    print(f"最大步长: {args.max_step}px/帧")
    print(f"最小步长: {args.min_step}px")
    print(f"最大锁定距离: {args.max_lock_dist}px")
    print(f"滑动区域: {swipe_region}")
    print(f"试运行模式: {'是' if args.dry_run else '否'}")
    print("=" * 70)

    # 初始化ADB客户端
    adb = AdbClient(serial=args.serial)

    # 创建队列
    screenshot_queue = queue.Queue(maxsize=5)  # 截图到YOLO
    yolo_queue = queue.Queue(maxsize=3)        # YOLO到主线程
    control_queue = queue.Queue(maxsize=10)    # 主线程到ADB控制

    # 停止事件
    stop_event = threading.Event()

    # 性能监控
    perf_monitor = PerformanceMonitor(window_size=30)

    # 先获取一次屏幕分辨率
    test_img = adb.screencap()
    if test_img is None:
        print("[错误] 无法获取屏幕分辨率")
        return
    img_h, img_w = test_img.shape[:2]
    print(f"[屏幕分辨率] {img_w}x{img_h}")

    # 创建并启动截图线程
    screenshot_thread = ScreenshotThread(
        adb_client=adb,
        output_queue=screenshot_queue,
        stop_event=stop_event,
        perf_monitor=perf_monitor
    )
    screenshot_thread.start()

    # 创建并启动YOLO线程
    yolo_thread = YOLOThread(
        model_path=args.model,
        classes=classes,
        input_queue=screenshot_queue,
        output_queue=yolo_queue,
        stop_event=stop_event,
        perf_monitor=perf_monitor,
        device=args.device,
        imgsz=args.imgsz,
        conf=args.conf,
        max_det=args.max_det
    )
    yolo_thread.start()

    # 创建并启动ADB控制线程（传入屏幕分辨率）
    adb_thread = ADBControlThread(
        adb_client=adb,
        input_queue=control_queue,
        stop_event=stop_event,
        perf_monitor=perf_monitor,
        args=args,
        img_w=img_w,
        img_h=img_h
    )
    adb_thread.start()

    # 初始化跟踪器
    tracker = SimpleTracker(
        max_match_dist=args.max_match_dist,
        max_missed=max(args.max_lost, 15),
    )

    # 初始化CSV
    csv_file, csv_writer = open_csv_writer(args.save_csv, args.log_dir)
    if csv_file is not None:
        print(f"CSV 日志文件: {csv_file.name}")

    # 主循环变量
    frame_count = 0
    locked_track_id = None
    locked_target = None
    lock_loss_count = 0
    last_detections = []
    last_frame_data = None
    last_result = None  # 保存最近的检测结果用于显示
    
    # 滑动轨迹记录
    swipe_points = []
    last_swipe_cmd = None  # 最后一次滑动命令

    # 预加载字体
    get_chinese_font()

    # 主循环
    try:
        while not stop_event.is_set():
            loop_start = time.time()

            # 从YOLO队列获取检测结果（非阻塞）
            try:
                result = yolo_queue.get_nowait()
                if result:
                    frame_count = result.frame_id
                    last_detections = result.detections
                    last_frame_data = {
                        'frame_id': result.frame_id,
                        'detections': result.detections,
                        'img_w': result.img_w,
                        'img_h': result.img_h,
                        'center_x': result.center_x,
                        'center_y': result.center_y,
                    }
                    last_result = result  # 保存最新的结果用于显示
            except queue.Empty:
                pass

            # 如果有检测结果，进行跟踪和锁定
            slide_info = None
            if last_detections and last_frame_data:
                # 更新跟踪器
                last_detections = tracker.update(last_detections)
                
                center_x = last_frame_data['center_x']
                center_y = last_frame_data['center_y']
                
                # 锁定逻辑
                prev_locked = locked_track_id
                locked_track_id, locked_target = select_locked_target(
                    last_detections,
                    locked_track_id,
                    tracker,
                    center_x,
                    center_y,
                    args.max_lock_dist
                )
                
                if locked_track_id is None:
                    lock_loss_count += 1
                    if lock_loss_count > args.max_lost:
                        locked_target = None
                else:
                    lock_loss_count = 0
                    
                    if locked_track_id != prev_locked and locked_target:
                        dist = math.hypot(locked_target["cx"] - center_x, 
                                         locked_target["cy"] - center_y)
                        print(
                            f"\n[{datetime.now().strftime('%H:%M:%S')}] "
                            f"[锁定] {locked_target['name']} (ID:{locked_track_id}) 距离:{dist:.1f}px"
                        )
                
                # 计算控制命令
                if locked_target is not None:
                    dx = locked_target["cx"] - center_x
                    dy = locked_target["cy"] - center_y
                    dist = math.hypot(dx, dy)
                    
                    if dist > args.aim_tol:
                        slide_x, slide_y, slide_dist = calculate_swipe(dx, dy, args)
                        
                        # 创建控制命令
                        cmd = ControlCommand(
                            dx=dx,
                            dy=dy,
                            dist=dist,
                            slide_x=slide_x,
                            slide_y=slide_y,
                            slide_dist=slide_dist,
                            target_id=locked_track_id,
                            target_name=locked_target['name']
                        )
                        
                        # 发送到ADB线程（非阻塞）
                        try:
                            control_queue.put_nowait(cmd)
                        except queue.Full:
                            pass  # 队列满了就丢弃
                        
                        slide_info = {
                            "slide_x": slide_x,
                            "slide_y": slide_y,
                            "slide_dist": slide_dist
                        }
                        
                        if args.debug and slide_dist > 0:
                            print(f"\n  [控制] 偏差:({dx:.1f},{dy:.1f})px → 滑动:({slide_x:.1f},{slide_y:.1f})px")
            
            # 获取ADB线程的最后一次命令（用于可视化）
            last_swipe_cmd = adb_thread.get_last_command()
            
            # 获取性能统计
            perf_stats = perf_monitor.get_stats()
            
            # 记录CSV
            if last_frame_data and csv_writer:
                frame_data = last_frame_data.copy()
                frame_data['locked_track_id'] = locked_track_id
                frame_data['slide_info'] = slide_info
                frame_data['detections'] = last_detections
                log_detections(csv_writer, frame_data, locked_target, perf_stats)

            # 预览显示
            if args.show_preview and last_result is not None and last_result.image is not None:
                img_disp = last_result.image.copy()
                center_x, center_y = last_result.center_x, last_result.center_y

                # 绘制滑动区域
                sx, sy, sw, sh = swipe_region
                sx_px, sy_px = int(sx * last_result.img_w), int(sy * last_result.img_h)
                sw_px, sh_px = int(sw * last_result.img_w), int(sh * last_result.img_h)
                cv2.rectangle(img_disp, (sx_px, sy_px), (sx_px + sw_px, sy_px + sh_px), (255, 255, 255), 1)
                img_disp = draw_chinese_text(img_disp, "滑动区域", (sx_px, sy_px - 5), 15, (255, 255, 255))

                # 绘制所有目标框
                for det in last_detections:
                    x1, y1, x2, y2 = det["box"].astype(int)
                    is_locked = (det["track_id"] == locked_track_id)

                    if is_locked:
                        color = (0, 0, 255)  # 红色锁定目标
                        thickness = 3
                        # 添加外发光效果
                        cv2.rectangle(img_disp, (x1-2, y1-2), (x2+2, y2+2), (0, 0, 255), 1)
                    else:
                        color = (0, 255, 0)  # 绿色其他目标
                        thickness = 2

                    cv2.rectangle(img_disp, (x1, y1), (x2, y2), color, thickness)

                    # 添加标签
                    label = f"{det['name']} {det['conf']:.2f}"
                    if is_locked:
                        label = f"★ {label} ★"
                    
                    cv2.putText(
                        img_disp,
                        label,
                        (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        color,
                        2,
                    )

                    # 绘制轨迹
                    track = tracker.get_track(det["track_id"])
                    if track and len(track["history"]) >= 2:
                        draw_trajectory(img_disp, track["history"], color=(255, 255, 0), thickness=2)

                # 准星
                draw_crosshair(img_disp, center_x, center_y)

                # 绘制ADB滑动箭头（如果有）
                if last_swipe_cmd is not None:
                    # 箭头颜色：试运行模式用黄色，实际控制用紫色
                    arrow_color = (255, 255, 0) if args.dry_run else (255, 0, 255)
                    
                    # 绘制滑动起点到终点的箭头
                    start_point = (last_swipe_cmd.start_x_px, last_swipe_cmd.start_y_px)
                    end_point = (last_swipe_cmd.end_x_px, last_swipe_cmd.end_y_px)
                    
                    # 根据滑动距离决定箭头大小
                    tip_length = max(10, min(30, int(last_swipe_cmd.slide_dist / 2)))
                    
                    # 绘制箭头
                    draw_arrow(img_disp, start_point, end_point, arrow_color, thickness=3, tip_length=tip_length)
                    
                    # 在起点和终点添加标记
                    cv2.circle(img_disp, start_point, 5, arrow_color, -1)
                    cv2.circle(img_disp, end_point, 5, arrow_color, -1)
                    
                    # 添加文字说明
                    arrow_text = f"滑动: {last_swipe_cmd.slide_dist:.1f}px"
                    if args.dry_run:
                        arrow_text += " (试运行)"
                    img_disp = draw_chinese_text(
                        img_disp,
                        arrow_text,
                        (start_point[0] + 10, start_point[1] - 10),
                        15,
                        arrow_color
                    )

                # 锁定目标信息
                if locked_target is not None:
                    tx = int(locked_target["cx"])
                    ty = int(locked_target["cy"])
                    dx = locked_target["cx"] - center_x
                    dy = locked_target["cy"] - center_y
                    dist = math.hypot(dx, dy)

                    # 绘制连线（青色）
                    cv2.line(img_disp, (center_x, center_y), (tx, ty), (255, 255, 0), 2)
                    cv2.circle(img_disp, (tx, ty), 8, (255, 255, 0), -1)

                    track = tracker.get_track(locked_track_id)
                    vx = track["vx"] if track else 0.0
                    vy = track["vy"] if track else 0.0

                    # 信息面板
                    info_lines = [
                        f"锁定: {locked_target['name']} (ID:{locked_track_id})",
                        f"偏差: dx={dx:.1f} dy={dy:.1f} 距离:{dist:.1f}px",
                        f"速度: vx={vx:.1f} vy={vy:.1f}",
                    ]
                    
                    if slide_info:
                        info_lines.append(f"计算滑动: ({slide_info['slide_x']:.1f}, {slide_info['slide_y']:.1f})px")
                    
                    if last_swipe_cmd:
                        info_lines.append(f"实际滑动: ({last_swipe_cmd.slide_x:.1f}, {last_swipe_cmd.slide_y:.1f})px")
                    
                    if args.dry_run:
                        info_lines.append("模式: 试运行 (无控制)")
                    else:
                        info_lines.append("模式: 实际控制")

                    for i, text in enumerate(info_lines):
                        img_disp = draw_chinese_text(
                            img_disp,
                            text,
                            (10, 30 + i * 25),
                            18,
                            (255, 255, 0)
                        )

                # 性能信息
                yolo_fps = perf_stats.get('yolo_fps', 0)
                screenshot_fps = perf_stats.get('screenshot_fps', 0)
                adb_fps = perf_stats.get('adb_fps', 0)
                
                cv2.putText(
                    img_disp,
                    f"截图: {screenshot_fps:.1f} fps | YOLO: {yolo_fps:.1f} fps | ADB: {adb_fps:.1f} fps",
                    (10, last_result.img_h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2,
                )
                cv2.putText(
                    img_disp,
                    f"帧: {frame_count} | 检测: {len(last_detections)}",
                    (10, last_result.img_h - 45),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2,
                )

                # 缩放显示
                scale_percent = 50
                disp_w = int(img_disp.shape[1] * scale_percent / 100)
                disp_h = int(img_disp.shape[0] * scale_percent / 100)
                img_resized = cv2.resize(img_disp, (disp_w, disp_h), interpolation=cv2.INTER_AREA)

                cv2.imshow("YOLO Control Preview (Async)", img_resized)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("d"):
                    args.dry_run = not args.dry_run
                    print(f"\n[切换] 试运行模式: {'开启' if args.dry_run else '关闭'}")

            # 终端状态
            if locked_target is not None and last_frame_data:
                center_x = last_frame_data['center_x']
                center_y = last_frame_data['center_y']
                dx = locked_target["cx"] - center_x
                dy = locked_target["cy"] - center_y
                dist = math.hypot(dx, dy)
                
                status = "🔴" if not args.dry_run else "🟡"
                print(
                    f"\r{status} [帧:{frame_count:4d}] 锁定:{locked_target['name']:12s} "
                    f"ID:{locked_track_id:3d} 距离:{dist:6.1f}px 检测:{len(last_detections):2d} "
                    f"截图:{perf_stats.get('screenshot_fps', 0):.1f}fps YOLO:{perf_stats.get('yolo_fps', 0):.1f}fps",
                    end=""
                )
            else:
                print(
                    f"\r⚪ [帧:{frame_count:4d}] 未锁定 检测:{len(last_detections):2d} "
                    f"截图:{perf_stats.get('screenshot_fps', 0):.1f}fps YOLO:{perf_stats.get('yolo_fps', 0):.1f}fps",
                    end=""
                )

            # 主循环限流（保持高响应性）
            loop_time = time.time() - loop_start
            perf_monitor.add_time('total', loop_time)
            
            # 如果循环太快，稍微休眠避免CPU占用过高
            if loop_time < 0.005:
                time.sleep(0.005 - loop_time)

    except KeyboardInterrupt:
        print("\n\n" + "=" * 70)
        print("[退出] 正在停止线程...")
    finally:
        # 发送停止信号
        stop_event.set()
        
        # 等待线程结束
        screenshot_thread.join(timeout=2.0)
        yolo_thread.join(timeout=2.0)
        adb_thread.join(timeout=2.0)
        
        # 关闭CSV文件
        if csv_file is not None:
            csv_file.close()
        
        # 关闭预览窗口
        if args.show_preview:
            cv2.destroyAllWindows()
        
        # 显示最终统计
        print("\n" + "=" * 70)
        print("最终统计信息")
        print("=" * 70)
        perf_stats = perf_monitor.get_stats()
        for metric, value in perf_stats.items():
            if '_ms' in metric:
                print(f"{metric}: {value:.1f}ms")
            elif '_fps' in metric:
                print(f"{metric}: {value:.1f}fps")
        print("=" * 70)


if __name__ == "__main__":
    main()