"""
安全版目标跟踪测试脚本
功能：
1. YOLO 检测
2. 多目标实例匹配
3. 锁定目标显示（仅视觉层面，不控制设备）
4. OpenCV 预览
5. CSV 调试日志

使用方法：
python aim_test_safe.py --serial e5081c2a --model runs/detect/train4/weights/best.pt --show-preview

说明：
- 保留 ADB 截屏能力，但不发送任何滑动/点击控制命令
- 仅用于检测、跟踪、日志分析
"""

import argparse
import csv
import json5
import math
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from war_drone.adb_client import AdbClient


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
    ap.add_argument("--imgsz", type=int, default=1280, help="YOLO 推理尺寸，实时预览建议 960~1600")
    ap.add_argument("--conf", type=float, default=0.15, help="YOLO 置信度阈值")
    ap.add_argument("--max-det", type=int, default=20, help="单帧最大检测数")
    return ap.parse_args()


def draw_crosshair(img, center_x, center_y, color=(0, 0, 255)):
    cv2.circle(img, (center_x, center_y), 15, color, 2)
    cv2.line(img, (center_x - 25, center_y), (center_x + 25, center_y), color, 2)
    cv2.line(img, (center_x, center_y - 25), (center_x, center_y + 25), color, 2)


def draw_trajectory(img, points, color=(0, 255, 255), thickness=2):
    if len(points) < 2:
        return
    for i in range(len(points) - 1):
        cv2.line(img, points[i], points[i + 1], color, thickness)


class SimpleTracker:
    """
    简单的基于“类别 + 最近距离” 的目标实例匹配器
    不依赖外部跟踪库，适合先做工程验证
    """

    def __init__(self, max_match_dist=100.0, max_missed=15):
        self.max_match_dist = max_match_dist
        self.max_missed = max_missed
        self.next_track_id = 1
        self.tracks = {}  # track_id -> dict

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
        now = time.time()

        # 标记所有已有轨迹先+1 missed，后续匹配成功再归零
        for track in self.tracks.values():
            track["missed"] += 1

        unmatched_det_indices = set(range(len(detections)))
        track_ids = list(self.tracks.keys())

        # 贪心匹配：按距离从小到大尝试配对
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
        return self.tracks.get(track_id)


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def open_csv_writer(enabled, log_dir):
    if not enabled:
        return None, None

    ensure_dir(log_dir)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = Path(log_dir) / f"aim_test_safe_{ts}.csv"
    f = open(csv_path, "w", newline="", encoding="utf-8-sig")
    writer = csv.writer(f)
    writer.writerow([
        "timestamp",
        "frame",
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
    ])
    return f, writer


def log_detections(csv_writer, frame_count, detections, locked_track_id, center_x, center_y, tracker):
    if csv_writer is None:
        return

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        dx = det["cx"] - center_x
        dy = det["cy"] - center_y
        dist = math.hypot(dx, dy)

        track = tracker.get_track(det["track_id"])
        vx = track["vx"] if track else 0.0
        vy = track["vy"] if track else 0.0

        csv_writer.writerow([
            ts,
            frame_count,
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
        ])


def load_config(cfg_path):
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json5.load(f)
    return cfg


def validate_classes(cfg_classes, model_names):
    """
    尽量兼容：
    - cfg_classes 可能是 list
    - model.names 可能是 dict 或 list
    """
    if isinstance(model_names, dict):
        model_class_list = [model_names[k] for k in sorted(model_names.keys())]
    else:
        model_class_list = list(model_names)

    ok = (list(cfg_classes) == model_class_list)
    return ok, model_class_list


def select_locked_target(detections, locked_track_id, tracker, target_priority):
    """
    锁定策略：
    1. 若当前已锁定目标仍存在，优先保持
    2. 否则按优先级 + 置信度 选新的
    """
    if locked_track_id is not None:
        same = [d for d in detections if d["track_id"] == locked_track_id]
        if same:
            return same[0]["track_id"], same[0]

    def priority_of(name):
        try:
            return target_priority.index(name)
        except ValueError:
            return 999

    if not detections:
        return None, None

    detections_sorted = sorted(
        detections,
        key=lambda d: (priority_of(d["name"]), -d["conf"])
    )
    best = detections_sorted[0]
    return best["track_id"], best


def main():
    args = parse_args()

    cfg = load_config(args.cfg)
    classes = cfg["classes"]
    target_priority = cfg.get(
        "target_priority",
        ["AbramsTank", "CV90Tank", "BTR-82A", "Tigr-M", "Infantry"]
    )

    print("=" * 60)
    print("安全版目标跟踪测试脚本")
    print("=" * 60)
    print(f"设备: {args.device}")
    print(f"模型: {args.model}")
    print(f"配置: {args.cfg}")
    print(f"imgsz: {args.imgsz}")
    print(f"conf: {args.conf}")
    print(f"max_det: {args.max_det}")
    print(f"最大匹配距离: {args.max_match_dist}")
    print(f"最大丢失帧数: {args.max_lost}")
    print(f"CSV日志: {'开启' if args.save_csv else '关闭'}")
    print("=" * 60)

    model = YOLO(args.model)

    if hasattr(model, "names"):
        ok, model_class_list = validate_classes(classes, model.names)
        print(f"模型类别: {model_class_list}")
        if not ok:
            print("[警告] 配置文件 classes 与模型类别顺序/内容不一致！")
            print(f"配置类别: {classes}")
            print("建议先统一两边类别顺序，否则会发生类别错位。")

    adb = AdbClient(serial=args.serial)

    use_half = str(args.device).lower() != "cpu"

    tracker = SimpleTracker(
        max_match_dist=args.max_match_dist,
        max_missed=max(args.max_lost, 15),
    )

    csv_file, csv_writer = open_csv_writer(args.save_csv, args.log_dir)
    if csv_file is not None:
        print(f"CSV 日志文件: {csv_file.name}")

    frame_count = 0
    fail_count = 0
    start_time = time.time()

    locked_track_id = None
    locked_target = None
    lock_loss_count = 0

    try:
        while True:
            t0 = time.time()
            frame_count += 1

            # 1. 截屏
            bgr = adb.screencap()
            if bgr is None:
                fail_count += 1
                sleep_time = min(0.1 * fail_count, 1.0)
                print(f"[错误] 截屏失败，等待 {sleep_time:.1f}s")
                time.sleep(sleep_time)
                continue

            fail_count = 0

            img_h, img_w = bgr.shape[:2]
            center_x, center_y = img_w // 2, img_h // 2

            # 2. YOLO 检测
            res = model.predict(
                bgr,
                imgsz=args.imgsz,
                conf=args.conf,
                device=args.device,
                verbose=False,
                half=use_half,
                augment=False,
                max_det=args.max_det,
            )[0]

            # 3. 解析结果
            detections = []
            if res.boxes is not None and len(res.boxes) > 0:
                boxes = res.boxes.xyxy.detach().cpu().numpy()
                cls_list = res.boxes.cls.detach().cpu().numpy().astype(int)
                conf_list = res.boxes.conf.detach().cpu().numpy()

                for box, cls_idx, score in zip(boxes, cls_list, conf_list):
                    if cls_idx < 0 or cls_idx >= len(classes):
                        if args.debug:
                            print(f"[警告] 类别索引越界: {cls_idx}")
                        continue

                    x1, y1, x2, y2 = box
                    w = x2 - x1
                    h = y2 - y1
                    if w < 2 or h < 2:
                        continue

                    cx = (x1 + x2) / 2.0
                    cy = (y1 + y2) / 2.0
                    name = classes[cls_idx]

                    detections.append({
                        "name": name,
                        "conf": float(score),
                        "cx": float(cx),
                        "cy": float(cy),
                        "box": np.array([x1, y1, x2, y2], dtype=float),
                        "track_id": None,
                    })

            # 4. 实例匹配
            detections = tracker.update(detections)

            # 5. 锁定逻辑（仅视觉锁定）
            prev_locked_track_id = locked_track_id
            locked_track_id, locked_target = select_locked_target(
                detections,
                locked_track_id,
                tracker,
                target_priority
            )

            if locked_track_id is None:
                lock_loss_count += 1
                if lock_loss_count > args.max_lost:
                    locked_target = None
                    prev_locked_track_id = None
            else:
                lock_loss_count = 0

            if locked_track_id is not None and locked_track_id != prev_locked_track_id:
                print(
                    f"\n[{datetime.now().strftime('%H:%M:%S')}] "
                    f"[锁定] {locked_target['name']} (TrackID:{locked_track_id})"
                )

            # 6. 写 CSV
            log_detections(
                csv_writer,
                frame_count,
                detections,
                locked_track_id,
                center_x,
                center_y,
                tracker,
            )

            # 7. 预览
            if args.show_preview:
                img_disp = bgr.copy()

                # 所有目标框
                for det in detections:
                    x1, y1, x2, y2 = det["box"].astype(int)
                    is_locked = (det["track_id"] == locked_track_id)

                    color = (255, 0, 255) if is_locked else (0, 255, 0)
                    thickness = 3 if is_locked else 2

                    cv2.rectangle(img_disp, (x1, y1), (x2, y2), color, thickness)

                    label = f"{det['name']} {det['conf']:.2f} ID:{det['track_id']}"
                    cv2.putText(
                        img_disp,
                        label,
                        (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        color,
                        2,
                    )

                    track = tracker.get_track(det["track_id"])
                    if track and len(track["history"]) >= 2:
                        draw_trajectory(img_disp, track["history"], color=(255, 255, 0), thickness=2)

                # 准星
                draw_crosshair(img_disp, center_x, center_y)

                # 锁定目标信息
                if locked_target is not None:
                    tx = int(locked_target["cx"])
                    ty = int(locked_target["cy"])
                    dx = locked_target["cx"] - center_x
                    dy = locked_target["cy"] - center_y
                    dist = math.hypot(dx, dy)

                    cv2.line(img_disp, (center_x, center_y), (tx, ty), (255, 255, 0), 2)
                    cv2.circle(img_disp, (tx, ty), 8, (255, 255, 0), -1)

                    track = tracker.get_track(locked_track_id)
                    vx = track["vx"] if track else 0.0
                    vy = track["vy"] if track else 0.0

                    info_lines = [
                        f"锁定: {locked_target['name']} (ID:{locked_track_id})",
                        f"中心偏差: dx={dx:.1f}, dy={dy:.1f}",
                        f"距离: {dist:.1f}px",
                        f"速度: vx={vx:.1f}, vy={vy:.1f}",
                        "状态: 仅视觉锁定 / 无设备控制",
                    ]
                    for i, text in enumerate(info_lines):
                        cv2.putText(
                            img_disp,
                            text,
                            (10, 30 + i * 25),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (255, 255, 0),
                            2,
                        )

                elapsed_total = time.time() - start_time
                fps = frame_count / elapsed_total if elapsed_total > 0 else 0.0

                debug_lines = [
                    f"FPS: {fps:.1f}",
                    f"帧数: {frame_count}",
                    f"检测数: {len(detections)}",
                ]
                for i, text in enumerate(debug_lines):
                    cv2.putText(
                        img_disp,
                        text,
                        (img_w - 220, 30 + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (255, 255, 255),
                        2,
                    )

                # 缩放显示
                scale_percent = 50
                disp_w = int(img_disp.shape[1] * scale_percent / 100)
                disp_h = int(img_disp.shape[0] * scale_percent / 100)
                img_resized = cv2.resize(img_disp, (disp_w, disp_h), interpolation=cv2.INTER_AREA)

                cv2.imshow("YOLO Safe Tracking Preview", img_resized)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

            # 8. 终端简略状态
            if locked_target is not None:
                dx = locked_target["cx"] - center_x
                dy = locked_target["cy"] - center_y
                dist = math.hypot(dx, dy)
                print(
                    f"\r[帧:{frame_count:4d}] 锁定:{locked_target['name']:12s} "
                    f"ID:{locked_track_id:3d} 距离:{dist:6.1f}px 检测数:{len(detections):2d}",
                    end=""
                )
            else:
                print(
                    f"\r[帧:{frame_count:4d}] 未锁定 检测数:{len(detections):2d}",
                    end=""
                )

            # 9. 控制循环频率
            elapsed = time.time() - t0
            if elapsed < 0.03:
                time.sleep(0.03 - elapsed)

    except KeyboardInterrupt:
        print("\n\n" + "=" * 60)
        print("[退出] 统计信息")
        print("=" * 60)
        total_time = time.time() - start_time
        fps = frame_count / total_time if total_time > 0 else 0.0
        print(f"运行时间: {total_time:.1f} 秒")
        print(f"总帧数: {frame_count}")
        print(f"平均 FPS: {fps:.1f}")
        print("=" * 60)

    finally:
        if csv_file is not None:
            csv_file.close()
        if args.show_preview:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()