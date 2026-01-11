# -*- coding: utf-8 -*-
"""
单张/多张图片检测脚本，避免 CLI 兼容性问题。
用法示例：
  python -m scripts.yolo_predict_once ^
    --model runs/detect/train4/weights/best.pt ^
    --source combat.jpg ^
    --imgsz 960 ^
    --conf 0.1 ^
    --device cpu ^
    --save-dir runs/yolo_pred
"""
import argparse
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="权重文件，如 runs/detect/train4/weights/best.pt")
    ap.add_argument("--source", required=True, help="图片或目录，支持通配符")
    ap.add_argument("--imgsz", type=int, default=960, help="推理分辨率")
    ap.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    ap.add_argument("--device", default="cpu", help="cpu 或 0/0,1 等 GPU")
    ap.add_argument("--save-dir", default="runs/yolo_pred", help="输出目录")
    args = ap.parse_args()

    model = YOLO(args.model)
    model.predict(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        save=True,
        project=args.save_dir,
        name=".",
        exist_ok=True,
        verbose=True,
    )


if __name__ == "__main__":
    main()
