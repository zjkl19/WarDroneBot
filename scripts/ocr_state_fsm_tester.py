# scripts/ocr_state_fsm_tester.py
"""
手工测试 OCR 状态机：给定截屏，按配置判断状态并打印调试信息。

用法：
  python scripts/ocr_state_fsm_tester.py --image path/to/screenshot.png ^
         --cfg configs/ocr_states_fsm.json5

提示：
  - 需要安装 easyocr（pip install easyocr）
  - 配置中的 rois/关键词需根据你的截屏微调
"""
import argparse
import json
import os

import cv2

from war_drone.ocr_state_detector import OcrStateDetector


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="待检测的截屏路径（png/jpg）")
    ap.add_argument("--cfg", default="configs/ocr_states_fsm.json5", help="OCR 状态机配置路径")
    args = ap.parse_args()

    assert os.path.exists(args.image), f"找不到图像：{args.image}"
    assert os.path.exists(args.cfg), f"找不到配置：{args.cfg}"

    det = OcrStateDetector(cfg_path=args.cfg)
    img = cv2.imread(args.image)
    assert img is not None, f"无法读取图像：{args.image}"

    state, dbg = det.predict(img)
    print(f"PRED: {state}")
    print("SCORES:", {k: round(v, 2) for k, v in dbg["scores"].items()})

    # 打印命中的 OCR / 模板
    for name, detail in dbg["details"].items():
        hits = detail.get("ocr_hits", [])
        tmpl_hits = detail.get("tmpl_hits", [])
        if hits or tmpl_hits:
            print(f"\n[{name}]")
            for roi, txt, conf in hits:
                print(f"  OCR_HIT roi={roi} txt={txt} conf={conf:.2f}")
            for tmpl, sc in tmpl_hits:
                print(f"  TMPL_HIT {tmpl} score={sc:.2f}")

    # 可选：打印原始 OCR 文本以便调试
    print("\nRAW OCR (按 ROI):")
    for name, detail in dbg["details"].items():
        raw = detail.get("ocr_raw", {})
        for roi, rec in raw.items():
            print(f"  [{name}] roi={roi}")
            for txt, conf in rec.get("raw", []):
                print(f"    RAW : {txt} (conf={conf:.2f})")
            for txt, conf in rec.get("norm", []):
                print(f"    NORM: {txt} (conf={conf:.2f})")


if __name__ == "__main__":
    main()
