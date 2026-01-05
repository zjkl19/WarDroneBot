# scripts/ocr_debug_scores.py
import sys, os, json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
from war_drone.ocr_state_detector import OcrStateDetector

def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/ocr_debug_scores.py <image_path> [cfg_path]")
        return
    img_path = sys.argv[1]
    cfg_path = (sys.argv[2] if len(sys.argv) >= 3 else "configs/ocr_states.json5")

    print(f"[INFO] img={img_path}")
    print(f"[INFO] cfg={cfg_path}")

    det = OcrStateDetector(cfg_path)
    img = cv2.imread(img_path)
    assert img is not None, f"cannot read {img_path}"

    # 打印当前 states 列表与 rois 里关键键是否存在
    state_names = [s["name"] for s in det.states]
    print("[INFO] states:", state_names)
    print("[INFO] has ROI 'pre_start_btn':", "pre_start_btn" in det.rois)
    if "pre_start_btn" in det.rois:
        print("[INFO] ROI pre_start_btn:", det.rois["pre_start_btn"])
    print("[INFO] has ROI 'list_start':", "list_start" in det.rois)
    if "list_start" in det.rois:
        print("[INFO] ROI list_start:", det.rois["list_start"])

    # 直接走一次预测，拿到 full debug
    pred, dbg = det.predict(img)
    print("PRED:", pred)
    print("SCORES:", {k: round(v, 2) for k, v in dbg["scores"].items()})

    # 重点打印 prebattle 与 list 的原始 OCR 结果（无论命不中也打印）
    for key in ("prebattle", "list"):
        if key in dbg["details"]:
            print(f"\n[{key}]")
            d = dbg["details"][key]
            # 原始 OCR 文本（raw/norm）
            if "ocr_raw" in d:
                for roi, rec in d["ocr_raw"].items():
                    print(f"  (ocr_raw) roi={roi}")
                    for txt, conf in rec.get("raw", []):
                        print(f"    RAW : {txt} (conf={conf:.2f})")
                    for txt, conf in rec.get("norm", []):
                        print(f"    NORM: {txt} (conf={conf:.2f})")
            # 命中的 contains/regex
            for r in d.get("ocr_hits", []):
                print("  OCR_HIT:", r)
            # 模板命中
            for r in d.get("tmpl_hits", []):
                print("  TMPL_HIT:", r)

if __name__ == "__main__":
    main()
