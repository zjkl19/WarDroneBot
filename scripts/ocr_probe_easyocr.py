# scripts/ocr_probe_easyocr.py
import argparse, json5, cv2
import numpy as np
import easyocr

def crop_rel(img, rel):
    H, W = img.shape[:2]
    cx, cy, w, h = rel
    ww, hh = int(w*W), int(h*H)
    x1 = max(0, int(cx*W - ww/2)); y1 = max(0, int(cy*H - hh/2))
    x2 = min(W, x1+ww); y2 = min(H, y1+hh)
    return img[y1:y2, x1:x2]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/ocr_states.json5")
    ap.add_argument("--screen", required=True)
    ap.add_argument("--roi-key", required=True)       # e.g. goal_text / hp_bar / collect_btn
    ap.add_argument("--lang", default="ch_sim")       # ch_sim / en 等
    args = ap.parse_args()

    cfg = json5.load(open(args.cfg, "r", encoding="utf-8"))
    img = cv2.imread(args.screen); assert img is not None
    rel = cfg["rois"][args.roi_key]

    tile = crop_rel(img, rel)
    reader = easyocr.Reader([args.lang], gpu=False)
    res = reader.readtext(tile)

    print("[EasyOCR 结果]")
    for box, txt, conf in res:
        print(f"  {txt}  (conf={conf:.2f})")

if __name__ == "__main__":
    main()
