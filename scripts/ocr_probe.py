# -*- coding: utf-8 -*-
import argparse, json5, cv2, numpy as np
from paddleocr import PaddleOCR

def crop_rel(img, rel, wh):
    cx,cy,w,h = rel
    W,H = wh
    ww,hh = int(w*W), int(h*H)
    x1 = max(0, int(cx*W - ww/2))
    y1 = max(0, int(cy*H - hh/2))
    x2 = min(W, x1+ww)
    y2 = min(H, y1+hh)
    return img[y1:y2, x1:x2]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/ocr_states.json5")
    ap.add_argument("--screen", required=True)
    ap.add_argument("--roi-key", required=True, help="如 goal_text / hp_bar / collect_btn 等")
    ap.add_argument("--lang", default="ch")  # ch / en
    args = ap.parse_args()

    cfg = json5.load(open(args.cfg,"r",encoding="utf-8"))
    rois = cfg["rois"]; assert args.roi_key in rois, f"缺少 ROI {args.roi_key}"
    img = cv2.imread(args.screen); assert img is not None

    tile = crop_rel(img, rois[args.roi_key], (cfg["screen"]["width"], cfg["screen"]["height"]))

    ocr = PaddleOCR(use_angle_cls=True, lang=("ch" if args.lang=="ch" else "en"))
    res = ocr.ocr(tile, cls=True)

    texts = []
    for line in res:
        for box, (txt, score) in line:
            texts.append((txt, float(score)))
    print("[OCR 结果]")
    for t,sc in texts:
        print(f"  {t}  (conf={sc:.2f})")

if __name__ == "__main__":
    main()
