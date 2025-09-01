# scripts/score_dataset.py
import os, glob, csv
import cv2, json5
from collections import defaultdict
from war_drone.state_detector import TemplateStateDetector

DATA_ROOT = "tests/dataset"
OUT_DIR   = "runs/report"
os.makedirs(OUT_DIR, exist_ok=True)

def read_bgr(p):
    img = cv2.imread(p); assert img is not None, f"read fail: {p}"; return img

def main():
    # 读取配置，拼出状态列表
    cfg = json5.load(open("configs/config.json5", "r", encoding="utf-8"))
    base_states = ["list", "prebattle", "combat", "settlement"]
    extra_states = [s["name"] for s in cfg.get("extra_states", [])]
    state_order = base_states + extra_states

    det = TemplateStateDetector(cfg_path="configs/config.json5",
                                templates_dir="templates",
                                use_edges=True, use_mask=True,
                                method="CCORR_NORMED",
                                default_thresh=0.85)

    rows = []
    per_state_scores = defaultdict(list)  # 正样本分布
    neg_best_scores = []

    # 正样本
    for st_name in state_order:
        folder = os.path.join(DATA_ROOT, st_name)
        if not os.path.isdir(folder):
            continue
        for p in sorted(glob.glob(os.path.join(folder, "*.jpg"))):
            img = read_bgr(p)
            # 计算所有状态分数
            scores = {}
            for cand in state_order:
                matches = det._best_matches_in_state(img, cand)
                s = max([m[0] for m in matches], default=0.0)
                scores[cand] = s
            best_state = max(scores.items(), key=lambda kv: kv[1])[0]
            best_score = scores[best_state]

            row = [p, st_name] + [scores.get(n, "") for n in state_order] + [best_state, best_score]
            rows.append(row)

            per_state_scores[st_name].append(scores.get(st_name, 0.0))

            # 误判另存
            if best_state != st_name:
                vis = img.copy()
                cv2.putText(vis, f"GT={st_name} PRED={best_state} score={best_score:.3f}",
                            (20,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
                outp = os.path.join(OUT_DIR, f"mis_{os.path.basename(p)}")
                cv2.imwrite(outp, vis)

    # 负样本（可选）
    neg_dir = os.path.join(DATA_ROOT, "negatives")
    if os.path.isdir(neg_dir):
        for p in sorted(glob.glob(os.path.join(neg_dir, "*.jpg"))):
            img = read_bgr(p)
            best = 0.0
            for cand in state_order:
                matches = det._best_matches_in_state(img, cand)
                s = max([m[0] for m in matches], default=0.0)
                if s > best: best = s
            neg_best_scores.append(best)
            rows.append([p, "negative"] + ["-"]*len(state_order) + ["best_any", best])

    # 写 CSV
    csv_path = os.path.join(OUT_DIR, "scores.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        header = ["path","gt"] + [f"score_{n}" for n in state_order] + ["pred","pred_score"]
        w.writerow(header)
        w.writerows(rows)

    # 简单建议阈值
    def suggest_threshold(pos_scores, neg_scores, margin=0.05):
        pos_scores = sorted(pos_scores)
        pos_q = pos_scores[max(0, int(0.15*len(pos_scores))-1)] if pos_scores else 0.8  # 15%分位
        neg_max = max(neg_scores) if neg_scores else 0.4
        return max(0.5, min(0.98, max(pos_q - margin, neg_max + margin)))

    print("\n=== 建议阈值（参考） ===")
    for st_name in state_order:
        pos_scores = per_state_scores[st_name]
        thr = suggest_threshold(pos_scores, neg_best_scores, margin=0.05)
        if pos_scores:
            print(f"{st_name:16s} pos_avg={sum(pos_scores)/len(pos_scores):.3f}  pos_min={min(pos_scores):.3f}  "
                  f"neg_max={max(neg_best_scores) if neg_best_scores else 0:.3f}  suggest_thr={thr:.3f}")
        else:
            print(f"{st_name:16s} (无正样本)")

    print(f"\n[OK] 写入报告: {csv_path}")
    print(f"[OK] 误判可视化（如有）在: {OUT_DIR}")

if __name__ == "__main__":
    main()
