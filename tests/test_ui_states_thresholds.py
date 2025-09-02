# tests/test_ui_states_thresholds.py
"""
阈值测试：
- 正样本：所属状态分数应 ≥ POS_THR[state]
- 负样本（tests/dataset/negatives/*.jpg）：任意状态分数的最大值 ≤ NEG_THR
说明：
- 初始阈值给了一个保守默认；你可按 score_dataset.py 报告微调
"""
import os, glob, cv2, json5, pytest
from war_drone.state_detector import TemplateStateDetector

DATA_ROOT = "tests/dataset"

# 全局/逐状态阈值（可按 runs/report/scores.csv 调整）
GLOBAL_POS_THR = 0.80
NEG_THR = 0.85
POS_THR = {
    "list":       GLOBAL_POS_THR,
    "prebattle":  0.85,            # 战前一般更高
    "combat":     0.70,            # 若你已在检测器里对 combat 关闭边缘/放大 ROI，可升回 0.8
    "settlement": GLOBAL_POS_THR,
    # 扩展状态（若有）可在此加键覆盖
}

def _load_state_order():
    cfg = json5.load(open("configs/config.json5", "r", encoding="utf-8"))
    base = ["list", "prebattle", "combat", "settlement"]
    extra = [s["name"] for s in cfg.get("extra_states", [])]
    return base + extra

STATE_ORDER = _load_state_order()

def _iter_pos_samples():
    any_found = False
    for st in STATE_ORDER:
        folder = os.path.join(DATA_ROOT, st)
        if not os.path.isdir(folder): continue
        paths = sorted([*glob.glob(os.path.join(folder, "*.jpg")),
                        *glob.glob(os.path.join(folder, "*.png"))])
        for p in paths:
            any_found = True
            yield st, p
    if not any_found:
        pytest.skip("没有任何正样本，跳过阈值测试")

def _iter_neg_samples():
    folder = os.path.join(DATA_ROOT, "negatives")
    if not os.path.isdir(folder):
        pytest.skip("没有 negatives/ 负样本，跳过负样本测试")
    paths = sorted([*glob.glob(os.path.join(folder, "*.jpg")),
                    *glob.glob(os.path.join(folder, "*.png"))])
    if not paths:
        pytest.skip("negatives/ 为空，跳过负样本测试")
    for p in paths:
        yield p

def _read_bgr(p):
    img = cv2.imread(p); assert img is not None, f"无法读取：{p}"; return img

@pytest.fixture(scope="module")
def detector():
    return TemplateStateDetector(
        cfg_path="configs/config.json5",
        templates_dir="templates",
        use_edges=True,
        use_mask=True,
        method="CCORR_NORMED",
        default_thresh=0.85,
    )

@pytest.mark.parametrize("st,path", list(_iter_pos_samples()))
def test_positive_threshold(detector: TemplateStateDetector, st, path):
    img = _read_bgr(path)
    scores = {}
    for cand in STATE_ORDER:
        matches = detector._best_matches_in_state(img, cand)
        scores[cand] = max([m[0] for m in matches], default=0.0)
    pos_thr = POS_THR.get(st, GLOBAL_POS_THR)
    assert scores[st] >= pos_thr, (
        f"[正类分数过低] {path}\n"
        f"  state={st} score={scores[st]:.3f} < thr={pos_thr}\n"
        f"  全部分数={ {k: round(v,3) for k,v in scores.items()} }"
    )

@pytest.mark.parametrize("path", list(_iter_neg_samples()))
def test_negative_threshold(detector: TemplateStateDetector, path):
    img = _read_bgr(path)
    best = 0.0
    for cand in STATE_ORDER:
        matches = detector._best_matches_in_state(img, cand)
        s = max([m[0] for m in matches], default=0.0)
        if s > best: best = s
    assert best <= NEG_THR, f"[负样本过阈] {path} best={best:.3f} > NEG_THR={NEG_THR}"
