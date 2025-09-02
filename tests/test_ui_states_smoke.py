# tests/test_ui_states_smoke.py
"""
烟雾测试（最小化）：
- 按配置动态枚举状态（基础四状态 + extra_states）
- 对 tests/dataset/<state>/*.jpg 的每张图，计算各状态分数，取最高者
- 断言最高者 == 该图所属状态
- 若某状态目录无图片，自动跳过
"""
import os, glob, cv2, json5, pytest
from war_drone.state_detector import TemplateStateDetector

DATA_ROOT = "tests/dataset"

def _load_state_order():
    cfg = json5.load(open("configs/config.json5", "r", encoding="utf-8"))
    base = ["list", "prebattle", "combat", "settlement"]
    extra = [s["name"] for s in cfg.get("extra_states", [])]
    return base + extra

STATE_ORDER = _load_state_order()

def _iter_samples():
    found_any = False
    for st in STATE_ORDER:
        folder = os.path.join(DATA_ROOT, st)
        if not os.path.isdir(folder):
            continue
        paths = sorted([*glob.glob(os.path.join(folder, "*.jpg")),
                        *glob.glob(os.path.join(folder, "*.png"))])
        if not paths:
            continue
        found_any = True
        for p in paths:
            yield st, p
    if not found_any:
        pytest.skip("tests/dataset/* 没有任何样本图，跳过烟雾测试")

def _read_bgr(p):
    img = cv2.imread(p)
    assert img is not None, f"无法读取：{p}"
    return img

@pytest.mark.parametrize("gt_state,path", list(_iter_samples()))
def test_smoke_pick_best(detector: TemplateStateDetector, gt_state, path):
    img = _read_bgr(path)

    # 计算每个状态的最大匹配分数
    scores = {}
    for cand in STATE_ORDER:
        matches = detector._best_matches_in_state(img, cand)
        s = max([m[0] for m in matches], default=0.0)
        scores[cand] = s

    pred_state = max(scores.items(), key=lambda kv: kv[1])[0]

    assert pred_state == gt_state, (
        f"[烟雾失败] {path}\n"
        f"  期望={gt_state} 实际={pred_state}\n"
        f"  分数={ {k: round(v,3) for k,v in scores.items()} }"
    )
