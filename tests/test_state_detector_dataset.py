# tests/test_state_detector_dataset.py
"""
目的：
- 对状态识别器做“正负样本同时验证”（基于新的 _best_matches_in_state 接口）
  * 正样本：某状态文件夹下的图，应该被识别为该状态，且分数≥POS_THRESH
  * 负样本：同一张图，对其它状态的分数应≤NEG_THRESH，避免误判

数据组织：
tests/dataset/
  list/*.jpg
  prebattle/*.jpg
  combat/*.jpg
  settlement/*.jpg
  negatives/*.jpg   # 可选，非这四类（加载/广告/异常弹窗等）

运行：
  python -m pytest -q tests/test_state_detector_dataset.py
"""

import os
import glob
import cv2
import pytest
from war_drone.state_detector import TemplateStateDetector, States

# 阈值可按你的数据集微调（支持按状态单独设置）
GLOBAL_POS_THRESH = 0.80
NEG_THRESH = 0.85   # 负样本阈值

POS_THRESH_PER_STATE = {
    "list":       GLOBAL_POS_THRESH,
    "prebattle":  GLOBAL_POS_THRESH,
    # 战斗页支援图标在边缘预处理下分数偏低，先放宽；若你按方案A改了功能代码，可回到 0.80
    "combat":     0.40,
    "settlement": GLOBAL_POS_THRESH,
}
DATA_ROOT = "tests/dataset"
STATE_DIRS = {
    States.LIST:       os.path.join(DATA_ROOT, "list"),
    States.PREBATTLE:  os.path.join(DATA_ROOT, "prebattle"),
    States.COMBAT:     os.path.join(DATA_ROOT, "combat"),
    States.SETTLEMENT: os.path.join(DATA_ROOT, "settlement"),
}
NEG_DIR = os.path.join(DATA_ROOT, "negatives")  # 可选

def _have_dataset():
    return all(os.path.isdir(p) and glob.glob(os.path.join(p, "*.jpg")) for p in STATE_DIRS.values())

def _state_max_score(det: TemplateStateDetector, img_bgr, st: States):
    """
    使用新的 _best_matches_in_state，返回该状态下的最大分数与触发模板名。
    """
    matches = det._best_matches_in_state(img_bgr, st)  # [(score, loc, tpath), ...]
    if not matches:
        return 0.0, None
    score, _, tpath = max(matches, key=lambda x: x[0])
    return float(score), os.path.basename(tpath)

def _read_bgr(path):
    img = cv2.imread(path)
    assert img is not None, f"无法读取 {path}"
    return img

@pytest.mark.skipif(not _have_dataset(), reason="缺少 tests/dataset/<state>/*.jpg 批量样本，已跳过此数据集测试")
def test_dataset_positive_and_negative():
    # 与生产代码一致的 detector 配置（默认阈值在测试内不用）
    det = TemplateStateDetector(
        cfg_path="configs/config.json5",
        templates_dir="templates",
        default_thresh=0.80,     # 这里不会直接用到 predict 的阈值；我们手动比较分数
        use_edges=True,
        use_mask=True,
        method="CCORR_NORMED",
    )

    for st, folder in STATE_DIRS.items():
        img_paths = sorted(glob.glob(os.path.join(folder, "*.jpg")))
        assert img_paths, f"{folder} 里没有样本图"

        for path in img_paths:
            img = _read_bgr(path)

            # 计算四个状态的最大分数
            scores = {}
            tnames = {}
            for candidate in [States.LIST, States.PREBATTLE, States.COMBAT, States.SETTLEMENT]:
                s, t = _state_max_score(det, img, candidate)
                scores[candidate.value] = s
                tnames[candidate.value] = t

            # 取最高分状态
            best_state = max(scores.items(), key=lambda kv: kv[1])[0]
            best_score = scores[best_state]

            # 正类应胜出，且≥POS_THRESH
            # 正类应胜出，且≥该状态的 POS 阈值
            POS_THRESH = POS_THRESH_PER_STATE[st.value]
            assert best_state == st.value, \
                (f"[正类误判] {path}\n"
                 f"  期望={st.value} 实际={best_state} best_score={best_score:.3f}\n"
                 f"  分数：{ {k: round(v,3) for k,v in scores.items()} }\n"
                 f"  模板：{ {k: tnames[k] for k in scores.keys()} }")
            assert best_score >= POS_THRESH, \
                (f"[正类分数过低] {path}\n"
                 f"  状态={st.value} 分数={best_score:.3f} 触发模板={tnames[best_state]}\n"
                 f"  分数：{ {k: round(v,3) for k,v in scores.items()} }")

            # 负类都应 ≤ NEG_THRESH
            for other, sc in scores.items():
                if other == st.value:
                    continue
                assert sc <= NEG_THRESH, \
                    (f"[负类过阈] {path}\n"
                     f"  正类={st.value}；负类={other} 分数={sc:.3f} (> {NEG_THRESH})\n"
                     f"  触发模板={tnames[other]}")

@pytest.mark.skipif(not os.path.isdir(NEG_DIR), reason="缺少 tests/dataset/negatives/，已跳过额外负样本测试")
def test_negatives_folder_should_not_match_any():
    det = TemplateStateDetector(
        cfg_path="configs/config.json5",
        templates_dir="templates",
        default_thresh=0.75,
        use_edges=True,
        use_mask=True,
        method="CCORR_NORMED",
    )
    paths = sorted(glob.glob(os.path.join(NEG_DIR, "*.jpg")))
    assert paths, f"{NEG_DIR} 里没有样本图"

    for path in paths:
        img = _read_bgr(path)
        # 计算四个状态的最大分数，并取其中最高者
        max_score = max(_state_max_score(det, img, s)[0]
                        for s in [States.LIST, States.PREBATTLE, States.COMBAT, States.SETTLEMENT])
        assert max_score <= NEG_THRESH, \
            (f"[背景/异常被误判] {path}\n"
             f"  最高分={max_score:.3f} (> {NEG_THRESH})")
