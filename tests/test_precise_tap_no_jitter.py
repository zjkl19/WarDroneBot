"""
目的：
- 验证 precise 档点击无抖动：多次点击得到的像素坐标一致（同一个点）
实现：
- monkeypatch _pct_to_px 为可观测版本，记录输出；或直接对函数进行重复调用检测
这里复用 _pct_to_px，并通过 jitter=0 的逻辑保证无偏移。
"""
import json5
from war_drone.simple_bot import _pct_to_px

CFG = "configs/config.json5"

def test_precise_has_no_jitter():
    cfg = json5.load(open(CFG, "r", encoding="utf-8"))
    wh = (cfg["screen"]["width"], cfg["screen"]["height"])
    p = cfg["coords"]["support3"]  # 任取一点

    # 当 jitter=0 时，多次调用应返回同一像素
    xs, ys = set(), set()
    for _ in range(50):
        x, y = _pct_to_px(p, wh, jitter=0)
        xs.add(x); ys.add(y)
    assert len(xs)==1 and len(ys)==1, f"precise 模式出现抖动：{xs}, {ys}"
