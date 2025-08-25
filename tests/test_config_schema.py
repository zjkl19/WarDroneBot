"""
目的：
- 校验 configs/config.json5 的结构与数值范围
- 特别检查：6 个 support 坐标是否存在且都在 [0,1] 区间内
"""
import json5

CFG = "configs/config.json5"

def test_config_schema_and_ranges():
    cfg = json5.load(open(CFG, "r", encoding="utf-8"))

    # 关键字段必须存在
    for k in ["package","screen","coords","timing","support_cd","energy","random"]:
        assert k in cfg, f"缺少关键字段：{k}"

    # 屏幕分辨率
    W = cfg["screen"]["width"]; H = cfg["screen"]["height"]
    assert isinstance(W,int) and isinstance(H,int) and W>0 and H>0

    # 基础坐标键
    coords = cfg["coords"]
    for k in ["list_start","pre_start","collect","menu"]:
        assert k in coords, f"coords 缺少 {k}"

    # 6 个支援坐标必须齐全、合法
    supports = sorted([k for k in coords if k.startswith("support")])
    assert len(supports) == 6, f"期望 6 个 support*，当前 {len(supports)}：{supports}"
    for k in supports:
        x,y = coords[k]
        assert 0<=x<=1 and 0<=y<=1, f"{k} 超出范围：{coords[k]}"

    # 冷却表也应包含 6 个（若缺则运行会走默认值，这里建议齐全）
    for k in supports:
        assert k in cfg["support_cd"], f"support_cd 缺少 {k}"
        assert float(cfg["support_cd"][k]) >= 0
