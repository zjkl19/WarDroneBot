# scripts/check_dataset.py
"""
检查 tests/dataset/ 下各状态样本覆盖情况：
- 按 configs/config.json5 中配置的状态列表（基础4状态 + extra_states）
- 输出每个状态的样本数，标记不足/缺失
- 可选：检查 negatives/ 文件夹
"""

import os, glob, json5

DATA_ROOT = "tests/dataset"
CFG_PATH = "configs/config.json5"

def load_state_order():
    cfg = json5.load(open(CFG_PATH, "r", encoding="utf-8"))
    base = ["list", "prebattle", "combat", "settlement"]
    extra = [s["name"] for s in cfg.get("extra_states", [])]
    return base + extra

def main():
    state_order = load_state_order()

    print("=== 数据集状态覆盖检查 ===")
    for st in state_order:
        folder = os.path.join(DATA_ROOT, st)
        jpgs = glob.glob(os.path.join(folder, "*.jpg"))
        pngs = glob.glob(os.path.join(folder, "*.png"))
        count = len(jpgs) + len(pngs)

        if count == 0:
            print(f"[缺失] {st:<15s} → 0 张")
        elif count < 5:
            print(f"[不足] {st:<15s} → {count} 张（建议 ≥5）")
        else:
            print(f"[OK]   {st:<15s} → {count} 张")

    # negatives 单独列
    neg_dir = os.path.join(DATA_ROOT, "negatives")
    if os.path.isdir(neg_dir):
        negs = glob.glob(os.path.join(neg_dir, "*.jpg")) + glob.glob(os.path.join(neg_dir, "*.png"))
        if negs:
            print(f"[OK]   negatives        → {len(negs)} 张")
        else:
            print("[空]   negatives 文件夹存在但无图片")
    else:
        print("[缺失] negatives 文件夹不存在")

    print("\n提示：每个状态至少准备 5~10 张，负样本也建议 ≥10 张。")

if __name__ == "__main__":
    main()
