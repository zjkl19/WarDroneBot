# war_drone/runner.py
import argparse
import time
from war_drone.simple_bot import SimpleSupportBot

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--serial", default=None, help="adb 设备序列号（可选）")
    ap.add_argument("--once", action="store_true", help="只执行一局（一次循环）")
    ap.add_argument("--minutes", type=int, default=None, help="运行多少分钟（与 --once 互斥）")
    ap.add_argument("--no-edges", action="store_true", help="禁用边缘预处理")
    ap.add_argument("--no-mask", action="store_true", help="禁用模板掩码匹配")
    ap.add_argument("--debug", action="store_true", help="更详细日志输出")
    return ap.parse_args()

def main():
    args = parse_args()
    bot = SimpleSupportBot(
        serial=args.serial,
        use_edges=not args.no_edges,
        use_mask=not args.no_mask,
        debug=args.debug,
    )

    if args.once:
        bot.run_one_cycle()
        return

    if args.minutes is None:
        print("[INFO] 未指定 --minutes，默认跑 30 分钟")
        end_ts = time.time() + 30*60
    else:
        end_ts = time.time() + args.minutes*60

    i = 0
    while time.time() < end_ts:
        i += 1
        print(f"[INFO] 开始第 {i} 次循环")
        bot.run_one_cycle()
        # 体力机制：每次 1 点体力，15 分钟恢复 1 点
        # 如果你希望等待体力，可在此处 sleep；第一版我们不自动等待，交由你手动控制 --minutes 或 --once。
        time.sleep(3)

if __name__ == "__main__":
    main()
