import argparse, time
from .simple_bot import SimpleSupportBot

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--for-minutes", type=float, default=0.0, help="总运行时长（分钟，0=无限）")
    ap.add_argument("--once", action="store_true", help="只跑一局（不循环等体力）")
    ap.add_argument("--serial", type=str, default=None, help="ADB 设备序列号(可选)")
    args = ap.parse_args()

    bot = SimpleSupportBot(serial=args.serial)

    if args.once:
        bot.run_one_round()
        return

    end_ts = time.time() + args.for_minutes*60 if args.for_minutes>0 else float("inf")
    i = 0
    while time.time() < end_ts:
        i += 1
        print(f"\n===== ROUND #{i} =====")
        try:
            bot.run_one_round()
        except Exception as e:
            print("[WARN] Exception in round:", e)
        # 体力恢复等待（v0 简化为固定等待一格体力）
        bot.sleep_for_energy()

if __name__ == "__main__":
    main()
