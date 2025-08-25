import argparse, time, os
from .simple_bot import SimpleSupportBot
from .logger import Logger, now_str

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/config.json5", help="配置文件路径（JSON5）")
    ap.add_argument("--for-minutes", type=float, default=0.0, help="总运行时长（分钟，0=无限）")
    ap.add_argument("--once", action="store_true", help="只跑一局（不循环等体力）")
    ap.add_argument("--serial", type=str, default=None, help="ADB 设备序列号(可选)")
    ap.add_argument("--session-name", type=str, default=None, help="会话目录名（默认用时间戳）")
    args = ap.parse_args()

    session = args.session_name or now_str()
    session_dir = os.path.join("runs", session)
    log = Logger(session_dir=session_dir)

    log.info("SESSION DIR:", session_dir)
    log.info("CONFIG:", args.config)

    bot = SimpleSupportBot(cfg_path=args.config, serial=args.serial, logger=log)

    if args.once:
        log.info("MODE: once")
        bot.run_one_round()
        log.info("DONE one round")
        return

    log.info("MODE: loop", f"for {args.for_minutes} minutes (0 = infinite)")
    end_ts = time.time() + args.for_minutes*60 if args.for_minutes>0 else float("inf")
    i = 0
    while time.time() < end_ts:
        i += 1
        log.info(f"=== LOOP #{i} ===")
        try:
            bot.run_one_round()
        except Exception as e:
            log.err("Loop exception:", e)
            time.sleep(3)
        bot.sleep_for_energy()

if __name__ == "__main__":
    main()
