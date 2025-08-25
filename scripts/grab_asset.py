# scripts/grab_asset.py
import argparse, os
from war_drone.adb_client import ADBClient

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="保存文件名（不含扩展名），例如 list_screen")
    ap.add_argument("--outdir", default="tests/assets", help="输出目录")
    ap.add_argument("--serial", default=None)
    args = ap.parse_args()

    adb = ADBClient(args.serial)
    os.makedirs(args.outdir, exist_ok=True)
    img = adb.screencap()
    path = os.path.join(args.outdir, f"{args.name}.jpg")
    with open(path, "wb") as f:
        f.write(img)
    print("Saved:", path)

if __name__ == "__main__":
    main()
