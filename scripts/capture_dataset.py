# scripts/capture_dataset.py
import os, time, argparse, subprocess
import numpy as np
import cv2

def adb_exec(args, serial=None, out=False):
    base = ["adb"]
    if serial: base += ["-s", serial]
    base += args
    if out:
        return subprocess.check_output(base)
    subprocess.check_call(base)

def screencap_bgr(serial=None):
    data = adb_exec(["exec-out","screencap","-p"], serial=serial, out=True)
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=True, help="任意状态名，例如 list / prebattle / combat / settlement / negatives / splash / upgrade ...")
    ap.add_argument("--serial", default=None)
    ap.add_argument("--count", type=int, default=5)
    args = ap.parse_args()

    out_dir = os.path.join("tests","dataset", args.state)
    os.makedirs(out_dir, exist_ok=True)
    for i in range(args.count):
        img = screencap_bgr(args.serial)
        ts = time.strftime("%Y%m%d_%H%M%S")
        path = os.path.join(out_dir, f"{args.state}_{ts}_{i:02d}.jpg")
        cv2.imwrite(path, img)
        print("saved:", path)
        time.sleep(0.8)

if __name__ == "__main__":
    main()
