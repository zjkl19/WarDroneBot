# war_drone/logger.py
import os
import cv2
from datetime import datetime

class RunLogger:
    def __init__(self, run_dir: str, verbose=False):
        self.run_dir = run_dir
        self.verbose = verbose
        self.log_path = os.path.join(run_dir, "run.log")
        os.makedirs(run_dir, exist_ok=True)

    def _stamp(self):
        return datetime.now().strftime("%H%M%S_%f")[:-3]

    def info(self, msg: str):
        self._write("INFO", msg)

    def warn(self, msg: str):
        self._write("WARN", msg)

    def section(self, name: str):
        self._write("----", f"===== {name} =====")

    def _write(self, level, msg):
        line = f"{self._stamp()} [{level}] {msg}"
        if self.verbose or level != "INFO":
            print(line)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def save_image(self, bgr, suffix="cap"):
        name = f"{self._stamp()}_{suffix}.png"
        path = os.path.join(self.run_dir, name)
        cv2.imwrite(path, bgr)
        return path

    def save_overlay(self, bgr, suffix="overlay"):
        name = f"{self._stamp()}_{suffix}.png"
        path = os.path.join(self.run_dir, name)
        cv2.imwrite(path, bgr)
        return path
