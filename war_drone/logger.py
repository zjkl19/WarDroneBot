import os, time, io

def now_str(): return time.strftime("%Y-%m-%d_%H-%M-%S")

class Logger:
    def __init__(self, session_dir="runs"):
        # 创建会话目录
        if session_dir is None:
            session_dir = os.path.join("runs", now_str())
        if not os.path.isdir(session_dir):
            os.makedirs(session_dir, exist_ok=True)
        self.session_dir = session_dir
        self.img_dir = os.path.join(session_dir, "images")
        os.makedirs(self.img_dir, exist_ok=True)
        self.log_path = os.path.join(session_dir, "logs")
        os.makedirs(self.log_path, exist_ok=True)
        self.log_file = os.path.join(self.log_path, "run.log")

        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(f"=== SESSION START {now_str()} ===\n")

    def _write(self, lvl, *a):
        msg = " ".join(str(x) for x in a)
        line = f"[{lvl}] {time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def info(self, *a): self._write("INFO", *a)
    def warn(self, *a): self._write("WARN", *a)
    def err(self,  *a): self._write("ERR ", *a)

    def save_png_bytes(self, img_bytes: bytes, filename: str) -> str:
        path = os.path.join(self.img_dir, filename)
        with open(path, "wb") as f:
            f.write(img_bytes)
        return path
