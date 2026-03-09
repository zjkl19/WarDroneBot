#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Paddle Runner 单文件启动脚本
直接运行此文件即可启动带有指定参数的paddle_runner
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    # 获取当前脚本所在目录
    current_dir = Path(__file__).parent.absolute()
    
    # 构建命令
    cmd = [
        sys.executable,  # 使用当前Python解释器
        "-m",
        "scripts.paddle_runner",
        "--serial", "e5081c2a",
        "--cfg", str(current_dir / "configs/ocr_states_fsm.json5"),
        "--det-dir", r"E:\.paddleocr\whl\det\ch_PP-OCRv4_det_infer",
        "--rec-dir", r"E:\.paddleocr\whl\rec\ch_PP-OCRv4_rec_infer",
        "--cls-dir", r"E:\.paddleocr\whl\ch_ppocr_mobile_v2.0_cls_infer",
        "--combat-macro", str(current_dir / "recordings/mission12_01.json"),
        "--combat-macro-loops", "1",
        "--max-combat", "1",
        "--prestart-macro",
        "--prestart-delay", "0.1",
        "--interval", "0.5"
    ]
    
    # 打印要执行的命令
    print("正在执行命令:")
    print(" ".join(cmd))
    print("-" * 50)
    
    try:
        # 执行命令
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"命令执行失败，错误代码: {e.returncode}")
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print("\n用户中断执行")
        sys.exit(0)
    except Exception as e:
        print(f"执行过程中出现错误: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()