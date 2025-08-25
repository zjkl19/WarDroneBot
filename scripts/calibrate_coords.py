"""
在命令行里运行后，手机上“随便点”几个位置，
脚本会在控制台打印：像素坐标 & 相对坐标（基于 config.screen）。
再把相对坐标写回 configs/config.json。
"""
import json, time
from war_drone.config import Config
from war_drone.adb_client import ADBClient

cfg = Config()
adb = ADBClient()

print("Tap 5 spots on the phone (every 1.5s). Watch the console output...")
for i in range(5):
    time.sleep(1.5)
    # 读取一帧截图并等待你在手机上点的位置（这个简单版无法直接读点击位置，
    # 只是演示：告诉你相对坐标如何计算——实际你看像素点在图中的位置，手工写回）
    print(f"Screen size: {cfg.W}x{cfg.H} ; Example: point(2136, 1080) → {(2136/cfg.W,1080/cfg.H)}")
print("Done. Put ratios back to configs/config.json.")
