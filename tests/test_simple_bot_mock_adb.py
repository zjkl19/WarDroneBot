"""
目的：
- 在“无真机”的情况下，验证简单流程逻辑不会抛异常
- 通过 MockADB 记录 tap 调用（名称与次数）
- 只检查代码结构与时序是否可跑，不涉及图像识别

使用：
pytest -q tests/test_simple_bot_mock_adb.py
"""
import json5
import types
import time

from war_drone.simple_bot import SimpleSupportBot

class MockADB:
    """
    伪造的 ADB 客户端：
    - 记录 tap/launch/key 调用
    - 不做任何真实设备操作
    """
    def __init__(self):
        self.log = []

    def tap(self, x, y):
        self.log.append(("tap", x, y))

    def swipe(self, *a, **k):
        self.log.append(("swipe", a))

    def key(self, code):
        self.log.append(("key", code))

    def launch(self, pkg):
        self.log.append(("launch", pkg))

    def kill(self, pkg):
        self.log.append(("kill", pkg))

def mk_fast_bot():
    bot = SimpleSupportBot()
    # 注入 MockADB
    bot.adb = MockADB()
    # 加速：缩短所有等待
    bot.t["launch_wait_s"]       = 0.1
    bot.t["prebattle_wait_s"]    = 0.1
    bot.t["combat_seconds"]      = 2.0
    bot.t["settle_wait_s"]       = 0.1
    bot.t["post_collect_wait_s"] = 0.1
    # 支援冷却加速
    for k in bot.cd.keys():
        bot.cd[k] = 0.3
    return bot

def test_run_one_round_mockadb():
    bot = mk_fast_bot()
    bot.run_one_round()
    # 至少应该有：launch / list_start / pre_start / 多次 support / collect
    names = [x[0] for x in bot.adb.log]
    assert "launch" in names, "未调用 launch()"
    assert "tap" in names, "未调用 tap()"
    assert ("key" in names) or True  # 允许不出现

    # 简单检查支援是否被点到（坐标不校验，仅检查次数 > 0）
    tap_cnt = sum(1 for x in bot.adb.log if x[0]=="tap")
    assert tap_cnt >= 5, f"tap 次数偏少：{tap_cnt}"
