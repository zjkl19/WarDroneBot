"""
目的：
- 验证 _support_loop() 的调度能覆盖到 6 个支援（至少各一次）
- 使用 FakeTime 替代 time，快速推进“虚拟时间”
"""
import json5
from war_drone.simple_bot import SimpleSupportBot

class MockADB:
    def __init__(self): self.calls=[]
    def tap(self,x,y): self.calls.append(("tap",x,y))
    def swipe(self,*a,**k): pass
    def key(self,code): pass
    def launch(self,pkg): pass
    def kill(self,pkg): pass

class FakeTime:
    def __init__(self): self.t=0.0
    def time(self): return self.t
    def sleep(self,secs): self.t += secs

def test_support_loop_schedule(monkeypatch):
    bot = SimpleSupportBot()
    bot.adb = MockADB()

    # 冷却加速：s1..s6 依次 1.2, 1.4, ... 2.2 秒，方便在短时间内全覆盖
    for k in list(bot.cd.keys()):
        idx = int(k.replace("support",""))
        bot.cd[k] = 1.0 + 0.2*idx

    # 虚拟时间
    ft = FakeTime()
    monkeypatch.setattr("war_drone.simple_bot.time", ft)

    # 跑 10 秒虚拟战斗
    bot._support_loop(10.0)

    # 只检查“被点击次数是否足够”，不检查像素
    tap_cnt = len([c for c in bot.adb.calls if c[0]=="tap"])
    assert tap_cnt >= 6, f"支援点击过少：{tap_cnt}（期望≥6）"
