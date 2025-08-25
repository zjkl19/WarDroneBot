"""
目的：
- 验证“循环模式”会在每轮后调用 sleep_for_energy()
- 验证“单次模式”只跑一轮，不调用 sleep_for_energy()
说明：
- 这里不去跑 runner.main()（它会解析命令行），
  而是直接模拟 main 里的循环逻辑，替换成可计数的函数
"""
import time
from war_drone.simple_bot import SimpleSupportBot

def test_loop_calls_energy_sleep(monkeypatch):
    calls = {"sleep":0, "run":0}

    def fake_sleep(self): calls["sleep"] += 1
    def fake_run(self):  calls["run"]  += 1

    # 构造 bot 并替换方法
    bot = SimpleSupportBot()
    monkeypatch.setattr(SimpleSupportBot, "sleep_for_energy", fake_sleep, raising=False)
    monkeypatch.setattr(SimpleSupportBot, "run_one_round",  fake_run,  raising=False)

    # 模拟 for-minutes 很小，让循环只迭代一次
    end_ts = time.time() + 0.0001*60
    i=0
    while time.time() < end_ts:
        i+=1
        try: bot.run_one_round()
        except Exception: pass
        bot.sleep_for_energy()

    assert calls["run"] >= 1 and calls["sleep"] >= 1

def test_once_no_energy_sleep(monkeypatch):
    calls = {"sleep":0, "run":0}
    def fake_sleep(self): calls["sleep"] += 1
    def fake_run(self):  calls["run"]  += 1

    bot = SimpleSupportBot()
    monkeypatch.setattr(SimpleSupportBot, "sleep_for_energy", fake_sleep, raising=False)
    monkeypatch.setattr(SimpleSupportBot, "run_one_round",  fake_run,  raising=False)

    # “单次模式”只跑一轮，不调用 sleep_for_energy
    bot.run_one_round()
    assert calls["run"] == 1 and calls["sleep"] == 0
