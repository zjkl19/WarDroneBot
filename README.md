# WarDroneBot (support-only, no-ads)

- 平台：Windows 10 + ADB + Python 3.10+
- 设备：小米 14（2670×1200 横屏）
- 游戏包名：com.miniclip.drone1
- 策略：只使用右下角 **支援**（坦克、狙击、范围炸弹1/2），不触发广告
- 体力：不足时自动等待 15 分钟（减 30s 安全裕度）

## 1. 安装
```bash
pip install -r requirements.txt
adb devices
