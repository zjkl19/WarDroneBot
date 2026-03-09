"""
Swipe校准测试脚本
用于测试swipe距离和实际视角移动的关系
考虑无人机自身的定速滑动

使用方法：
python swipe_calibrate.py --serial e5081c2a
"""

import argparse
import time
import math
import cv2
import numpy as np
from war_drone.adb_client import AdbClient
from datetime import datetime
import json5

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--serial", default=None, help="adb 序列号")
    ap.add_argument("--cfg", default="configs/yolo_combat.json5", help="配置文件")
    ap.add_argument("--show-preview", action="store_true", help="显示预览窗口")
    return ap.parse_args()

def _pct_to_px(p, wh):
    return int(p[0] * wh[0]), int(p[1] * wh[1])

def clamp(v, lo=0.0, hi=1.0):
    return max(lo, min(hi, v))

def calculate_optical_flow(img1, img2):
    """计算两帧之间的光流，估计整体位移"""
    # 转换为灰度图
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # 使用ORB特征点匹配
    orb = cv2.ORB_create(nfeatures=500)
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)
    
    if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
        return 0, 0, 0
    
    # 特征匹配
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    
    # 按距离排序，取前50个好匹配
    matches = sorted(matches, key=lambda x: x.distance)[:50]
    
    if len(matches) < 5:
        return 0, 0, 0
    
    # 计算位移
    dx_sum = 0
    dy_sum = 0
    valid_matches = 0
    
    for match in matches:
        pt1 = kp1[match.queryIdx].pt
        pt2 = kp2[match.trainIdx].pt
        
        # 过滤离群点（位移太大的可能是噪声）
        dx = pt2[0] - pt1[0]
        dy = pt2[1] - pt1[1]
        dist = math.hypot(dx, dy)
        
        if dist < 200:  # 过滤太离谱的匹配
            dx_sum += dx
            dy_sum += dy
            valid_matches += 1
    
    if valid_matches == 0:
        return 0, 0, 0
    
    avg_dx = dx_sum / valid_matches
    avg_dy = dy_sum / valid_matches
    
    return avg_dx, avg_dy, valid_matches

def calculate_template_match(img1, img2, roi=None):
    """使用模板匹配计算位移（适用于有静止参照物的情况）"""
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    if roi is None:
        # 默认使用中心区域作为模板
        h, w = gray1.shape
        roi = [w//4, h//4, w//2, h//2]
    
    x, y, w, h = roi
    template = gray1[y:y+h, x:x+w]
    
    # 模板匹配
    result = cv2.matchTemplate(gray2, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    
    if max_val < 0.3:  # 匹配度太低
        return 0, 0, 0
    
    dx = max_loc[0] - x
    dy = max_loc[1] - y
    
    return dx, dy, max_val

def main():
    args = parse_args()
    
    # 加载配置获取屏幕尺寸
    cfg = json5.load(open(args.cfg, "r", encoding="utf-8"))
    W, H = cfg["screen"]["width"], cfg["screen"]["height"]
    
    adb = AdbClient(serial=args.serial)
    
    print("="*60)
    print("Swipe校准测试脚本")
    print("="*60)
    print(f"屏幕分辨率: {W}x{H}")
    print("\n测试选项:")
    print("1. 单次滑动测试 - 测试特定滑动距离")
    print("2. 往复测试 - 滑动过去再回来，检查回位精度")
    print("3. 参数扫描 - 测试不同滑动距离和时间")
    print("4. 无人机定速测试 - 测量无人机自身漂移")
    print("="*60)
    
    choice = input("请选择测试类型 (1-4): ").strip()
    
    # 滑动区域设置
    print("\n滑动区域设置:")
    print("1. 屏幕中心")
    print("2. 屏幕下半部分（模拟手指）")
    print("3. 自定义区域")
    
    area_choice = input("请选择 (1-3): ").strip()
    
    if area_choice == "1":
        start_x_rel, start_y_rel = 0.5, 0.5
    elif area_choice == "2":
        start_x_rel, start_y_rel = 0.5, 0.75
    elif area_choice == "3":
        start_x_rel = float(input("起点X (0-1): "))
        start_y_rel = float(input("起点Y (0-1): "))
    else:
        start_x_rel, start_y_rel = 0.5, 0.75
    
    # 转换为像素
    start_x, start_y = _pct_to_px((start_x_rel, start_y_rel), (W, H))
    
    if choice == "1":
        # 单次滑动测试
        test_distance = int(input("测试滑动距离 (像素): "))
        test_duration = float(input("滑动时间 (秒, 如0.1): "))
        test_direction = input("方向 (left/right/up/down): ").strip().lower()
        
        # 计算终点
        if test_direction == "left":
            end_x = start_x - test_distance
            end_y = start_y
        elif test_direction == "right":
            end_x = start_x + test_distance
            end_y = start_y
        elif test_direction == "up":
            end_x = start_x
            end_y = start_y - test_distance
        elif test_direction == "down":
            end_x = start_x
            end_y = start_y + test_distance
        else:
            print("无效方向")
            return
        
        end_x = int(clamp(end_x, 0, W-1))
        end_y = int(clamp(end_y, 0, H-1))
        
        print(f"\n测试参数:")
        print(f"  起点: ({start_x}, {start_y})")
        print(f"  终点: ({end_x}, {end_y})")
        print(f"  滑动距离: {test_distance}px")
        print(f"  滑动时间: {test_duration}s")
        
        # 截取参考帧
        input("按Enter开始测试...")
        
        # 截取滑动前画面
        img_before = adb.screencap()
        if img_before is None:
            print("截屏失败")
            return
        
        # 执行滑动
        dur_ms = int(test_duration * 1000)
        print(f"执行滑动...")
        adb._cmd(["shell", "input", "swipe", 
                 str(start_x), str(start_y), 
                 str(end_x), str(end_y), 
                 str(dur_ms)])
        
        # 等待滑动完成
        time.sleep(test_duration + 0.1)
        
        # 截取滑动后画面
        img_after = adb.screencap()
        if img_after is None:
            print("截屏失败")
            return
        
        # 计算位移
        dx, dy, matches = calculate_optical_flow(img_before, img_after)
        
        print(f"\n结果:")
        print(f"  预期滑动: ({test_distance}px in {test_direction})")
        print(f"  实际画面位移: dx={dx:.1f}px, dy={dy:.1f}px")
        print(f"  位移距离: {math.hypot(dx, dy):.1f}px")
        print(f"  匹配特征点: {matches}")
        
        if test_direction in ["left", "right"]:
            ratio = abs(dx) / test_distance if test_distance != 0 else 0
            print(f"  滑动效率: {ratio:.2f} (1.0表示1:1映射)")
        else:
            ratio = abs(dy) / test_distance if test_distance != 0 else 0
            print(f"  滑动效率: {ratio:.2f} (1.0表示1:1映射)")
        
        # 推荐参数
        recommended_gain = 1.0 / ratio if ratio > 0 else 1.0
        print(f"\n推荐参数:")
        print(f"  swipe_gain = {recommended_gain:.2f}")
        
        # 显示预览
        if args.show_preview:
            # 创建对比图
            h, w = img_before.shape[:2]
            comparison = np.hstack([img_before, img_after])
            
            # 绘制特征点匹配（简化版）
            cv2.putText(comparison, f"Before", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(comparison, f"After", (w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(comparison, f"位移: ({dx:.1f}, {dy:.1f})px", (10, h-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # 调整大小显示
            scale = 0.5
            width = int(comparison.shape[1] * scale)
            height = int(comparison.shape[0] * scale)
            comparison = cv2.resize(comparison, (width, height))
            
            cv2.imshow("Swipe Test", comparison)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    
    elif choice == "2":
        # 往复测试
        test_distance = int(input("测试滑动距离 (像素): "))
        test_duration = float(input("滑动时间 (秒, 如0.1): "))
        
        print(f"\n测试参数:")
        print(f"  滑动距离: {test_distance}px")
        print(f"  滑动时间: {test_duration}s")
        
        input("按Enter开始测试 (向右滑动再向左滑回)...")
        
        # 截取初始画面
        img_initial = adb.screencap()
        if img_initial is None:
            print("截屏失败")
            return
        
        # 向右滑动
        end_x = min(start_x + test_distance, W-1)
        dur_ms = int(test_duration * 1000)
        print(f"向右滑动...")
        adb._cmd(["shell", "input", "swipe", 
                 str(start_x), str(start_y), 
                 str(end_x), str(end_y), 
                 str(dur_ms)])
        
        time.sleep(test_duration + 0.2)
        
        # 向左滑回
        print(f"向左滑回...")
        adb._cmd(["shell", "input", "swipe", 
                 str(end_x), str(end_y), 
                 str(start_x), str(start_y), 
                 str(dur_ms)])
        
        time.sleep(test_duration + 0.2)
        
        # 截取最终画面
        img_final = adb.screencap()
        if img_final is None:
            print("截屏失败")
            return
        
        # 计算净位移（应该接近0）
        dx, dy, matches = calculate_optical_flow(img_initial, img_final)
        
        print(f"\n往复测试结果:")
        print(f"  总滑动距离: {test_distance * 2}px")
        print(f"  净位移: dx={dx:.1f}px, dy={dy:.1f}px")
        print(f"  回位误差: {math.hypot(dx, dy):.1f}px")
        
        if abs(dx) < 10:
            print("  ✓ 回位精度高")
        elif abs(dx) < 30:
            print("  ⚠ 回位精度中等")
        else:
            print("  ✗ 回位精度低")
        
        # 如果回位误差大，说明有无人机自身漂移
        if abs(dx) > 20:
            print(f"\n检测到无人机自身漂移: {dx:.1f}px/周期")
    
    elif choice == "3":
        # 参数扫描
        print("\n参数扫描测试")
        print("将测试不同滑动距离和时间的组合")
        
        distances = [50, 100, 200]
        durations = [0.05, 0.1, 0.2]
        
        results = []
        
        for dist in distances:
            for dur in durations:
                print(f"\n测试: 距离={dist}px, 时间={dur}s")
                
                # 向右滑动
                end_x = min(start_x + dist, W-1)
                
                # 截取滑动前
                img_before = adb.screencap()
                if img_before is None:
                    continue
                
                # 执行滑动
                dur_ms = int(dur * 1000)
                adb._cmd(["shell", "input", "swipe", 
                         str(start_x), str(start_y), 
                         str(end_x), str(end_y), 
                         str(dur_ms)])
                
                time.sleep(dur + 0.2)
                
                # 截取滑动后
                img_after = adb.screencap()
                if img_after is None:
                    continue
                
                # 计算位移
                dx, dy, matches = calculate_optical_flow(img_before, img_after)
                
                results.append({
                    "distance": dist,
                    "duration": dur,
                    "dx": dx,
                    "dy": dy,
                    "matches": matches
                })
                
                print(f"  实际位移: {dx:.1f}px")
                
                # 等待一下再继续
                time.sleep(0.5)
        
        # 显示结果表格
        print("\n" + "="*60)
        print("参数扫描结果")
        print("="*60)
        print(f"{'距离(px)':<10} {'时间(s)':<10} {'实际位移(px)':<15} {'效率':<10}")
        print("-"*50)
        
        for r in results:
            efficiency = abs(r["dx"]) / r["distance"] if r["distance"] > 0 else 0
            print(f"{r['distance']:<10} {r['duration']:<10} {abs(r['dx']):<15.1f} {efficiency:<10.2f}")
        
        # 推荐最佳参数
        print("\n推荐参数组合:")
        best = max(results, key=lambda x: abs(x["dx"]) / x["distance"] if x["distance"] > 0 else 0)
        print(f"  距离: {best['distance']}px, 时间: {best['duration']}s, 效率: {abs(best['dx'])/best['distance']:.2f}")
    
    elif choice == "4":
        # 无人机定速测试
        print("\n无人机自身漂移测试")
        print("将在5秒内连续截图，测量自然漂移")
        
        test_duration = int(input("测试持续时间 (秒, 如5): "))
        
        input("请确保无人机处于悬停状态，按Enter开始...")
        
        frames = []
        timestamps = []
        
        for i in range(test_duration * 2):  # 2fps
            img = adb.screencap()
            if img is not None:
                frames.append(img)
                timestamps.append(time.time())
            time.sleep(0.5)
        
        if len(frames) < 2:
            print("截图不足")
            return
        
        # 计算每帧之间的位移
        total_dx = 0
        total_dy = 0
        valid_pairs = 0
        
        for i in range(len(frames) - 1):
            dx, dy, matches = calculate_optical_flow(frames[i], frames[i+1])
            if matches > 10:
                total_dx += dx
                total_dy += dy
                valid_pairs += 1
        
        if valid_pairs > 0:
            avg_drift_x = total_dx / valid_pairs
            avg_drift_y = total_dy / valid_pairs
            
            print(f"\n无人机漂移结果:")
            print(f"  平均每帧漂移: ({avg_drift_x:.1f}, {avg_drift_y:.1f})px")
            print(f"  漂移方向: {math.degrees(math.atan2(avg_drift_y, avg_drift_x)):.1f}°")
            print(f"  漂移速度: {math.hypot(avg_drift_x, avg_drift_y):.1f}px/帧")
            
            # 推荐补偿参数
            print(f"\n推荐补偿:")
            print(f"  如果需要补偿漂移，可在每次滑动时添加反向偏移")
    
    print("\n测试完成")

if __name__ == "__main__":
    main()