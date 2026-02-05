import os
import sys
import cv2
import ssl
import easyocr
import urllib.request
import traceback

def setup_ssl_context():
    """配置SSL上下文以解决证书验证问题"""
    try:
        # 尝试创建一个不验证证书的上下文（临时解决方案）
        ssl_context = ssl._create_unverified_context()
        # 应用这个上下文到urllib
        urllib.request.install_opener(
            urllib.request.build_opener(
                urllib.request.HTTPSHandler(context=ssl_context)
            )
        )
        print("[SSL] 已配置SSL上下文（临时绕过证书验证）")
        return True
    except Exception as e:
        print(f"[SSL] 配置SSL上下文失败: {e}")
        return False

def check_gpu_availability():
    """检查GPU是否可用"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown"
            print(f"[GPU] 检测到CUDA设备: {device_name}")
            print(f"[GPU] 可用GPU数量: {gpu_count}")
            
            # 测试GPU计算
            test_tensor = torch.tensor([1.0]).cuda()
            _ = test_tensor * 2
            print("[GPU] GPU计算测试通过")
            return True
        else:
            print("[GPU] CUDA不可用，将使用CPU模式")
            return False
    except ImportError:
        print("[GPU] 未安装PyTorch或PyTorch版本不支持CUDA")
        print("      请安装: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        return False
    except Exception as e:
        print(f"[GPU] GPU检查失败: {e}")
        return False

def download_models_manually():
    """提供手动下载模型的指引"""
    print("\n" + "="*60)
    print("模型下载指引（如果自动下载失败）")
    print("="*60)
    print("EasyOCR需要下载以下模型文件：")
    print("1. 中文模型: zh_sim_g2.zip")
    print("2. 英文模型: english_g2.zip")
    print("\n手动下载步骤：")
    print("1. 访问: https://github.com/JaidedAI/EasyOCR/releases")
    print("2. 下载上述模型文件")
    print("3. 解压到以下目录：")
    print(f"   {os.path.expanduser('~')}\\.EasyOCR\\model\\")
    print("4. 重新运行此程序")
    print("="*60 + "\n")
    
    # 创建目录结构
    model_dir = os.path.expanduser('~') + "\\.EasyOCR\\model\\"
    os.makedirs(model_dir, exist_ok=True)
    print(f"[目录] 已创建模型目录: {model_dir}")
    return model_dir

def main():
    img_path = r"D:\PythonProjects\WarDroneBot\captures\settlement.png"
    
    # 1. 文件检查
    if not os.path.exists(img_path):
        print(f"[错误] 找不到文件: {img_path}")
        sys.exit(1)
    
    if not os.path.isfile(img_path):
        print(f"[错误] 路径不是文件: {img_path}")
        sys.exit(1)
    
    print("="*60)
    print(f"处理图片: {os.path.basename(img_path)}")
    print(f"图片路径: {img_path}")
    print("="*60)
    
    # 2. 解决SSL证书问题
    print("\n[步骤1] 配置网络环境...")
    setup_ssl_context()
    
    # 3. 检查GPU可用性
    print("\n[步骤2] 检查GPU环境...")
    gpu_available = check_gpu_availability()
    
    # 4. 初始化EasyOCR
    print("\n[步骤3] 初始化EasyOCR...")
    
    # 设置模型存储路径
    model_dir = os.path.expanduser('~') + "\\.EasyOCR\\model\\"
    os.makedirs(model_dir, exist_ok=True)
    
    # 尝试不同策略初始化EasyOCR
    max_retries = 2
    reader = None
    
    for attempt in range(max_retries):
        try:
            print(f"\n尝试初始化（第 {attempt+1} 次）...")
            
            if attempt == 0:
                # 第一次尝试：使用GPU（如果可用）并启用下载
                use_gpu = gpu_available
                download_enabled = True
                print(f"  配置: GPU={use_gpu}, 自动下载=开启")
            else:
                # 第二次尝试：使用CPU并禁止下载（假设已手动下载）
                use_gpu = False
                download_enabled = False
                print(f"  配置: GPU={use_gpu}, 自动下载=关闭（使用本地模型）")
            
            reader = easyocr.Reader(
                lang_list=['ch_sim', 'en'],
                gpu=use_gpu,
                model_storage_directory=model_dir,
                download_enabled=download_enabled,
                verbose=False  # 减少日志输出
            )
            
            print("✅ EasyOCR初始化成功")
            break
            
        except Exception as e:
            error_msg = str(e).lower()
            print(f"❌ 初始化失败: {e}")
            
            if "certificate" in error_msg or "ssl" in error_msg:
                print("   原因: SSL证书验证失败")
                if attempt == 0:
                    print("   将尝试使用更宽松的SSL设置...")
                    # 更激进的SSL绕过
                    import ssl
                    ssl._create_default_https_context = ssl._create_unverified_context
            elif "download" in error_msg or "model" in error_msg:
                print("   原因: 模型下载失败")
                if attempt == 0:
                    print("   将尝试使用本地模型...")
            else:
                print(f"   未知错误类型")
            
            if attempt == max_retries - 1:
                print("\n⚠️ 所有初始化尝试都失败了！")
                download_models_manually()
                print("请手动下载模型后重新运行程序。")
                sys.exit(1)
    
    if reader is None:
        print("[错误] 无法初始化EasyOCR")
        sys.exit(1)
    
    # 5. 读取图片
    print("\n[步骤4] 读取图片...")
    img = cv2.imread(img_path)
    if img is None:
        print(f"[错误] OpenCV无法读取图片: {img_path}")
        print("可能原因: 文件损坏、路径错误、权限问题")
        sys.exit(1)
    
    print(f"图片尺寸: {img.shape[1]}x{img.shape[0]}, 通道数: {img.shape[2]}")
    
    # 6. 执行OCR
    print("\n[步骤5] 执行OCR识别...")
    
    try:
        # 可以根据图片大小调整参数
        height, width = img.shape[:2]
        
        # 根据图片大小调整参数
        if max(height, width) > 2000:
            print("检测到大尺寸图片，调整识别参数...")
            results = reader.readtext(
                img,
                detail=1,
                paragraph=False,
                width_ths=0.7,    # 放宽宽度阈值
                height_ths=0.7,   # 放宽高度阈值
                ycenter_ths=0.5   # 放宽Y中心阈值
            )
        else:
            results = reader.readtext(
                img,
                detail=1,
                paragraph=False,
                batch_size=10,    # 批处理大小
                workers=4         # 工作线程数
            )
        
    except Exception as e:
        print(f"[错误] OCR识别失败: {e}")
        print("尝试简化参数重新识别...")
        results = reader.readtext(img, detail=1, paragraph=False)
    
    # 7. 处理识别结果
    print("\n" + "="*60)
    print("OCR识别结果")
    print("="*60)
    
    if not results:
        print("[提示] 未识别到任何文本")
        return
    
    print(f"识别到 {len(results)} 个文本区域")
    
    # 按Y坐标排序（从上到下）
    results_sorted = sorted(results, key=lambda x: x[0][0][1])
    
    for i, (bbox, text, conf) in enumerate(results_sorted, start=1):
        # 计算边框的边界
        xs = [int(p[0]) for p in bbox]
        ys = [int(p[1]) for p in bbox]
        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        
        # 格式化输出
        confidence_icon = '✓' if conf > 0.8 else ('~' if conf > 0.5 else '?')
        area = (x2 - x1) * (y2 - y1)
        
        print(f"\n{i:03d}. 文本: {text}")
        print(f"    置信度: {conf:.4f} {confidence_icon}")
        print(f"    位置: ({x1}, {y1}) -> ({x2}, {y2})")
        print(f"    尺寸: {x2-x1}x{y2-y1} (面积: {area})")
        
        if conf < 0.5:
            print(f"    ⚠️  置信度较低，建议人工核对")
        if conf > 0.95:
            print(f"    ✅ 高置信度文本")
    
    # 8. 可视化结果
    print("\n[步骤6] 生成可视化结果...")
    vis = img.copy()
    
    # 为不同置信度使用不同颜色
    for bbox, text, conf in results:
        xs = [int(p[0]) for p in bbox]
        ys = [int(p[1]) for p in bbox]
        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        
        # 根据置信度选择颜色
        if conf > 0.8:
            color = (0, 255, 0)    # 绿色 - 高置信度
        elif conf > 0.5:
            color = (0, 200, 255)  # 黄色 - 中等置信度
        else:
            color = (0, 0, 255)    # 红色 - 低置信度
        
        # 绘制边框
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        
        # 绘制置信度
        label = f"{conf:.2f}"
        (label_width, label_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        
        # 绘制文本背景
        cv2.rectangle(
            vis,
            (x1, max(0, y1 - label_height - 10)),
            (x1 + label_width, max(0, y1)),
            color,
            -1  # 填充
        )
        
        # 绘制文本
        cv2.putText(
            vis,
            label,
            (x1, max(5, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),  # 白色文字
            2
        )
    
    # 保存结果
    out_dir = os.path.dirname(img_path)
    out_name = os.path.splitext(os.path.basename(img_path))[0]
    out_path = os.path.join(out_dir, f"{out_name}_easyocr_result.png")
    
    cv2.imwrite(out_path, vis)
    print(f"✅ 可视化结果已保存: {out_path}")
    
    # 9. 统计信息
    print("\n" + "="*60)
    print("识别统计")
    print("="*60)
    
    if results:
        confidences = [conf for _, _, conf in results]
        avg_conf = sum(confidences) / len(confidences)
        high_conf = sum(1 for conf in confidences if conf > 0.8)
        medium_conf = sum(1 for conf in confidences if 0.5 < conf <= 0.8)
        low_conf = sum(1 for conf in confidences if conf <= 0.5)
        
        print(f"平均置信度: {avg_conf:.4f}")
        print(f"高置信度 (>0.8): {high_conf} 个")
        print(f"中置信度 (0.5-0.8): {medium_conf} 个")
        print(f"低置信度 (≤0.5): {low_conf} 个")
        
        if low_conf > 0:
            print(f"⚠️  有 {low_conf} 个低置信度文本需要人工核对")
        
        # 显示最高和最低的识别结果
        if results:
            best_result = max(results, key=lambda x: x[2])
            worst_result = min(results, key=lambda x: x[2])
            print(f"\n最高置信度文本: \"{best_result[1]}\" ({best_result[2]:.4f})")
            print(f"最低置信度文本: \"{worst_result[1]}\" ({worst_result[2]:.4f})")
    
    print("\n" + "="*60)
    print("OCR处理完成！")
    print("="*60)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[用户中断] 程序被用户终止")
    except Exception as e:
        print(f"\n[未处理异常] 程序运行出错: {e}")
        traceback.print_exc()
        print("\n建议解决方案:")
        print("1. 检查图片文件是否完整")
        print("2. 手动下载EasyOCR模型文件")
        print("3. 使用PaddleOCR替代: pip install paddleocr")