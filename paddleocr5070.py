import paddle
from paddleocr import PaddleOCR, draw_ocr
import cv2
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 检查 GPU 和 PaddlePaddle 版本
print("PaddlePaddle版本:", paddle.__version__)
print("设备:", paddle.get_device())
print("CUDA支持:", paddle.is_compiled_with_cuda())
if paddle.device.is_compiled_with_cuda():
    print("GPU设备名称:", paddle.device.cuda.get_device_name())

# RTX 50系列需要确保安装正确的PaddlePaddle版本
# 建议安装：pip install paddlepaddle-gpu==2.5.2.post120 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html

# 初始化 OCR - 更新参数格式
ocr = PaddleOCR(
    use_angle_cls=True,    # 启用方向分类
    lang="ch",             # 中文识别
    use_gpu=True,          # 启用 GPU
    gpu_mem=8000,          # 显存限制（单位MB），RTX 5070通常有12GB+
    det_db_thresh=0.3,     # 文本框阈值
    det_db_box_thresh=0.5,
    det_db_unclip_ratio=1.6,
    max_text_length=50,    # 最大文本长度
    rec_image_shape="3, 48, 320",  # 识别图像尺寸
    show_log=False         # 关闭详细日志
)

# 读取图像
img_path = "image.jpg"
image = cv2.imread(img_path)
if image is None:
    raise FileNotFoundError(f"无法读取图片: {img_path}")

# OCR 识别 - 两种方式都有效
try:
    # 方法1：使用图片路径
    result = ocr.ocr(img_path, cls=True)
    # 方法2：使用numpy数组（已加载的图像）
    # result = ocr.ocr(image, cls=True)
except Exception as e:
    print(f"OCR识别失败: {e}")
    # 尝试关闭GPU加速
    print("尝试使用CPU模式...")
    ocr = PaddleOCR(use_gpu=False, lang="ch")
    result = ocr.ocr(img_path, cls=True)

# 处理结果
result = result[0] if result else []

if not result:
    print("未识别到任何文字！")
else:
    # 输出识别内容
    print(f"识别到 {len(result)} 个文本框:")
    for i, item in enumerate(result):
        box = item[0]  # 坐标框
        text, score = item[1]  # 文字和置信度
        print(f"{i+1}. 文字: {text}, 置信度: {score:.4f}")

# 可视化（仅在识别到内容时）
if result:
    boxes = [item[0] for item in result]
    txts = [item[1][0] for item in result]
    scores = [item[1][1] for item in result]
    
    # 绘制结果
    img_vis = draw_ocr(
        image, 
        boxes, 
        txts, 
        scores,
        font_path="simhei.ttf"  # Windows字体
        # Linux/Mac可以使用：font_path="/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"
    )
    
    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB))
    plt.title(f"OCR识别结果 - RTX 5070")
    plt.axis("off")
    plt.tight_layout()
    plt.show()
    
    # 可选：保存结果图像
    cv2.imwrite("ocr_result.jpg", img_vis)
    print("结果已保存为 ocr_result.jpg")