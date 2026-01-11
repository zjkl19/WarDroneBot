from ultralytics import YOLO

def main():
    model = YOLO("yolov8n.pt")
    model.train(
        data="configs/yolo_combat.yaml",
        imgsz=2670,      # 低显存安全
        batch=2,
        epochs=100,
        device="cpu",   # GT 730 不支持当前 CUDA，强制 CPU
        amp=False,      # 关闭 AMP 避免兼容问题
    )

if __name__ == "__main__":
    main()
