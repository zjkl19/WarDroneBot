# YOLO Combat (PC inference, detect-only)

This path is isolated from the macro flow. Use it only when you want AI combat.

## 1) Dataset layout (YOLO)
```
datasets/combat_yolo/
  images/
    train/
    val/
  labels/
    train/
    val/
```

Classes (order matters):
- 0: enemy_vehicle
- 1: enemy_soldier

Use `configs/yolo_combat.yaml` as the data file.

## 2) Capture frames
You can reuse the existing capture tool:
```
python -m scripts.capture_dataset --state combat --count 100 --serial <adb-serial>
```
Move/rename the captured images into `datasets/combat_yolo/images/train` and label them.

## 3) Labeling
Use any YOLO labeling tool (LabelImg/CVAT).
- Only label enemy bodies.
- Do NOT label UI arrows or radar markers.

## 4) Train (Ultralytics)
Example:
```
yolo detect train data=configs/yolo_combat.yaml model=yolov8n.pt imgsz=640 epochs=100
```

## 5) Detect-only loop (no control)
```
python -m scripts.yolo_detect_only --serial <adb-serial> --model <weights>.pt --cfg configs/yolo_combat.json5
```
Optional:
```
--conf 0.25 --imgsz 640 --interval 0.3 --save-dir runs/yolo_debug --max-frames 200
```

## 6) UI masks / thresholds
Update `configs/yolo_combat.json5`:
- `masks`: rectangles to ignore UI areas.
- `min_box_px`: ignore tiny targets.
- `swipe_region`: safe swipe area for camera rotation.
- `fire.pos`: placeholder for fire button.
