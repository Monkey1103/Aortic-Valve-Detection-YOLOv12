import os
import cv2
from ultralytics import YOLO

# ----------------------------
# 設定路徑
# ----------------------------
model_path = r"D:\task2\run\yolo12x-neg\train\weights/best.pt"
val_img_dir = r"D:\task2\new datasets\val\images"
val_label_dir = r"D:\task2\new datasets\val\labels"
save_dir = r"D:\task2\val_vis"
os.makedirs(save_dir, exist_ok=True)

# ----------------------------
# 載入 YOLO 模型
# ----------------------------
model = YOLO(model_path)

# ----------------------------
# YOLO 標註 (cx,cy,w,h) → xyxy
# ----------------------------
def yolo_to_xyxy(cx, cy, w, h, img_w, img_h):
    x1 = int((cx - w / 2) * img_w)
    y1 = int((cy - h / 2) * img_h)
    x2 = int((cx + w / 2) * img_w)
    y2 = int((cy + h / 2) * img_h)
    return x1, y1, x2, y2

# ----------------------------
# 進行推論
# ----------------------------
results = model.predict(
    source=val_img_dir,
    verbose=True,
    save=False,
    imgsz=640,
    device=0,
    augment=True,
)

# ----------------------------
# 處理每張影像 & 可視化
# ----------------------------
for res in results:
    img_path = res.path
    img_name = os.path.basename(img_path)

    img = cv2.imread(img_path)
    if img is None:
        print(f"Read failed: {img_path}")
        continue

    h, w = img.shape[:2]

    # ----------------------------
    # Ground Truth
    # ----------------------------
    label_path = os.path.join(val_label_dir, os.path.splitext(img_name)[0] + ".txt")
    gt_lines = []
    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            gt_lines = [line.strip() for line in f.readlines() if line.strip()]

    has_gt = len(gt_lines) > 0
    has_pred = (res.boxes is not None and len(res.boxes) > 0)

    # Case 1: 有 GT 但沒有預測
    if has_gt and not has_pred:
        print(f"[ONLY GT] No prediction for image: {img_name}")

    # Case 2: 有預測但沒有 GT
    if has_pred and not has_gt:
        print(f"[ONLY PRED] No ground truth for image: {img_name}")

    # ----------------------------
    # 畫 Ground Truth（綠色）
    # ----------------------------
    if has_gt:
        for line in gt_lines:
            cls, cx, cy, bw, bh = line.split()
            cx, cy, bw, bh = map(float, (cx, cy, bw, bh))
            x1, y1, x2, y2 = yolo_to_xyxy(cx, cy, bw, bh, w, h)

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"GT {cls}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # ----------------------------
    # 畫預測框（紅色）
    # ----------------------------
    if has_pred:
        for box in res.boxes:
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            x1, y1, x2, y2 = xyxy

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(img, f"Pred {cls_id} {conf:.2f}",
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # ----------------------------
    # 儲存影像
    # ----------------------------
    save_path = os.path.join(save_dir, img_name)
    cv2.imwrite(save_path, img)
    # print(f"Saved: {save_path}")

print("Done! 所有影像已可視化。")
