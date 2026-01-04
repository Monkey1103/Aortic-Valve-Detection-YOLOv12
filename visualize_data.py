import os
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib
import numpy as np
matplotlib.rcParams['font.family'] = 'Microsoft JhengHei'  # 使用微軟正黑體
matplotlib.rcParams['axes.unicode_minus'] = False          # 正確顯示負號

# ==========================================================
# 路徑設定（請依照你的資料結構修改）
# ==========================================================
image_root = r"D:\task2\training_image"      # 主影像資料夾
label_root = r"D:\task2\training_label"      # YOLO txt 標記資料夾
output_root = r"D:\task2\train_vis"  # 輸出畫好框的影像
os.makedirs(output_root, exist_ok=True)

img_exts = {".jpg", ".png", ".jpeg", ".bmp"}
image_paths = {}
label_paths = {}

# 掃描影像
for root, dirs, files in os.walk(image_root):
    for f in files:
        if os.path.splitext(f)[1].lower() in img_exts:
            key = os.path.splitext(f)[0]
            image_paths[key] = os.path.join(root, f)

# 掃描標記
for root, dirs, files in os.walk(label_root):
    for f in files:
        if f.endswith(".txt"):
            key = os.path.splitext(f)[0]
            label_paths[key] = os.path.join(root, f)

# ======================================
# 統計 matched / unmatched
# ======================================
keys_images = set(image_paths.keys())
keys_labels = set(label_paths.keys())

matched = keys_images & keys_labels
unmatched = keys_images - keys_labels

print(f"有標記的影像：{len(matched)}")
print(f"沒有標記的影像：{len(unmatched)}")

# ======================================
# 圖 1：Matched / Unmatched 長條圖
# ======================================
plt.figure(figsize=(6,4))
plt.bar(["有標記", "無標記"], [len(matched), len(unmatched)])
plt.title("影像標記對應情況")
plt.ylabel("數量")
plt.tight_layout()
plt.show()

# ======================================
# 畫框並收集 bbox 統計
# ======================================
bbox_w, bbox_h = [], []

for key in tqdm(matched, desc="Drawing boxes"):
    img = cv2.imread(image_paths[key])
    h, w = img.shape[:2]

    with open(label_paths[key], "r") as f:
        lines = f.readlines()

    for line in lines:
        c, x, y, bw, bh = line.strip().split()
        x, y, bw, bh = map(float, [x, y, bw, bh])

        # 絕對像素
        abs_w = bw * w
        abs_h = bh * h
        bbox_w.append(abs_w)
        bbox_h.append(abs_h)

        cx = x * w
        cy = y * h
        x1 = int(cx - abs_w/2)
        y1 = int(cy - abs_h/2)
        x2 = int(cx + abs_w/2)
        y2 = int(cy + abs_h/2)

        cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(img, str(c), (x1, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    cv2.imwrite(os.path.join(output_root, f"{key}.jpg"), img)

bbox_w = np.array(bbox_w)
bbox_h = np.array(bbox_h)

# ======================================
# 統計資訊
# ======================================
stats = {
    "width_avg": bbox_w.mean(),
    "width_min": bbox_w.min(),
    "width_max": bbox_w.max(),
    "height_avg": bbox_h.mean(),
    "height_min": bbox_h.min(),
    "height_max": bbox_h.max(),
}

print("\n=== Bounding Box 統計 ===")
for k, v in stats.items():
    print(f"{k}: {v:.2f}")

# ======================================
# 圖 2：BBox 寬度直方圖
# ======================================
plt.figure(figsize=(6,4))
plt.hist(bbox_w, bins=50)
plt.title("邊界框寬度分布")
plt.xlabel("寬度 (pixel)")
plt.ylabel("數量")
plt.tight_layout()
plt.show()

# ======================================
# 圖 3：BBox 高度直方圖
# ======================================
plt.figure(figsize=(6,4))
plt.hist(bbox_h, bins=50)
plt.title("邊界框高度分布")
plt.xlabel("高度 (pixel)")
plt.ylabel("數量")
plt.tight_layout()
plt.show()

# ======================================
# 圖 4：寬度 vs 高度 散佈圖
# ======================================
plt.figure(figsize=(6,6))
plt.scatter(bbox_w, bbox_h, s=5)
plt.title("BBox 寬 vs 高 散佈圖")
plt.xlabel("寬度 (pixel)")
plt.ylabel("高度 (pixel)")
plt.tight_layout()
plt.show()

# ======================================
# 圖 5：寬度 / 高度 箱型圖（Boxplot）
# ======================================
plt.figure(figsize=(6,4))
plt.boxplot([bbox_w, bbox_h], labels=["寬度", "高度"])
plt.title("BBox 寬/高 Boxplot")
plt.tight_layout()
plt.show()

# ======================================
# 圖 6：寬度 統計數值長條圖
# ======================================
plt.figure(figsize=(6,4))
plt.bar(["平均", "最小", "最大"],
        [stats["width_avg"], stats["width_min"], stats["width_max"]])
plt.title("BBox 寬度統計")
plt.ylabel("像素")
plt.tight_layout()
plt.show()

# ======================================
# 圖 7：高度 統計數值長條圖
# ======================================
plt.figure(figsize=(6,4))
plt.bar(["平均", "最小", "最大"],
        [stats["height_avg"], stats["height_min"], stats["height_max"]])
plt.title("BBox 高度統計")
plt.ylabel("像素")
plt.tight_layout()
plt.show()