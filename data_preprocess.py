import os
import shutil

# === 設定你的路徑 ===
images_root = r"D:\task2\training_image"
labels_root = r"D:\task2\training_label"

output_with_label_img = r"D:\task2\data\with_label\images"
output_with_label_lbl = r"D:\task2\data\with_label\labels"

output_no_label_img = r"D:\task2\data\no_label\images"
output_no_label_lbl = r"D:\task2\data\no_label\labels"

# 建立輸出資料夾
for p in [output_with_label_img, output_with_label_lbl, output_no_label_img, output_no_label_lbl]:
    os.makedirs(p, exist_ok=True)

# 支援的影像格式
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# 把 label 目錄中的所有 label path 建成 dictionary（加速查詢）
label_dict = {}

for root, dirs, files in os.walk(labels_root):
    for f in files:
        if f.endswith(".txt"):
            name = os.path.splitext(f)[0]
            full_path = os.path.join(root, f)
            label_dict[name] = full_path  # key = 檔名(不含副檔名)

print(f"共偵測到 {len(label_dict)} 個 label 檔案")

# ===== 主流程：掃描所有影像 =====
for root, dirs, files in os.walk(images_root):
    for f in files:
        ext = os.path.splitext(f)[1].lower()
        if ext not in IMG_EXT:
            continue

        img_name = os.path.splitext(f)[0]
        img_full_path = os.path.join(root, f)

        # 是否有對應 label?
        if img_name in label_dict:
            # Copy 有 label 的影像與 label
            shutil.copy(img_full_path, os.path.join(output_with_label_img, f))
            shutil.copy(label_dict[img_name],
                        os.path.join(output_with_label_lbl, img_name + ".txt"))
        else:
            # 沒 label → 建一個空的 txt
            shutil.copy(img_full_path, os.path.join(output_no_label_img, f))

            empty_label_path = os.path.join(output_no_label_lbl, img_name + ".txt")
            with open(empty_label_path, "w") as fp:
                fp.write("")  # 建立空檔案

print("處理完成！")
