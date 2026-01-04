import os
import shutil
import random

# ==== 來源資料 ====
with_label_img_dir = r"D:\task2\data\with_label\images"
with_label_lbl_dir = r"D:\task2\data\with_label\labels"

no_label_img_dir = r"D:\task2\data\no_label\images"
no_label_lbl_dir = r"D:\task2\data\no_label\labels"

# ==== 輸出資料 ====
output_root = r"D:\task2\datasets"
train_img_out = os.path.join(output_root, "train", "images")
train_lbl_out = os.path.join(output_root, "train", "labels")
val_img_out   = os.path.join(output_root, "val", "images")
val_lbl_out   = os.path.join(output_root, "val", "labels")

for p in [train_img_out, train_lbl_out, val_img_out, val_lbl_out]:
    os.makedirs(p, exist_ok=True)

# ==== 設定比例 ====
mix_ratio_no = 1     # 無 label
mix_ratio_yes = 1    # 有 label

train_ratio = 0.85
val_ratio = 0.15

# ==== 收集檔案 ====
with_label_files = [
    f for f in os.listdir(with_label_img_dir)
    if os.path.splitext(f)[1].lower() in {".jpg", ".jpeg", ".png"}
]

no_label_files = [
    f for f in os.listdir(no_label_img_dir)
    if os.path.splitext(f)[1].lower() in {".jpg", ".jpeg", ".png"}
]

# ==== 決定混和數量 ====
max_pairs = min(len(with_label_files) // mix_ratio_yes,
                len(no_label_files) // mix_ratio_no)

use_yes = max_pairs * mix_ratio_yes
use_no  = max_pairs * mix_ratio_no

random.shuffle(with_label_files)
random.shuffle(no_label_files)

mixed_images = (
    [("yes", f) for f in with_label_files[:use_yes]] +
    [("no", f) for f in no_label_files[:use_no]]
)

random.shuffle(mixed_images)

# ==== 依 8:2 切分 Train/Val ====
split_idx = int(len(mixed_images) * train_ratio)

train_set = mixed_images[:split_idx]
val_set = mixed_images[split_idx:]

def copy_pair(src_img, src_lbl, dst_img, dst_lbl):
    shutil.copy(src_img, dst_img)
    shutil.copy(src_lbl, dst_lbl)

def process_set(dataset, img_out, lbl_out):
    for label_type, filename in dataset:
        name, ext = os.path.splitext(filename)

        if label_type == "yes":
            src_img = os.path.join(with_label_img_dir, filename)
            src_lbl = os.path.join(with_label_lbl_dir, name + ".txt")
        else:
            src_img = os.path.join(no_label_img_dir, filename)
            src_lbl = os.path.join(no_label_lbl_dir, name + ".txt")

        dst_img = os.path.join(img_out, filename)
        dst_lbl = os.path.join(lbl_out, name + ".txt")

        shutil.copy(src_img, dst_img)
        shutil.copy(src_lbl, dst_lbl)

# ==== 開始處理 ====
process_set(train_set, train_img_out, train_lbl_out)
process_set(val_set,   val_img_out,   val_lbl_out)

print("處理完成！")
print(f"Train 數量：{len(train_set)}")
print(f"Val   數量：{len(val_set)}")
