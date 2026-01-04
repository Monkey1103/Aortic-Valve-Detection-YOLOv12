# import locale
# locale.getpreferredencoding = lambda do_setlocale=True: "UTF-8"

# import os
# import shutil
# from ultralytics import YOLO
# import cv2
# import numpy as np

# # -----------------------------
# # 1️⃣ 清空舊資料夾
# folders = [r"D:\task2\predict_vis", r"D:\task2\predict_txt"]

# for folder in folders:
#     if os.path.exists(folder):
#         for filename in os.listdir(folder):
#             file_path = os.path.join(folder, filename)
#             if os.path.isfile(file_path):
#                 os.remove(file_path)
#                 print(f"Deleted: {file_path}")
#     else:
#         os.makedirs(folder, exist_ok=True)

# # -----------------------------
# # 2️⃣ 載入模型並做推論
# model = YOLO(r'D:\task2\run\yolo12x-neg\train\weights/best.pt')

# results = model.predict(
#     source=r"D:\task2\test_image",
#     save=False,
#     imgsz=640,
#     device=0,
#     augment=True,
# )

# # -----------------------------
# # 3️⃣ 生成 txt 檔
# output_dir = r"D:\task2\predict_txt"
# os.makedirs(output_dir, exist_ok=True)
# output_path = os.path.join(output_dir, 'test_image.txt')

# with open(output_path, 'w') as output_file:
#     for res in results:
#         filename = os.path.splitext(os.path.basename(res.path))[0]

#         boxes = res.boxes
#         if boxes is None or len(boxes) == 0:
#             continue

#         xyxy = boxes.xyxy.cpu().numpy()
#         confs = boxes.conf.cpu().numpy()
#         labels = boxes.cls.cpu().numpy().astype(int)

#         for (x1, y1, x2, y2), conf, label in zip(xyxy, confs, labels):
#             line = f"{filename} {label} {conf:.4f} {int(x1)} {int(y1)} {int(x2)} {int(y2)}\n"
#             output_file.write(line)

# print(f"TXT saved to {output_path}")

# # -----------------------------
# # 4️⃣ 可視化影像並保存
# save_vis_dir = r"D:\task2\predict_vis"
# os.makedirs(save_vis_dir, exist_ok=True)

# # 預設顏色列表，可循環使用
# colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0)]

# for res in results:
#     img_path = res.path
#     img = cv2.imread(img_path)
#     if img is None:
#         print(f"Failed to read {img_path}")
#         continue

#     boxes = res.boxes
#     if boxes is None or len(boxes) == 0:
#         continue

#     xyxy = boxes.xyxy.cpu().numpy()
#     confs = boxes.conf.cpu().numpy()
#     labels = boxes.cls.cpu().numpy().astype(int)

#     # for i, ((x1, y1, x2, y2), conf, label) in enumerate(zip(xyxy, confs, labels)):
#     #     x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
#     #     color = colors[i % len(colors)]  # 循環選顏色
#     #     cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
#     #     text = f"{label} {conf:.2f}"
#     #     cv2.putText(img, text, (x1, max(y1-5, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

#     for i, ((x1, y1, x2, y2), conf, label) in enumerate(zip(xyxy, confs, labels)):
#         x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
#         color = colors[i % len(colors)]  # 循環選顏色
#         cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

#         # 計算文字偏移，避免重疊
#         text_offset = 15 + i * 15  # 第一個框上方15px，第二個框上方30px，依此類推
#         text_pos = (x1, max(y1 - text_offset, 0))

#         text = f"{label} {conf:.2f}"
#         cv2.putText(img, text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

#     save_path = os.path.join(save_vis_dir, os.path.basename(img_path))
#     cv2.imwrite(save_path, img)
#     print(f"Saved visualization: {save_path}")


#---------------------------------------------------------------------------------------------


import locale
locale.getpreferredencoding = lambda do_setlocale=True: "UTF-8"

import os
import shutil
from ultralytics import YOLO
import cv2
import numpy as np

# -----------------------------
# 1️⃣ 清空舊資料夾
folders = [r"D:\task2\predict_vis", r"D:\task2\predict_txt"]

for folder in folders:
    if os.path.exists(folder):
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"Deleted: {file_path}")
    else:
        os.makedirs(folder, exist_ok=True)

# -----------------------------
# 2️⃣ 載入模型
model = YOLO(r'D:\task2\run\yolo12x-neg\train\weights/best.pt')

results = model.predict(
    source=r"D:\task2\test_image",
    save=False,
    imgsz=640,
    device=0,
    augment=True,
)

# -----------------------------
# 3️⃣ 生成 txt 檔
output_dir = r"D:\task2\predict_txt"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, 'test_image.txt')

with open(output_path, 'w') as output_file:
    for res in results:
        filename = os.path.splitext(os.path.basename(res.path))[0]

        boxes = res.boxes
        if boxes is None or len(boxes) == 0:
            continue

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        labels = boxes.cls.cpu().numpy().astype(int)

        # -----------------------------
        # ★ 若偵測到兩個以上 → 僅保留面積最小的框
        # -----------------------------
        if len(xyxy) > 1:
            areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
            smallest_idx = np.argmin(areas)

            xyxy = xyxy[smallest_idx:smallest_idx+1]
            confs = confs[smallest_idx:smallest_idx+1]
            labels = labels[smallest_idx:smallest_idx+1]

        # -----------------------------
        # 寫入 TXT（只 1 個框）
        # -----------------------------
        for (x1, y1, x2, y2), conf, label in zip(xyxy, confs, labels):
            line = f"{filename} {label} {conf:.4f} {int(x1)} {int(y1)} {int(x2)} {int(y2)}\n"
            output_file.write(line)

print(f"TXT saved to {output_path}")

# -----------------------------
# 4️⃣ 可視化影像（也只畫面積最小框）
save_vis_dir = r"D:\task2\predict_vis"
os.makedirs(save_vis_dir, exist_ok=True)

for res in results:
    img_path = res.path
    img = cv2.imread(img_path)
    if img is None:
        print(f"Failed to read {img_path}")
        continue

    boxes = res.boxes
    if boxes is None or len(boxes) == 0:
        continue

    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    labels = boxes.cls.cpu().numpy().astype(int)

    # -----------------------------
    # ★ 取最小 bbox
    # -----------------------------
    if len(xyxy) > 1:
        areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
        smallest_idx = np.argmin(areas)

        xyxy = xyxy[smallest_idx:smallest_idx+1]
        confs = confs[smallest_idx:smallest_idx+1]
        labels = labels[smallest_idx:smallest_idx+1]

    # -----------------------------
    # 畫 bounding box（只 1 個）
    # -----------------------------
    for (x1, y1, x2, y2), conf, label in zip(xyxy, confs, labels):
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        text = f"{label} {conf:.2f}"
        cv2.putText(img, text, (x1, max(y1-5, 0)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    save_path = os.path.join(save_vis_dir, os.path.basename(img_path))
    cv2.imwrite(save_path, img)
    print(f"Saved visualization: {save_path}")


#---------------------------------------------------------------------------------------





# import locale
# locale.getpreferredencoding = lambda do_setlocale=True: "UTF-8"

# import os
# import shutil
# from ultralytics import YOLO
# import cv2
# import numpy as np

# # -----------------------------
# # 1️⃣ 清空舊資料夾
# folders = [r"D:\task2\predict_vis", r"D:\task2\predict_txt"]

# for folder in folders:
#     if os.path.exists(folder):
#         for filename in os.listdir(folder):
#             file_path = os.path.join(folder, filename)
#             if os.path.isfile(file_path):
#                 os.remove(file_path)
#                 print(f"Deleted: {file_path}")
#     else:
#         os.makedirs(folder, exist_ok=True)

# # -----------------------------
# # 2️⃣ 載入模型
# model = YOLO(r'D:\task2\run\yolo12x\train\weights\best.pt')

# results = model.predict(
#     source=r"D:\task2\test_image",
#     save=False,
#     imgsz=640,
#     device=0,
#     augment=True,
# )

# # -----------------------------
# # 3️⃣ 生成 txt 檔（只保留最大框）
# output_dir = r"D:\task2\predict_txt"
# os.makedirs(output_dir, exist_ok=True)
# output_path = os.path.join(output_dir, 'test_image.txt')

# with open(output_path, 'w') as output_file:
#     for res in results:
#         filename = os.path.splitext(os.path.basename(res.path))[0]

#         boxes = res.boxes
#         if boxes is None or len(boxes) == 0:
#             continue

#         xyxy = boxes.xyxy.cpu().numpy()
#         confs = boxes.conf.cpu().numpy()
#         labels = boxes.cls.cpu().numpy().astype(int)

#         # -----------------------------
#         # ★ 若偵測到兩個以上 → 僅保留面積最大的框
#         # -----------------------------
#         if len(xyxy) > 1:
#             areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
#             largest_idx = np.argmax(areas)   # <-- ★ 改成取最大面積★

#             xyxy = xyxy[largest_idx:largest_idx+1]
#             confs = confs[largest_idx:largest_idx+1]
#             labels = labels[largest_idx:largest_idx+1]

#         # -----------------------------
#         # 寫入 TXT（只 1 個框）
#         # -----------------------------
#         for (x1, y1, x2, y2), conf, label in zip(xyxy, confs, labels):
#             line = f"{filename} {label} {conf:.4f} {int(x1)} {int(y1)} {int(x2)} {int(y2)}\n"
#             output_file.write(line)

# print(f"TXT saved to {output_path}")

# # -----------------------------
# # 4️⃣ 可視化影像（只畫最大框）
# save_vis_dir = r"D:\task2\predict_vis"
# os.makedirs(save_vis_dir, exist_ok=True)

# for res in results:
#     img_path = res.path
#     img = cv2.imread(img_path)
#     if img is None:
#         print(f"Failed to read {img_path}")
#         continue

#     boxes = res.boxes
#     if boxes is None or len(boxes) == 0:
#         continue

#     xyxy = boxes.xyxy.cpu().numpy()
#     confs = boxes.conf.cpu().numpy()
#     labels = boxes.cls.cpu().numpy().astype(int)

#     # -----------------------------
#     # ★ 留下面積最大的 bbox
#     # -----------------------------
#     if len(xyxy) > 1:
#         areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
#         largest_idx = np.argmax(areas)   # <-- ★ 改成最大★

#         xyxy = xyxy[largest_idx:largest_idx+1]
#         confs = confs[largest_idx:largest_idx+1]
#         labels = labels[largest_idx:largest_idx+1]

#     # -----------------------------
#     # 畫 bounding box（只 1 個）
#     # -----------------------------
#     for (x1, y1, x2, y2), conf, label in zip(xyxy, confs, labels):
#         x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
#         cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

#         text = f"{label} {conf:.2f}"
#         cv2.putText(img, text, (x1, max(y1-5, 0)),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

#     save_path = os.path.join(save_vis_dir, os.path.basename(img_path))
#     cv2.imwrite(save_path, img)
#     print(f"Saved visualization: {save_path}")

# print("Done! Updated to keep ONLY the largest bounding box.")
