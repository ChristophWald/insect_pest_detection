import os
import cv2

# Paths
image_folder = "/user/christoph.wald/u15287/big-scratch/04_SSL_training_data/training_data/images/train"       # folder with images
label_folder = "/user/christoph.wald/u15287/big-scratch/04_SSL_training_data/training_data/labels/absolute"       # folder with original txts
yolo_folder = "/user/christoph.wald/u15287/big-scratch/04_SSL_training_data/training_data/labels/train"   # folder to save YOLO txts
os.makedirs(yolo_folder, exist_ok=True)

# Map filename prefixes to class ids
class_map = {
    "BRAIIM": 0,
    "LIRIBO": 1,
    "TRIAVA": 3
}

# Process each label txt
for txt_file in os.listdir(label_folder):
    if not txt_file.endswith(".txt"):
        continue

    # Determine class id from filename
    class_id = None
    for key, value in class_map.items():
        if txt_file.startswith(key):
            class_id = value
            break
    if class_id is None:
        print(f"Skipping {txt_file}, class not found")
        continue

    # Corresponding image file (adjust extension if needed)
    img_name = os.path.splitext(txt_file)[0] + ".jpg"
    img_path = os.path.join(image_folder, img_name)
    if not os.path.exists(img_path):
        print(f"Image not found for {txt_file}, skipping")
        continue

    # Get image size using cv2
    img = cv2.imread(img_path)
    img_h, img_w = img.shape[:2]

    # Read bounding boxes
    with open(os.path.join(label_folder, txt_file), "r") as f:
        lines = f.readlines()

    yolo_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Convert string to list (assumes format [x_min, y_min, w, h])
        bbox = eval(line)
        x_min, y_min, w, h = bbox

        # Convert to YOLO format
        x_center = x_min + w / 2
        y_center = y_min + h / 2

        x_center_norm = x_center / img_w
        y_center_norm = y_center / img_h
        w_norm = w / img_w
        h_norm = h / img_h

        yolo_lines.append(f"{class_id} {x_center_norm:.6f} {y_center_norm:.6f} {w_norm:.6f} {h_norm:.6f}")

    # Save YOLO txt
    with open(os.path.join(yolo_folder, txt_file), "w") as f:
        f.write("\n".join(yolo_lines))
