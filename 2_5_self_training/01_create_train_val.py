import os
import shutil

base_path = "/user/christoph.wald/u15287/big-scratch/02_splitted_data/train_labeled/SSL/split"

filetypes = ["images", "labels"]
usage = ["train", "val"]

# Create directory structure
for f in filetypes:
    for u in usage:
        os.makedirs(os.path.join(base_path, f, u), exist_ok=True)

image_src = "/user/christoph.wald/u15287/big-scratch/02_splitted_data/train_labeled/SSL/04_images_cropped"
labels_src = "/user/christoph.wald/u15287/big-scratch/02_splitted_data/train_labeled/SSL/05_created_labels"

with open("/user/christoph.wald/u15287/insect_pest_detection/2_4_image_processing/logs/images_not_processed_train_labeled.txt", "r") as f:
    not_processed = f.read().splitlines()

# Copy image and label files into train
for file in os.listdir(image_src):

    if file in not_processed:
        print(f"Skipping unprocessed file {file}")
        continue
    # copy file from image src to base_path/images/train
    src_img = os.path.join(image_src, file)
    dst_img = os.path.join(base_path, "images", "train", file)
    shutil.copy2(src_img, dst_img)

    # copy label file from label src to base_path/labels/train
    label_name = os.path.splitext(file)[0] + ".txt"
    src_label = os.path.join(labels_src, label_name)
    dst_label = os.path.join(base_path, "labels", "train", label_name)
    shutil.copy2(src_label, dst_label)

# copy folder /supervised/split/images/val to base_path/images/val
supervised_val_images = "/user/christoph.wald/u15287/big-scratch/02_splitted_data/train_labeled/supervised/split/images/val"
supervised_val_labels = "/user/christoph.wald/u15287/big-scratch/02_splitted_data/train_labeled/supervised/split/labels/val"

for file in os.listdir(supervised_val_images):
    shutil.copy2(
        os.path.join(supervised_val_images, file),
        os.path.join(base_path, "images", "val", file)
    )

for file in os.listdir(supervised_val_labels):
    shutil.copy2(
        os.path.join(supervised_val_labels, file),
        os.path.join(base_path, "labels", "val", file)
    )
