import os
import random
import shutil


# === CONFIG ===
base_path = "/user/christoph.wald/u15287/big-scratch/02_splitted_data/train_labeled/tiles/val"
images_path = os.path.join(base_path, "images")
labels_path = os.path.join(base_path, "labels")
seed = 42  # for reproducibility
target_background_ratio = 0.30  # 10%

# Lists
background_images = []
object_images = []

# Loop through images
for img_file in os.listdir(images_path):
   
    # Get corresponding label file (YOLO assumes same name but .txt)
    base_name = os.path.splitext(img_file)[0]
    label_file = os.path.join(labels_path, base_name + ".txt")
    img_path = os.path.join(images_path, img_file)


    # Check if label file is empty
    if os.path.getsize(label_file) == 0:
        background_images.append(img_path)
    else:
        object_images.append(img_path)

total_images = len(background_images) + len(object_images)
background_ratio = len(background_images) / total_images 
print(f"Total images: {total_images}")
print(f"Background images: {len(background_images)} ({background_ratio:.2%})")
print(f"Object images: {len(object_images)}")
'''
# === Calculate how many background images to keep for 10% ===
desired_background = int(target_background_ratio * total_images)
print(f"To reach {target_background_ratio*100:.0f}% background, keep {desired_background} images")

# === Random selection ===
random.seed(seed)
selected_background = random.sample(background_images, desired_background)

print(f"Selected {len(selected_background)} background images (seed={seed})")


# === CONFIG for output ===
out_base = "/user/christoph.wald/u15287/big-scratch/03_train_background_30"
out_images = os.path.join(out_base, "images/val")
out_labels = os.path.join(out_base, "labels/val")

os.makedirs(out_images, exist_ok=True)
os.makedirs(out_labels, exist_ok=True)

# helper to copy image + label
def copy_pair(img_path, labels_dir, out_images, out_labels):
    base = os.path.splitext(os.path.basename(img_path))[0]
    print("Copying ", base)
    label_file = os.path.join(labels_dir, base + ".txt")
    shutil.copy(img_path, out_images)
    if os.path.exists(label_file):
        shutil.copy(label_file, out_labels)

# copy object images
for img in object_images:
    copy_pair(img, labels_path, out_images, out_labels)

# copy selected background images
for img in selected_background:
    copy_pair(img, labels_path, out_images, out_labels)

print(f"Copied {len(object_images) + len(selected_background)} images to {out_base}")
'''