import os
import random

input_dir = "/user/christoph.wald/u15287/big-scratch/splitted_data/train1_labeled/tiles/images"
base_dir = "/user/christoph.wald/u15287/big-scratch/splitted_data/train1_labeled/split"

# Output folders
img_train = os.path.join(base_dir, 'images/train')
img_val = os.path.join(base_dir, 'images/val')
lbl_train = os.path.join(base_dir, 'labels/train')
lbl_val = os.path.join(base_dir, 'labels/val')

# Create output folders
for d in [img_train, img_val, lbl_train, lbl_val]:
    os.makedirs(d, exist_ok=True)

pest_types = ["BRAIIM", "FRANOC", "LIRIBO", "TRIAVA"]
image_files = [f for f in os.listdir(input_dir) if f.endswith(('.jpg', '.png'))]
separated = []

train_files = []
val_files = []
for pest in separated:
    random.shuffle(pest)
    split_idx = int(len(pest) * 0.8)
    train_files.append(pest[:split_idx])
    val_files.append(pest[split_idx:])
train_files = [item for sublist in train_files for item in sublist]
val_files = [item for sublist in val_files for item in sublist]

wrong input dir for src_lbl

# Helper to move image and label together
def copy_pair(files, dest_img_dir, dest_lbl_dir):
    for f in files:
        src_img = os.path.join(input_dir, f)
        src_lbl = os.path.join(input_dir, f.replace('.jpg', '.txt'))

        dst_img = os.path.join(dest_img_dir, f)
        dst_lbl = os.path.join(dest_lbl_dir, os.path.basename(src_lbl))

        print(src_lbl)
        if os.path.exists(src_lbl):
            print("Yes")
            #shutil.copy(src_img, dst_img)
            #shutil.copy(src_lbl, dst_lbl)

# Move files
copy_pair(train_files[:1], img_train, lbl_train)
copy_pair(val_files, img_val, lbl_val)

print(f"Training images: {len(train_files)}")
print(f"Validation images: {len(val_files)}")