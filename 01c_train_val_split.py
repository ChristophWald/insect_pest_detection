import os
import random
import shutil

random.seed(43)

'''
splits the labeld files into a training/validation set
'''

def copy_pair(files, dest_img_dir, dest_lbl_dir):
    '''
    copies images and label files 
    '''
    for f in files:
        src_img = os.path.join(os.path.join(input_dir, "images"), f)
        src_lbl = os.path.join(os.path.join(input_dir, "labels"), f.replace('.jpg', '.txt'))

        dst_img = os.path.join(dest_img_dir, f)
        dst_lbl = os.path.join(dest_lbl_dir, os.path.basename(src_lbl))

        if os.path.exists(src_lbl):
            shutil.copy(src_img, dst_img)
            shutil.copy(src_lbl, dst_lbl)

input_dir = "/user/christoph.wald/u15287/big-scratch/splitted_data/train_labeled/"
output_dir = "/user/christoph.wald/u15287/big-scratch/splitted_data/train_labeled/split/"

# Output folders
img_train = os.path.join(output_dir, 'images/train')
img_val = os.path.join(output_dir, 'images/val')
lbl_train = os.path.join(output_dir, 'labels/train')
lbl_val = os.path.join(output_dir, 'labels/val')
for d in [img_train, img_val, lbl_train, lbl_val]:
    os.makedirs(d, exist_ok=True)

pest_types = ["BRAIIM", "FRANOC", "LIRIBO", "TRIAVA"]
image_files = [f for f in os.listdir(os.path.join(input_dir, "images")) if f.endswith(('.jpg', '.png'))]
separated = []
for pest in pest_types:
    separated.append([f for f in image_files if pest in f])

train_files = []
val_files = []
for pest in separated:
    random.shuffle(pest)
    split_idx = int(len(pest) * 0.8)
    train_files.append(pest[:split_idx])
    val_files.append(pest[split_idx:])
train_files = [item for sublist in train_files for item in sublist]
val_files = [item for sublist in val_files for item in sublist]

# Move files
copy_pair(train_files, img_train, lbl_train)
copy_pair(val_files, img_val, lbl_val)

print(f"Training images: {len(train_files)}")
print(f"Validation images: {len(val_files)}")