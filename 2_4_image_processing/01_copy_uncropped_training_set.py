import json
import os
import shutil

# Labeled training set
with open("/user/christoph.wald/u15287/big-scratch/02_splitted_data/split_info/train_labeled.json", "r") as f:
    data = json.load(f)

# remove "Thrips" class
data.pop("Thrips", None)

for key in data:
    for file in data[key]:
        # copy image
        src_img = os.path.join("/user/christoph.wald/u15287/big-scratch/00_uncropped_dataset/images", file + ".jpg")
        dest_img = os.path.join("/user/christoph.wald/u15287/big-scratch/02_splitted_data/train_labeled/images_uncropped", file + ".jpg")
        os.makedirs(os.path.dirname(dest_img), exist_ok=True)
        shutil.copy2(src_img, dest_img)

        # copy label
        src_lbl = os.path.join("/user/christoph.wald/u15287/big-scratch/00_uncropped_dataset/labels", file + ".txt")
        dest_lbl = os.path.join("/user/christoph.wald/u15287/big-scratch/02_splitted_data/train_labeled/labels_uncropped", file + ".txt")
        os.makedirs(os.path.dirname(dest_lbl), exist_ok=True)
        shutil.copy2(src_lbl, dest_lbl) 

# Unlabeled training set
with open("/user/christoph.wald/u15287/big-scratch/02_splitted_data/split_info/train_unlabeled.json", "r") as f:
    data = json.load(f)

data.pop("Thrips", None)

for key in data:
    for file in data[key]:
        src_img = os.path.join("/user/christoph.wald/u15287/big-scratch/00_uncropped_dataset/images", file + ".jpg")
        dest_img = os.path.join("/user/christoph.wald/u15287/big-scratch/02_splitted_data/train_unlabeled/images_uncropped", file + ".jpg")
        
        os.makedirs(os.path.dirname(dest_img), exist_ok=True)
        shutil.copy2(src_img, dest_img)
