import json
import os
import shutil

with open("/user/christoph.wald/u15287/big-scratch/02_splitted_data/split_info/train_labeled.json", "r") as f:
    data = json.load(f)

del data["Thrips"]

for key in data:
    for file in data[key]:
        src_path = os.path.join("/user/christoph.wald/u15287/big-scratch/00_uncropped_dataset/images", file +".jpg")
        dest_path = os.path.join("big-scratch/02_splitted_data/train_labeled/images_uncropped", file +".jpg")
        shutil.copy2(src_path, dest_path)
        

with open("/user/christoph.wald/u15287/big-scratch/02_splitted_data/split_info/train_unlabeled.json", "r") as f:
    data = json.load(f)

del data["Thrips"]

for key in data:
    for file in data[key]:
        src_path = os.path.join("/user/christoph.wald/u15287/big-scratch/00_uncropped_dataset/images", file +".jpg")
        dest_path = os.path.join("/user/christoph.wald/u15287/big-scratch/02_splitted_data/train_unlabeled/images_uncropped", file +".jpg")
        shutil.copy2(src_path, dest_path)
