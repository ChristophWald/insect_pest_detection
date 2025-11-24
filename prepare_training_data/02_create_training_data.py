import os
import shutil
import yaml

'''
create a training folder for YOLO training by copying files and saving a data.yaml
'''


# Setup base folder
tiles_folder = "/user/christoph.wald/u15287/big-scratch/02_splitted_data/train_labeled/SSL/tiles_mininside08"
#tiles_folder = "/user/christoph.wald/u15287/big-scratch/02_splitted_data/train_unlabeled/tiles_mininside08_masked"
val_tiles_folder = "/user/christoph.wald/u15287/big-scratch/02_splitted_data/train_labeled/tiles"
training_folder = "/user/christoph.wald/u15287/big-scratch/04_SSL_training_data/training_data_mininside08"

# Setup file structure for destination
file_types = ["images", "labels"]
use_types = ["train", "val"]


#create folders and data.yaml first time

for f in file_types:
    for u in use_types:
        path = os.path.join(training_folder, f, u)
        os.makedirs(path, exist_ok=True)

data = {
    "train": os.path.join(training_folder, "images/train"),
    "val": os.path.join(training_folder, "images/val"),
    "nc": 4,
    "names": {
        0: "FungusGnats",
        1: "LeafMinerFlies",
        2: "Thrips",
        3: "WhiteFlies"
        }
}

with open(os.path.join(training_folder,"data.yaml"), "w") as f:
    yaml.dump(data, f, sort_keys=False)




# Copy all tiles and labels from train/val source folders
for u in use_types:

    if u == "train":
        print(f"Copying {u} data.")
        skipped = 0
        empty_files = []

        label_path = os.path.join(tiles_folder, "labels")
        img_path = os.path.join(tiles_folder, "images")
        label_dest_path = os.path.join(training_folder, "labels", u)
        img_dest_path = os.path.join(training_folder, "images", u)
        label_files = os.listdir(label_path)

        
        for file in label_files:
            label_src = os.path.join(label_path, file)
            with open(label_src, "r") as f:
                label_content = f.read().strip()

            is_empty = (label_content == "")
            img_file = os.path.splitext(file)[0] + ".jpg"
            img_src = os.path.join(img_path, img_file)
            img_dest = os.path.join(img_dest_path, img_file)

            # --- Behavior change here ---
            # Skip empty labels only for TRAIN
            if u == "train" and is_empty:
                skipped += 1
                empty_files.append(file)
                continue  # skip this one

            # Copy image if not already there
            if not os.path.exists(img_dest):
                shutil.copy2(img_src, img_dest)

            # Copy label file (always for val, only non-empty for train)
            shutil.copy2(label_src, os.path.join(label_dest_path, file))

        total_files = len(label_files)
        copied_non_empty = total_files - skipped
        print(f"Copied {copied_non_empty} label files from {total_files} total.")
        if skipped > 0:
            print(f"Skipped {skipped} empty label files ({u}).")

    if u == "val":
        print(f"Copying {u} data.")
        label_path = os.path.join(val_tiles_folder, "labels", u)
        img_path = os.path.join(val_tiles_folder, "images", u)
        label_dest_path = os.path.join(training_folder, "labels", u)
        img_dest_path = os.path.join(training_folder, "images", u)
        label_files = os.listdir(label_path)

        for file in label_files:
            label_src = os.path.join(label_path, file)
            with open(label_src, "r") as f:
                label_content = f.read().strip()

            img_file = os.path.splitext(file)[0] + ".jpg"
            img_src = os.path.join(img_path, img_file)
            img_dest = os.path.join(img_dest_path, img_file)

            # Copy image if not already there
            if not os.path.exists(img_dest):
                shutil.copy2(img_src, img_dest)

            # Copy label file (always for val, only non-empty for train)
            shutil.copy2(label_src, os.path.join(label_dest_path, file))

        total_files = len(label_files)
        
        print(f"Copied {total_files} label files.")

#copies empty background files
src_paths = ["/user/christoph.wald/u15287/big-scratch/02_splitted_data/empty_tiles/images", 
              "/user/christoph.wald/u15287/big-scratch/02_splitted_data/empty_tiles/images"]
dest_paths = ["images/train", "labels/train"]

for src, dest in zip(src_paths, dest_paths):
    files = os.listdir(src)
    for f in files:
        src_path = os.path.join(src, f)
        dest_path = os.path.join(training_folder, dest, f)
        shutil.copy2(src_path, dest_path)