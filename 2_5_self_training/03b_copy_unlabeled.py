import os
import shutil
import yaml



#Copying 2: Creating a YOLO training set

# Setup base folder
tiles_folder = "/user/christoph.wald/u15287/big-scratch/02_splitted_data/train_unlabeled/tiles"
training_folder = "/user/christoph.wald/u15287/big-scratch/04_SSL_training_data/training_data_new_more_tiles"

# Setup file structure for destination
file_types = ["images", "labels"]

# Copy all tiles and labels from train source folders


skipped = 0
empty_files = []

label_path = os.path.join(tiles_folder, "labels")
img_path = os.path.join(tiles_folder, "images")
label_dest_path = os.path.join(training_folder, "labels/train")
img_dest_path = os.path.join(training_folder, "images/train")
label_files = os.listdir(label_path)

# First pass: copy non-empty files and track empty files
for file in label_files:
    with open(os.path.join(label_path, file), "r") as f:
        if f.read().strip() == "":
            skipped += 1
            empty_files.append(file)
        else:
            img_file = os.path.splitext(file)[0] + ".jpg"
            img_src = os.path.join(img_path, img_file)
            img_dest = os.path.join(img_dest_path, img_file)

            # only copy if the image file isn’t already there
            if not os.path.exists(img_dest):
                shutil.copy2(img_src, img_dest)

            # copy label every time (might have changed)
            shutil.copy2(
                os.path.join(label_path, file),
                os.path.join(label_dest_path, file)
            )

total_files = len(label_files)
copied_non_empty = total_files - skipped
print(f"Copied {copied_non_empty} files from {total_files} files.")

