import os
import shutil
import random  # <--- import random

##########
#set destination folder
#set label source folder
#set percentage of empty files
##############


# Setup base folder
source_folder = "/user/christoph.wald/u15287/big-scratch/02_splitted_data/train_labeled/SSL/tiles"
dest_folder = "/user/christoph.wald/u15287/big-scratch/04_SSL_training_data/t0807_corrected_10background"

# Setup file structure for destination
file_types = ["images", "labels"]
use_types = ["train", "val"]

for f in file_types:
    for u in use_types:
        path = os.path.join(dest_folder, f, u)
        os.makedirs(path, exist_ok=True)

# Percentage of empty files to allow
empty_file_percentage = 10  # e.g., 25% of total files can be empty

# Copy all tiles and labels from train/val source folders
for u in use_types:

    print(f"Copying {u} data.")
    skipped = 0
    empty_files = []

    label_path = os.path.join(source_folder, u, "labels_t0807_corrected")
    img_path = os.path.join(source_folder, u, "images")
    label_dest_path = os.path.join(dest_folder, "labels", u)
    img_dest_path = os.path.join(dest_folder, "images", u)
    label_files = os.listdir(label_path)

    # First pass: copy non-empty files and track empty files
    for file in label_files:
        with open(os.path.join(label_path, file), "r") as f:
            if f.read().strip() == "":
                skipped += 1
                empty_files.append(file)
            else:
                img_file = os.path.splitext(file)[0] + ".jpg"
                shutil.copy2(os.path.join(img_path, img_file), os.path.join(img_dest_path, img_file))
                shutil.copy2(os.path.join(label_path, file), os.path.join(label_dest_path, file))

    total_files = len(label_files)
    copied_non_empty = total_files - skipped

    # Calculate number of empty files allowed
    max_empty_allowed = int(total_files * empty_file_percentage / 100)
    empty_to_copy = min(max_empty_allowed, skipped)

    # Second pass: copy allowed empty files randomly
    copied_empty = 0
    random_empty_files = random.sample(empty_files, empty_to_copy)  # <-- random selection
    for file in random_empty_files:
        img_file = os.path.splitext(file)[0] + ".jpg"
        shutil.copy2(os.path.join(img_path, img_file), os.path.join(img_dest_path, img_file))
        shutil.copy2(os.path.join(label_path, file), os.path.join(label_dest_path, file))
        copied_empty += 1

    total_copied = copied_non_empty + copied_empty
    print(f"Copied {total_copied} files from {total_files} files "
          f"({total_copied/total_files*100:.2f} % used, including {copied_empty} empty files).")
