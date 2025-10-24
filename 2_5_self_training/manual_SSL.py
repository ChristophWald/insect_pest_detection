import os
import random
import shutil

'''
From the train folder of the labeled train set, for each species image/label pairs 
are collected such that at least 2500 individuals per species are in the set.
For each selected base image, all its tile-level image/label pairs are copied.
Unselected base images and their corresponding tiles are also copied to a separate folder.
'''

# --- Configuration ---
label_select_folder = "/user/christoph.wald/u15287/big-scratch/02_splitted_data/train_labeled/supervised/split/labels/train"
image_select_folder = "/user/christoph.wald/u15287/big-scratch/02_splitted_data/train_labeled/supervised/split/images/train"
label_src_folder = "/user/christoph.wald/u15287/big-scratch/02_splitted_data/train_labeled/supervised/tiles/train/labels"
image_src_folder = "/user/christoph.wald/u15287/big-scratch/02_splitted_data/train_labeled/supervised/tiles/train/images"

dest_folder = "/user/christoph.wald/u15287/big-scratch/04_SSL_training_data/training_data_for1m"
unused_dest_folder = "/user/christoph.wald/u15287/big-scratch/02_splitted_data/train_labeled/manual_SSL/tiles"
os.makedirs(unused_dest_folder, exist_ok=True)
unused_full_dest_folder = "/user/christoph.wald/u15287/big-scratch/02_splitted_data/train_labeled/manual_SSL/images"
os.makedirs(unused_full_dest_folder, exist_ok=True)


prefixes = ["BRAIIM", "LIRIBO", "TRIAVA"]
line_limit = 1500

random.seed(43)  # reproducibility

# Prepare destination directories
for base in [dest_folder]:
    os.makedirs(os.path.join(base, "labels", "train"), exist_ok=True)
    os.makedirs(os.path.join(base, "images", "train"), exist_ok=True)

dest_labels = os.path.join(dest_folder, "labels", "train")
dest_images = os.path.join(dest_folder, "images", "train")

for prefix in prefixes:
    print(f"\n--- Processing {prefix} ---")

    # Collect .txt files starting with prefix from SPLIT folder (image-level)
    files = [
        os.path.join(label_select_folder, f)
        for f in os.listdir(label_select_folder)
        if f.endswith(".txt") and f.startswith(prefix)
    ]

    print(f"Found {len(files)} {prefix} base label files.")

    # Shuffle for random selection
    random.shuffle(files)

    selected_basenames = []
    total_lines = 0

    # Keep adding until total_lines > line_limit
    for f in files:
        with open(f, "r") as fh:
            line_count = sum(1 for _ in fh)
        total_lines += line_count
        selected_basenames.append(os.path.splitext(os.path.basename(f))[0])
        if total_lines > line_limit:
            break

    print(f"Selected {len(selected_basenames)} {prefix} base images with {total_lines} total lines.")

    # --- Determine unselected base images ---
    unselected_basenames = [
        os.path.splitext(os.path.basename(f))[0] for f in files
        if os.path.splitext(os.path.basename(f))[0] not in selected_basenames
    ]
    print(f"Unselected {len(unselected_basenames)} {prefix} base images.")

    
    # Copy all corresponding TILE files (labels + images)
    all_tile_labels = os.listdir(label_src_folder)

    # Selected
    copied_count_selected = 0
    for base_name in selected_basenames:
        matching_tiles = [f for f in all_tile_labels if f.startswith(base_name + "_tile_") and f.endswith(".txt")]
        for tile_file in matching_tiles:
            src_label = os.path.join(label_src_folder, tile_file)
            src_image = os.path.join(image_src_folder, os.path.splitext(tile_file)[0] + ".jpg")

            shutil.copy(src_label, dest_labels)
            if os.path.exists(src_image):
                shutil.copy(src_image, dest_images)
            else:
                print(f"⚠️ Missing image for {tile_file}")

            copied_count_selected += 1

    print(f"✅ Copied {copied_count_selected} tile-level label/image pairs for {prefix} (selected).")
'''    
    # Unselected
    copied_count_unselected = 0
    for base_name in unselected_basenames:
        matching_tiles = [f for f in all_tile_labels if f.startswith(base_name + "_tile_") and f.endswith(".txt")]
        for tile_file in matching_tiles:
            src_image = os.path.join(image_src_folder, os.path.splitext(tile_file)[0] + ".jpg")

            if os.path.exists(src_image):
                shutil.copy(src_image, unused_dest_folder)
            else:
                print(f"⚠️ Missing image for {tile_file}")

            copied_count_unselected += 1
    
    print(f"📁 Copied {copied_count_unselected} tile-level label/image pairs for {prefix} (unselected).")
 
    copied_full_unselected = 0
    for base_name in unselected_basenames:
        src_full_image = os.path.join(image_select_folder, base_name + ".jpg")
        if os.path.exists(src_full_image):
            shutil.copy(src_full_image, unused_full_dest_folder)
            copied_full_unselected += 1
        else:
            print(f"⚠️ Missing full image for {base_name}.jpg")

print("\n✅ Done — both selected and unselected sets copied successfully.")
'''