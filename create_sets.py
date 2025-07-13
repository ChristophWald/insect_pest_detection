import json
import os
import shutil

# ==== CONFIGURATION ====
json_dir = 'split_info'  # Directory containing multiple JSON files
dataset_images_root = '/user/christoph.wald/u15287/big-scratch/dataset/images'
dataset_labels_root = '/user/christoph.wald/u15287/big-scratch/dataset/labels'
base_output_dir = "/user/christoph.wald/u15287/big-scratch/splitted_data"
image_ext = '.jpg'
label_ext = '.txt'
# ========================

# List all JSON files in the directory
json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]

for json_file in json_files:
    json_path = os.path.join(json_dir, json_file)

    # Get folder name from JSON filename (no extension)
    base_name = os.path.splitext(json_file)[0]
    output_dir = os.path.join(base_output_dir, base_name)
    images_out_dir = os.path.join(output_dir, 'images')
    labels_out_dir = os.path.join(output_dir, 'labels')

    # Create directories
    os.makedirs(images_out_dir, exist_ok=True)
    os.makedirs(labels_out_dir, exist_ok=True)

    # Load JSON file
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Iterate over each class/sub-dict
    for class_name, items in data.items():
        for key in items.keys():
            # Build source paths based on class name
            src_img = os.path.join(dataset_images_root, class_name, key + image_ext)
            src_label = os.path.join(dataset_labels_root, class_name, key + label_ext)

            # Build destination paths
            dst_img = os.path.join(images_out_dir, key + image_ext)
            dst_label = os.path.join(labels_out_dir, key + label_ext)

            # Copy image
            if os.path.exists(src_img):
                shutil.copy2(src_img, dst_img)
            else:
                print(f"[WARNING] Image not found: {src_img}")

            # Copy label
            if os.path.exists(src_label):
                shutil.copy2(src_label, dst_label)
            else:
                print(f"[WARNING] Label not found: {src_label}")
