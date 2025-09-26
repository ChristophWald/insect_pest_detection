import os
import json

# --- Settings ---
folder_paths = [
    "/user/christoph.wald/u15287/big-scratch/02_splitted_data/train_labeled/tiles/train/labels",
    "/user/christoph.wald/u15287/big-scratch/02_splitted_data/train_labeled/tiles/val/labels"
]
json_path = "/user/christoph.wald/u15287/insect_pest_detection/2_5_self_training/predictions/predictions.json"
thresholds = [0.5, 0.6, 0.7]

species = ["BRAIIM", "LIRIBO", "FRANOC", "TRIAVA"]

# --- Load predictions ---
with open(json_path, "r") as f:
    data = json.load(f)

for threshold in thresholds:
    print(f"\nThreshold {threshold}")
    total_incorrect = 0
    for folder_path in folder_paths:
        print(f"Checking folder: {folder_path}")

        for filename in os.listdir(folder_path):
            if not filename.endswith(".txt"):
                continue

            in_path = os.path.join(folder_path, filename)

            # Extract base image name and tile id
            parts = filename.split("_tile_")
            base_name = parts[0] + ".jpg"
            tile_id = int(parts[1].split(".")[0])

            if base_name not in data:
                continue

            # Determine ground truth class from filename
            gt_class_str = next((s for s in species if filename.startswith(s)), None)
            if gt_class_str is None:
                continue  # skip if species not found in list
            gt_class_id = species.index(gt_class_str)

            # Check predictions
            for entry in data[base_name]:
                if entry['tile_id'] == tile_id and entry['prediction'][5] >= threshold:
                    pred_class_id = entry['prediction'][0]
                    if pred_class_id != gt_class_id:
                        total_incorrect += 1

    print(f"  Total incorrect class_ids: {total_incorrect}")
