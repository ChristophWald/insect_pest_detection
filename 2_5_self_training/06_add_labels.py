import os
import json
import shutil

#############
#set thresholds
#set output directory below
#set correction flag
#############

# --- Settings ---
folder_paths = [
    "/user/christoph.wald/u15287/big-scratch/02_splitted_data/train_labeled/SSL/tiles/train/labels",
    "/user/christoph.wald/u15287/big-scratch/02_splitted_data/train_labeled/SSL/tiles/val/labels"
]

# here I use the already thresholded (0.25) predictions right now
json_path = "/user/christoph.wald/u15287/insect_pest_detection/2_5_self_training/predictions/predictions.json"

# threshold depends on species prefix
thresholds = {
    "BRAIIM": 0.8,
    "LIRIBO": 0.8,
    "FRANOC": 0,
    "TRIAVA": 0.7,
}
species = list(thresholds.keys())

# --- Flag to choose behavior ---
# True: correct class ID according to filename
# False: skip incorrect class predictions
correct_class = True

# --- Load predictions ---
with open(json_path, "r") as f:
    data = json.load(f)

for folder_path in folder_paths:
    print(f"\nFolder {folder_path}")

    # --- Prepare output directory ---
    parent_dir = os.path.dirname(folder_path)
    out_dir = os.path.join(parent_dir, "labels_t0807_corrected")
    os.makedirs(out_dir, exist_ok=True)

    total_preds_appended = 0

    # --- Process files ---
    for filename in os.listdir(folder_path):

        in_path = os.path.join(folder_path, filename)
        out_path = os.path.join(out_dir, filename)

        # Extract base image name and tile id
        parts = filename.split("_tile_")
        base_name = parts[0] + ".jpg"
        tile_id = int(parts[1].split(".")[0])

        # Determine ground truth class from filename
        gt_class_str = next((s for s in species if filename.startswith(s)), None)
        gt_class_id = species.index(gt_class_str)

        # Pick threshold for this species
        threshold = thresholds[gt_class_str]

        # Default: copy file as-is
        shutil.copy(in_path, out_path)

        # Append predictions above threshold
        if base_name in data:
            for entry in data[base_name]:
                if entry['tile_id'] != tile_id:
                    continue
                pred = entry['prediction'][:5]
                conf = entry['prediction'][5]

                if conf < threshold:
                    continue

                pred_class_id = pred[0]

                if pred_class_id != gt_class_id:
                    if correct_class:
                        # Correct class according to filename
                        pred[0] = gt_class_id
                    else:
                        # Skip this prediction
                        continue

                # Append to file
                with open(out_path, "a") as f:
                    f.write(" ".join(map(str, pred)) + "\n")
                total_preds_appended += 1

    print(f"  Total predictions appended: {total_preds_appended}")



'''
import os
import json
import shutil

# --- Settings ---
folder_paths = [
    "/user/christoph.wald/u15287/big-scratch/02_splitted_data/train_labeled/tiles/train/labels",
    "/user/christoph.wald/u15287/big-scratch/02_splitted_data/train_labeled/tiles/val/labels"
]
json_path = "/user/christoph.wald/u15287/insect_pest_detection/2_5_self_training/predictions/predictions.json"
thresholds = [0.6, 0.7]  # confidence thresholds
species = ["BRAIIM", "LIRIBO", "FRANOC", "TRIAVA"]

# --- Flag to choose behavior ---
# True: correct class ID according to filename
# False: skip incorrect class predictions
correct_class = False

# --- Load predictions ---
with open(json_path, "r") as f:
    data = json.load(f)

for threshold in thresholds:
    print(f"\nThreshold {threshold}")
    for folder_path in folder_paths:
        print(f"Folder {folder_path}")

        # --- Prepare output directory ---
        parent_dir = os.path.dirname(folder_path)
        out_dir = os.path.join(parent_dir, f"label_threshold{int(threshold*10)}")
        os.makedirs(out_dir, exist_ok=True)

        total_preds_appended = 0

        # --- Process files ---
        for filename in os.listdir(folder_path):
            if not filename.endswith(".txt"):
                continue

            in_path = os.path.join(folder_path, filename)
            out_path = os.path.join(out_dir, filename)

            # Extract base image name and tile id
            parts = filename.split("_tile_")
            base_name = parts[0] + ".jpg"
            tile_id = int(parts[1].split(".")[0])

            # Determine ground truth class from filename
            gt_class_str = next((s for s in species if filename.startswith(s)), None)
            if gt_class_str is None:
                # skip file if species not recognized
                shutil.copy(in_path, out_path)
                continue
            gt_class_id = species.index(gt_class_str)

            # Default: copy file as-is
            shutil.copy(in_path, out_path)

            # Append predictions above threshold
            if base_name in data:
                for entry in data[base_name]:
                    if entry['tile_id'] != tile_id:
                        continue
                    pred = entry['prediction'][:5]
                    conf = entry['prediction'][5]

                    if conf < threshold:
                        continue

                    pred_class_id = pred[0]

                    if pred_class_id != gt_class_id:
                        if correct_class:
                            # Correct class according to filename
                            pred[0] = gt_class_id
                        else:
                            # Skip this prediction
                            continue

                    # Append to file
                    with open(out_path, "a") as f:
                        f.write(" ".join(map(str, pred)) + "\n")
                    total_preds_appended += 1

        print(f"  Total predictions appended: {total_preds_appended}")
'''