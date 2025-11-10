import os
import cv2
import ast
import csv
import math
import glob
import numpy as np

# Directories
train_dir = "/user/christoph.wald/u15287/big-scratch/02_splitted_data/train_labeled/SSL/04_images_cropped"
val_dir   = "/user/christoph.wald/u15287/big-scratch/02_splitted_data/train_labeled/SSL/04_images_cropped"
metrics_dir = "/user/christoph.wald/u15287/insect_pest_detection/training/metrics"

import os
import json
import cv2
import numpy as np
from math import ceil

def visualize_fp_boxes_global_grid(json_file, image_dir, output_dir, conf_range=(0.0, 1.0), grid_dim=(10, 10)):
    """
    Extract FP boxes from JSON across all images and save as 10x10 grids of crops.
    Saves each grid immediately after filling it to avoid high memory usage.
    """
    os.makedirs(output_dir, exist_ok=True)

    with open(json_file, "r") as f:
        data = json.load(f)

    fp_data = data.get("FP", {})
    rows, cols = grid_dim
    grid_size = rows * cols
    batch_crops = []
    max_h, max_w = 0, 0
    grid_count = 0

    for species, images in fp_data.items():
        for img_name, entries in images.items():
            boxes_to_draw = [
                entry for entry in entries
                if len(entry.get("prediction", [])) >= 6 and
                   conf_range[0] <= entry["prediction"][-1] <= conf_range[1]
            ]
            if not boxes_to_draw:
                continue

            img_path = os.path.join(image_dir, img_name)
            if not os.path.exists(img_path):
                print(f"[WARN] Image not found: {img_name}")
                continue

            img = cv2.imread(img_path)
            if img is None:
                print(f"[ERROR] Failed to read {img_path}")
                continue

            for entry in boxes_to_draw:
                _, x1, y1, x2, y2, conf = entry["prediction"]
                crop = img[int(y1):int(y2), int(x1):int(x2)]
                if crop.size == 0:
                    continue
                batch_crops.append(crop)
                max_h = max(max_h, crop.shape[0])
                max_w = max(max_w, crop.shape[1])

                # When batch reaches grid size, save and reset
                if len(batch_crops) == grid_size:
                    grid_img = np.ones((rows * max_h, cols * max_w, 3), dtype=np.uint8) * 255
                    for idx, c in enumerate(batch_crops):
                        r = idx // cols
                        c_idx = idx % cols
                        h, w = c.shape[:2]
                        grid_img[r*max_h:r*max_h+h, c_idx*max_w:c_idx*max_w+w] = c

                    grid_count += 1
                    out_path = os.path.join(output_dir, f"fp_grid_{grid_count}.jpg")
                    cv2.imwrite(out_path, grid_img)
                    print(f"Saved grid: {out_path}")

                    # Reset batch
                    batch_crops = []
                    max_h, max_w = 0, 0

    # Save remaining crops if any
    if batch_crops:
        grid_img = np.ones((rows * max_h, cols * max_w, 3), dtype=np.uint8) * 255
        for idx, c in enumerate(batch_crops):
            r = idx // cols
            c_idx = idx % cols
            h, w = c.shape[:2]
            grid_img[r*max_h:r*max_h+h, c_idx*max_w:c_idx*max_w+w] = c

        grid_count += 1
        out_path = os.path.join(output_dir, f"fp_grid_{grid_count}.jpg")
        cv2.imwrite(out_path, grid_img)
        print(f"Saved final grid with {len(batch_crops)} crops: {out_path}")


# Usage example
json_file = "/user/christoph.wald/u15287/insect_pest_detection/training/predictions/predictions_fullimage_4.json"
image_dir = "/user/christoph.wald/u15287/big-scratch/02_splitted_data/train_labeled/SSL/04_images_cropped"



output_dir = "/user/christoph.wald/u15287/big-scratch/test_crops_grids_09"
visualize_fp_boxes_global_grid(json_file, image_dir, output_dir, conf_range=(0.9, 1.0), grid_dim=(10, 10))
output_dir = "/user/christoph.wald/u15287/big-scratch/test_crops_grids_08"
visualize_fp_boxes_global_grid(json_file, image_dir, output_dir, conf_range=(0.8, 0.9), grid_dim=(10, 10))
output_dir = "/user/christoph.wald/u15287/big-scratch/test_crops_grids_07"
visualize_fp_boxes_global_grid(json_file, image_dir, output_dir, conf_range=(0.7, 0.8), grid_dim=(10, 10))
output_dir = "/user/christoph.wald/u15287/big-scratch/test_crops_grids_06"
visualize_fp_boxes_global_grid(json_file, image_dir, output_dir, conf_range=(0.6, 0.7), grid_dim=(10, 10))
output_dir = "/user/christoph.wald/u15287/big-scratch/test_crops_grids_05"
visualize_fp_boxes_global_grid(json_file, image_dir, output_dir, conf_range=(0.5, 0.6), grid_dim=(10, 10))

output_dir = "/user/christoph.wald/u15287/big-scratch/test_crops_grids_04"
visualize_fp_boxes_global_grid(json_file, image_dir, output_dir, conf_range=(0.4, 0.5), grid_dim=(10, 10))

output_dir = "/user/christoph.wald/u15287/big-scratch/test_crops_grids_03"
visualize_fp_boxes_global_grid(json_file, image_dir, output_dir, conf_range=(0.3, 0.4), grid_dim=(10, 10))
output_dir = "/user/christoph.wald/u15287/big-scratch/test_crops_grids_02"
visualize_fp_boxes_global_grid(json_file, image_dir, output_dir, conf_range=(0.2, 0.3), grid_dim=(10, 10))
output_dir = "/user/christoph.wald/u15287/big-scratch/test_crops_grids_01"
visualize_fp_boxes_global_grid(json_file, image_dir, output_dir, conf_range=(0.1, 0.2), grid_dim=(10, 10))




'''
# Grid size
grid_rows = 10
grid_cols = 10
tile_size = 640  # assume all tiles are square and same size

def draw_prediction(img, abs_box, rel_box, conf):
    h, w = img.shape[:2]

    # Absolute box (green)
    x1, y1, x2, y2 = map(int, abs_box)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(img, f"{conf:.3f}", (x1 + 3, y1 + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Relative box (blue)
    xc, yc, bw, bh = rel_box
    x1_r = int((xc - bw/2) * w)
    y1_r = int((yc - bh/2) * h)
    x2_r = int((xc + bw/2) * w)
    y2_r = int((yc + bh/2) * h)
    cv2.rectangle(img, (x1_r, y1_r), (x2_r, y2_r), (255, 0, 0), 2)
    cv2.putText(img, f"{conf:.3f}", (x1_r + 3, y1_r + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    return img

# Process all files starting with fp_labels_added_run
for input_file in glob.glob(os.path.join(metrics_dir, "fp_labels_added_run*.txt")):
    run_number = os.path.splitext(os.path.basename(input_file))[0].split("run")[-1]
    out_dir = os.path.join(metrics_dir, f"pred_boxes{run_number}")
    os.makedirs(out_dir, exist_ok=True)

    # Collect all tile images with predictions
    tile_images = []
    with open(input_file, "r") as f:
        reader = csv.reader(f)
        for line_num, parts in enumerate(reader):
            if line_num == 0 and "conf" in parts[4].lower():
                continue

            base_name = parts[1]
            tile_idx  = parts[2]
            conf      = float(parts[4])
            abs_box   = ast.literal_eval(",".join(parts[5:9]))
            rel_box   = ast.literal_eval(",".join(parts[9:13]))

            tile_file = f"{os.path.splitext(base_name)[0]}_tile_{tile_idx}.jpg"
            img_path = os.path.join(train_dir, tile_file)
            if not os.path.exists(img_path):
                img_path = os.path.join(val_dir, tile_file)
            if not os.path.exists(img_path):
                print(f"[WARN] File not found: {tile_file}")
                continue

            img = cv2.imread(img_path)
            if img is None:
                print(f"[ERROR] Failed to read {img_path}")
                continue

            img = draw_prediction(img, abs_box, rel_box, conf)
            tile_images.append(img)

    # Create grids of 10x10
    grid_count = math.ceil(len(tile_images) / (grid_rows * grid_cols))
    for g in range(grid_count):
        grid_tiles = tile_images[g * grid_rows * grid_cols : (g+1) * grid_rows * grid_cols]
        # Pad with black images if not enough
        while len(grid_tiles) < grid_rows * grid_cols:
            grid_tiles.append(np.zeros((tile_size, tile_size, 3), dtype=np.uint8))

        # Stack into 10x10 grid
        rows = []
        for r in range(grid_rows):
            row_imgs = grid_tiles[r*grid_cols:(r+1)*grid_cols]
            row = np.hstack(row_imgs)
            rows.append(row)
        grid_img = np.vstack(rows)

        # Save grid
        grid_file = os.path.join(out_dir, f"grid_{g+1}.png")
        cv2.imwrite(grid_file, grid_img)
        print(f"Saved grid: {grid_file}")
'''