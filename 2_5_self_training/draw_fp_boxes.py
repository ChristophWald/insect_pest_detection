import os
import cv2
import ast
import csv
import math
import glob
import numpy as np

# Directories
train_dir = "/user/christoph.wald/u15287/big-scratch/02_splitted_data/train_labeled/SSL/tiles/train/images"
val_dir   = "/user/christoph.wald/u15287/big-scratch/02_splitted_data/train_labeled/SSL/tiles/val/images"
metrics_dir = "/user/christoph.wald/u15287/insect_pest_detection/2_5_self_training/metrics"

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
