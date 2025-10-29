import sys
sys.path.append("/user/christoph.wald/u15287/insect_pest_detection/modules")
import cv2
import os
import math
import numpy as np
from modules import load_yolo_labels, compute_intersection_area
import ast
import random

def pad_to_multiple(image, tile_size=640, pad_value=(114, 114, 114)):
    h, w = image.shape[:2]
    pad_w = math.ceil(w / tile_size) * tile_size - w
    pad_h = math.ceil(h / tile_size) * tile_size - h
    padded = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=pad_value)
    return padded, w, h  # return original width/height for label conversion


def tile_and_save(
    image_path, label_path, dest_base_path, split,
    tile_size=640, stride=440, min_inside_ratio=0.4,
    yolo=True, keep_empty_prob=0.1
):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not read image: {image_path}")
        return

    # Load labels
    if yolo:
        boxes, classes = load_yolo_labels(label_path, image.shape[1], image.shape[0])
        abs_boxes = [[classes[i], *box] for i, box in enumerate(boxes)]
    else:
        abs_boxes = []
        with open(label_path, 'r') as f:
            for line in f:
                if line.strip():
                    cls, x1, y1, x2, y2 = map(float, ast.literal_eval(line))
                    abs_boxes.append([int(cls), x1, y1, x2, y2])

    # Pad image to a multiple of tile size
    padded_img, orig_w, orig_h = pad_to_multiple(image, tile_size=tile_size)
    p_h, p_w = padded_img.shape[:2]

    # Output structure:
    # tiles_improved/images/train/
    # tiles_improved/labels/train/
    images_out = os.path.join(dest_base_path, "images", split)
    labels_out = os.path.join(dest_base_path, "labels", split)
    os.makedirs(images_out, exist_ok=True)
    os.makedirs(labels_out, exist_ok=True)

    # Generate all tiles
    tiles = []
    for y in range(0, p_h - tile_size + 1, stride):
        for x in range(0, p_w - tile_size + 1, stride):
            tiles.append((x, y, x + tile_size, y + tile_size))

    # Assign each box to its best tile (highest overlap)
    tile_assignments = [[] for _ in range(len(tiles))]
    for (cls, bx1, by1, bx2, by2) in abs_boxes:
        best_ratio = 0
        best_tile = None
        for i, (tx1, ty1, tx2, ty2) in enumerate(tiles):
            inter_area = compute_intersection_area((tx1, ty1, tx2, ty2), (bx1, by1, bx2, by2))
            box_area = (bx2 - bx1) * (by2 - by1)
            if box_area == 0:
                continue
            inside_ratio = inter_area / box_area
            if inside_ratio > best_ratio:
                best_ratio = inside_ratio
                best_tile = i
        if best_ratio >= min_inside_ratio and best_tile is not None:
            tile_assignments[best_tile].append((cls, bx1, by1, bx2, by2))

    # Write tiles
    tile_id = 0
    for i, (tx1, ty1, tx2, ty2) in enumerate(tiles):
        tile_box = (tx1, ty1, tx2, ty2)
        tile_img = padded_img[ty1:ty2, tx1:tx2]
        assigned_boxes = tile_assignments[i]

        # Check if any *unassigned* objects overlap this tile → drop if so
        partial_overlap = False
        for (cls, bx1, by1, bx2, by2) in abs_boxes:
            inter_area = compute_intersection_area(tile_box, (bx1, by1, bx2, by2))
            if inter_area > 0 and (cls, bx1, by1, bx2, by2) not in assigned_boxes:
                partial_overlap = True
                break

        # Skip tiles with partial unlabeled objects
        if partial_overlap:
            continue

        # Convert assigned boxes to YOLO format
        tile_labels = []
        for (cls, bx1, by1, bx2, by2) in assigned_boxes:
            cx1 = max(bx1, tx1)
            cy1 = max(by1, ty1)
            cx2 = min(bx2, tx2)
            cy2 = min(by2, ty2)
            box_w = cx2 - cx1
            box_h = cy2 - cy1
            box_xc = cx1 + box_w / 2
            box_yc = cy1 + box_h / 2

            nx_c = (box_xc - tx1) / tile_size
            ny_c = (box_yc - ty1) / tile_size
            nw = box_w / tile_size
            nh = box_h / tile_size
            tile_labels.append([cls, nx_c, ny_c, nw, nh])

        # Decide whether to save tile
        if tile_labels or random.random() < keep_empty_prob:
            tile_filename = f"{os.path.splitext(os.path.basename(image_path))[0]}_tile_{tile_id}.jpg"
            label_filename = f"{os.path.splitext(os.path.basename(image_path))[0]}_tile_{tile_id}.txt"

            cv2.imwrite(os.path.join(images_out, tile_filename), tile_img)
            with open(os.path.join(labels_out, label_filename), "w") as f:
                for lbl in tile_labels:
                    f.write(f"{lbl[0]} {lbl[1]:.6f} {lbl[2]:.6f} {lbl[3]:.6f} {lbl[4]:.6f}\n")

            tile_id += 1

    print(f"Tiling complete for {os.path.basename(image_path)} — {tile_id} tiles saved.")


# === MAIN EXECUTION ===
base_path = "/user/christoph.wald/u15287/big-scratch/02_splitted_data/train_labeled"

# Set up source image paths
img_paths = {
    "train": os.path.join(base_path, "split/images/train"),
    "val": os.path.join(base_path, "split/images/val")
}

# Base output directory
dest_base = os.path.join(base_path, "tiles_improved")

for split, img_path in img_paths.items():
    print(f"\nProcessing split: {split}, path: {img_path}")
    os.makedirs(os.path.join(dest_base, "images", split), exist_ok=True)
    os.makedirs(os.path.join(dest_base, "labels", split), exist_ok=True)

    # Loop over image files
    files = os.listdir(img_path)
    for file in files:
        if not file.lower().endswith((".jpg", ".png", ".jpeg")):
            continue
        img = os.path.join(img_path, file)

        # Infer label path
        label_path = img.replace("images", "labels")
        label = os.path.splitext(label_path)[0] + '.txt'

        # Call tile_and_save with the unified base dest and current split
        tile_and_save(img, label, dest_base, split, yolo=True)
