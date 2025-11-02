### this is a working copy ###
### original from 2_2_supervised_training###
import sys
sys.path.append("/user/christoph.wald/u15287/insect_pest_detection/modules")
import cv2
import os
import math
import numpy as np
from modules import load_yolo_labels, compute_intersection_area
import ast

'''
creates 640x640 tiles
the path structure/file loading has to be revisited
'''

def pad_to_multiple(image, tile_size=640, pad_value=(114,114,114)):
    h, w = image.shape[:2]
    pad_w = math.ceil(w / tile_size) * tile_size - w
    pad_h = math.ceil(h / tile_size) * tile_size - h
    padded = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=pad_value)
    return padded, w, h  # return original width/height for label conversion



def tile_and_save(image_path, label_path, dest_path,
                  tile_size=640, stride=440, min_inside_ratio=0.8, yolo=True):
    import os
    import cv2
    import ast
    from modules import load_yolo_labels, compute_intersection_area

    image = cv2.imread(image_path)
    
    pest_types = ["BRAIIM", "LIRIBO", "FRANOC", "TRIAVA"]
    
    # Pad image
    padded_img, orig_w, orig_h = pad_to_multiple(image, tile_size=tile_size)
    p_h, p_w = padded_img.shape[:2]

    # Create output dirs
    images_out = os.path.join(dest_path, "images")
    labels_out = os.path.join(dest_path, "labels")
    os.makedirs(images_out, exist_ok=True)
    os.makedirs(labels_out, exist_ok=True)

    # Load labels if file exists
    abs_boxes = []
    if os.path.exists(label_path):
        if yolo:
            boxes, classes = load_yolo_labels(label_path, orig_w, orig_h)
            abs_boxes = [[classes[i], *box] for i, box in enumerate(boxes)]
        else:
            with open(label_path, 'r') as f:
                for line in f:
                    if line.strip():
                        try:
                            cls = [id in label_path for id in pest_types].index(True)
                        except ValueError:
                            cls = 0  # default class if not found
                        x, y, w, h = map(float, ast.literal_eval(line))
                        x1, y1, x2, y2 = x, y, x + w, y + h
                        abs_boxes.append([cls, x1, y1, x2, y2])
    else:
        print(f"[INFO] No label file for {os.path.basename(image_path)}. Tiles will have empty labels.")

    # Generate tiles
    tile_id = 0
    for y in range(0, p_h - tile_size + 1, stride):
        for x in range(0, p_w - tile_size + 1, stride):
            tile = padded_img[y:y+tile_size, x:x+tile_size]
            tile_box = (x, y, x + tile_size, y + tile_size)
            tile_labels = []

            for (cls, bx1, by1, bx2, by2) in abs_boxes:
                inter_area = compute_intersection_area(tile_box, (bx1, by1, bx2, by2))
                box_area = (bx2 - bx1) * (by2 - by1)
                if box_area == 0:
                    continue
                inside_ratio = inter_area / box_area
                if inside_ratio >= min_inside_ratio:
                    # Clip to tile
                    cx1 = max(bx1, x)
                    cy1 = max(by1, y)
                    cx2 = min(bx2, x + tile_size)
                    cy2 = min(by2, y + tile_size)

                    # YOLO normalized coords relative to tile
                    box_w = cx2 - cx1
                    box_h = cy2 - cy1
                    box_xc = cx1 + box_w / 2
                    box_yc = cy1 + box_h / 2
                    nx_c = (box_xc - x) / tile_size
                    ny_c = (box_yc - y) / tile_size
                    nw = box_w / tile_size
                    nh = box_h / tile_size

                    tile_labels.append([cls, nx_c, ny_c, nw, nh])

            # Save tile image
            tile_filename = f"{os.path.splitext(os.path.basename(image_path))[0]}_tile_{tile_id}.jpg"
            tile_path = os.path.join(images_out, tile_filename)
            cv2.imwrite(tile_path, tile)

            # Save tile labels (empty if none)
            label_filename = f"{os.path.splitext(os.path.basename(image_path))[0]}_tile_{tile_id}.txt"
            label_path_out = os.path.join(labels_out, label_filename)
            with open(label_path_out, 'w') as f:
                for lbl in tile_labels:
                    f.write(f"{lbl[0]} {lbl[1]:.6f} {lbl[2]:.6f} {lbl[3]:.6f} {lbl[4]:.6f}\n")

            tile_id += 1

    print(f"Tiling complete. {tile_id} tiles saved to {dest_path}")

def tile_and_save_improved(
    image_path, label_path, dest_path,
    tile_size=640, stride=440, min_inside_ratio=0.4,
    yolo=True, keep_empty_prob=1.0
):
 
    import random 
    image = cv2.imread(image_path)
    if image is None:
        print(f"[WARN] Could not read image: {image_path}")
        return

    # --- Load labels ---
    abs_boxes = []
    if os.path.exists(label_path):
        if yolo:
            boxes, classes = load_yolo_labels(label_path, image.shape[1], image.shape[0])
            abs_boxes = [[classes[i], *box] for i, box in enumerate(boxes)]
        else:
            with open(label_path, 'r') as f:
                for line in f:
                    if line.strip():
                        # Old format: [x, y, w, h] → absolute coords
                        vals = ast.literal_eval(line)
                        if len(vals) == 4:
                            x, y, w, h = map(float, vals)
                            cls = 0  # default class
                            x1, y1, x2, y2 = x, y, x + w, y + h
                            abs_boxes.append([cls, x1, y1, x2, y2])
                        elif len(vals) == 5:
                            cls, x1, y1, x2, y2 = map(float, vals)
                            abs_boxes.append([int(cls), x1, y1, x2, y2])
    else:
        print(f"[INFO] No label file found for {os.path.basename(image_path)}. Empty labels will be used.")

    # --- Pad image ---
    padded_img, orig_w, orig_h = pad_to_multiple(image, tile_size=tile_size)
    p_h, p_w = padded_img.shape[:2]

    # --- Create output dirs ---
    images_out = os.path.join(dest_path, "images")
    labels_out = os.path.join(dest_path, "labels")
    os.makedirs(images_out, exist_ok=True)
    os.makedirs(labels_out, exist_ok=True)

    # --- Generate tile positions ---
    tiles = [(x, y, x + tile_size, y + tile_size)
             for y in range(0, p_h - tile_size + 1, stride)
             for x in range(0, p_w - tile_size + 1, stride)]

    # --- Assign boxes to best tile ---
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

    # --- Write tiles ---
    tile_id = 0
    for i, (tx1, ty1, tx2, ty2) in enumerate(tiles):
        tile_box = (tx1, ty1, tx2, ty2)
        tile_img = padded_img[ty1:ty2, tx1:tx2]
        assigned_boxes = tile_assignments[i]

        # Skip tiles with partial unlabeled overlap
        partial_overlap = False
        for (cls, bx1, by1, bx2, by2) in abs_boxes:
            inter_area = compute_intersection_area(tile_box, (bx1, by1, bx2, by2))
            if inter_area > 0 and (cls, bx1, by1, bx2, by2) not in assigned_boxes:
                partial_overlap = True
                break
        if partial_overlap:
            continue

        # Convert boxes to YOLO format
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

        # Save tile if labeled or randomly keep empty
        if tile_labels or random.random() < keep_empty_prob:
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            tile_filename = f"{base_name}_tile_{tile_id}.jpg"
            label_filename = f"{base_name}_tile_{tile_id}.txt"

            cv2.imwrite(os.path.join(images_out, tile_filename), tile_img)
            with open(os.path.join(labels_out, label_filename), "w") as f:
                for lbl in tile_labels:
                    f.write(f"{lbl[0]} {lbl[1]:.6f} {lbl[2]:.6f} {lbl[3]:.6f} {lbl[4]:.6f}\n")

            tile_id += 1

    print(f"[DONE] {tile_id} tiles saved for {os.path.basename(image_path)} → {dest_path}")


'''

#for unlabeled images

img_path = "/user/christoph.wald/u15287/big-scratch/02_splitted_data/train_unlabeled/04_images_cropped"
label_path = "/user/christoph.wald/u15287/big-scratch/02_splitted_data/train_unlabeled/05_created_labels"
dest_path = "/user/christoph.wald/u15287/big-scratch/02_splitted_data/train_unlabeled/tiles_mininside08"
os.makedirs(dest_path, exist_ok=True)
files = os.listdir("/user/christoph.wald/u15287/big-scratch/02_splitted_data/train_unlabeled/03_images_masked")
for file in files:
    img = os.path.join(img_path, file)

    # Infer label path
    label = os.path.join(label_path, os.path.splitext(file)[0] + ".txt") 
    print(label)
    # Call tile_and_save with correct YOLO setting
    tile_and_save(img, label, dest_path, yolo=False)

#only train for labeled images

img_path = "/user/christoph.wald/u15287/big-scratch/02_splitted_data/train_labeled/SSL/04_images_cropped"
label_path = "/user/christoph.wald/u15287/big-scratch/02_splitted_data/train_labeled/SSL/05_created_labels"
dest_path = "/user/christoph.wald/u15287/big-scratch/02_splitted_data/train_labeled/SSL/tiles_mininside08"
os.makedirs(dest_path, exist_ok=True)
#to make sure only images from the train set are tiled
files = os.listdir("/user/christoph.wald/u15287/big-scratch/02_splitted_data/train_labeled/split/images/train")

with open("/user/christoph.wald/u15287/insect_pest_detection/2_4_image_processing/logs/images_not_processed_train_labeled.txt", "r") as f:
    not_processed = f.read().splitlines()


for file in files:
    if file.startswith("FRANOC"):
        continue
    if file in not_processed:
        print(f"Skipping unprocessed file {file}")
        continue
    img = os.path.join(img_path, file)

    # Infer label path
    label = os.path.join(label_path, os.path.splitext(file)[0] + ".txt") 
    print(label)
    # Call tile_and_save with correct YOLO setting
    tile_and_save(img, label, dest_path, yolo=False)
'''
#for augmented

img_path = "/user/christoph.wald/u15287/big-scratch/02_splitted_data/augmented_data/images"
label_path = "/user/christoph.wald/u15287/big-scratch/02_splitted_data/augmented_data/labels"
dest_path = "/user/christoph.wald/u15287/big-scratch/02_splitted_data/augmented_data/tiles"
os.makedirs(dest_path, exist_ok=True)
files = os.listdir(img_path)

for file in files:

    img = os.path.join(img_path, file)

    # Infer label path
    label = os.path.join(label_path, os.path.splitext(file)[0] + ".txt") 
    print(label)
    # Call tile_and_save with correct YOLO setting
    tile_and_save(img, label, dest_path, yolo=False)