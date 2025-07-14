import cv2
import os
import math
import numpy as np

from modules import get_files_by_subfolder

def load_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    return image

def pad_to_multiple(image, tile_size=640, pad_value=(114,114,114)):
    h, w = image.shape[:2]
    pad_w = math.ceil(w / tile_size) * tile_size - w
    pad_h = math.ceil(h / tile_size) * tile_size - h
    padded = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=pad_value)
    return padded, w, h  # return original width/height for label conversion

def load_labels(label_path):
    labels = []
    if os.path.isfile(label_path):
        with open(label_path, 'r') as f:
            for line in f.read().strip().split('\n'):
                if line:
                    cls, x_c, y_c, w, h = line.split()
                    labels.append([int(cls), float(x_c), float(y_c), float(w), float(h)])
    return labels

def box_absolute_coords(box, img_w, img_h):
    """Convert normalized YOLO box to absolute x1,y1,x2,y2 coords"""
    cls, x_c, y_c, w, h = box
    x1 = (x_c - w/2) * img_w
    y1 = (y_c - h/2) * img_h
    x2 = (x_c + w/2) * img_w
    y2 = (y_c + h/2) * img_h
    return cls, x1, y1, x2, y2

def box_intersection(boxA, boxB):
    """Calculate intersection area of two boxes: box = (x1,y1,x2,y2)"""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    return inter_w * inter_h

def tile_and_save(image_path, label_path, dest_path,
                  tile_size=640, stride=440, min_inside_ratio=0.8):
    # Load image and labels
    image = load_image(image_path)
    labels = load_labels(label_path)
    
    # Pad image
    padded_img, orig_w, orig_h = pad_to_multiple(image, tile_size=tile_size)
    p_h, p_w = padded_img.shape[:2]

    # Create output dirs
    images_out = os.path.join(dest_path, "images")
    labels_out = os.path.join(dest_path, "labels")
    os.makedirs(images_out, exist_ok=True)
    os.makedirs(labels_out, exist_ok=True)

    # Convert labels to absolute coords (original image size)
    abs_boxes = []
    for box in labels:
        cls, x1, y1, x2, y2 = box_absolute_coords(box, orig_w, orig_h)
        abs_boxes.append([cls, x1, y1, x2, y2])

    # Generate tiles by sliding window
    tile_id = 0
    for y in range(0, p_h - tile_size + 1, stride):
        for x in range(0, p_w - tile_size + 1, stride):
            tile = padded_img[y:y+tile_size, x:x+tile_size]
            
            # Find boxes with at least min_inside_ratio inside this tile
            tile_box = (x, y, x + tile_size, y + tile_size)
            tile_labels = []
            for (cls, bx1, by1, bx2, by2) in abs_boxes:
                # Compute intersection area
                inter_area = box_intersection(tile_box, (bx1, by1, bx2, by2))
                box_area = (bx2 - bx1) * (by2 - by1)
                if box_area == 0:
                    continue
                # Fraction of box inside tile
                inside_ratio = inter_area / box_area
                if inside_ratio >= min_inside_ratio:
                    # Clip box to tile boundaries
                    cx1 = max(bx1, x)
                    cy1 = max(by1, y)
                    cx2 = min(bx2, x + tile_size)
                    cy2 = min(by2, y + tile_size)

                    # Convert back to YOLO normalized format relative to tile
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

            # Save tile labels (empty file if no labels)
            label_filename = f"{os.path.splitext(os.path.basename(image_path))[0]}_tile_{tile_id}.txt"
            label_path_out = os.path.join(labels_out, label_filename)
            with open(label_path_out, 'w') as f:
                for lbl in tile_labels:
                    f.write(f"{lbl[0]} {lbl[1]:.6f} {lbl[2]:.6f} {lbl[3]:.6f} {lbl[4]:.6f}\n")

            tile_id += 1

    print(f"Tiling complete. {tile_id} tiles saved to {dest_path}")

'''
image_path = "/user/christoph.wald/u15287/big-scratch/dataset/images/FungusGnats/BRAIIM_0001.jpg"
label_path = "/user/christoph.wald/u15287/big-scratch/dataset/labels/FungusGnats/BRAIIM_0001.txt"
dest_path = "./"
'''

#set path
path = "/user/christoph.wald/u15287/big-scratch/splitted_data/test_set/"

#execute
dest_path = os.path.join(path, "tiles" )
os.makedirs(dest_path, exist_ok=True)
dest_path = os.path.join(path, "tiles" )
files = get_files_by_subfolder(path)
for key in files["images"]:
    print(f"Image: {os.path.join(path, key+'.jpg')}, Label: {os.path.join(path, key + '.txt')}")
    img = os.path.join(path, "images", key+'.jpg')
    label = os.path.join(path, "labels", key + '.txt')
    tile_and_save(img, label, dest_path)