import os
import cv2
import math
import numpy as np
import random
import shutil

# ---------- Cropping ----------
def create_binary_mask(image):
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    return mask

def crop(image):
    mask = create_binary_mask(image)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        cropped_image = image[y:y+h, x:x+w]
        return cropped_image
    else:
        return None

# ---------- Padding + Tiling ----------
def pad_to_multiple(image, tile_size=640, pad_value=(114,114,114)):
    h, w = image.shape[:2]
    pad_w = math.ceil(w / tile_size) * tile_size - w
    pad_h = math.ceil(h / tile_size) * tile_size - h
    padded = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=pad_value)
    return padded

def tile_and_save(image, base_name, dest_img_dir, dest_lbl_dir, tile_size=640, stride=440):
    padded = pad_to_multiple(image, tile_size)
    h, w = padded.shape[:2]
    tile_id = 0

    for y in range(0, h - tile_size + 1, stride):
        for x in range(0, w - tile_size + 1, stride):
            tile = padded[y:y+tile_size, x:x+tile_size]
            tile_name = f"{base_name}_tile_{tile_id}.jpg"
            cv2.imwrite(os.path.join(dest_img_dir, tile_name), tile)
            with open(os.path.join(dest_lbl_dir, tile_name.replace('.jpg', '.txt')), 'w') as f:
                pass
            tile_id += 1
    return tile_id

# ---------- Full Pipeline ----------
def split_and_tile(
    images_dir,
    output_dir,
    tile_size=640,
    stride=440,
    train_ratio=0.8,
    seed=43,
    crop_images=True  # <--- New flag to toggle cropping
):  
    #will only work, if folders are not created yet
    files = os.listdir(images_dir)

    # Create output directories
    img_out = os.path.join(output_dir, 'images')
    lbl_out = os.path.join(output_dir, 'labels')
    for d in [img_out, lbl_out]:
        os.makedirs(d, exist_ok=True)

    total_tiles = 0
    for img_name in files:
        image_path = os.path.join(images_dir, img_name)
        image = cv2.imread(image_path)
        print(image_path)

        # Crop if enabled
        if crop_images:
            cropped = crop(image)
            if cropped is None:
                print(f"Skipping {img_name} (no yellow region found).")
                continue
        else:
            cropped = image  # use full image if cropping disabled

        base_name = os.path.splitext(img_name)[0]
        tiles = tile_and_save(cropped, base_name, img_out, lbl_out, tile_size, stride)
        total_tiles += tiles
    


    print(f"Created {total_tiles} tiles.")
    print(f"All saved under: {output_dir}")


'''
#creating empty tiles
split_and_tile(
    images_dir="/user/christoph.wald/u15287/big-scratch/00_uncropped_dataset/emptyYST",
    output_dir="/user/christoph.wald/u15287/big-scratch/02_splitted_data/train_labeled/SSL/empty_tiles",
    tile_size=640,
    stride=440,
    train_ratio=0.8,
    crop_images=True  # toggle cropping here
)
'''

#creating tiles from unsegmented images
output_dir = "/user/christoph.wald/u15287/big-scratch/02_splitted_data/train_labeled/SSL/images_not_segmented"
images_dir ="/user/christoph.wald/u15287/big-scratch/02_splitted_data/train_labeled/SSL/04_images_cropped"

# Get list of files already segmented
files_used = os.listdir("/user/christoph.wald/u15287/big-scratch/02_splitted_data/train_labeled/SSL/split/images/train")
print(f"Files that are segmented: {len(files_used)}.")

# Find files that are not yet segmented
all_files = os.listdir(images_dir)
files_to_use = [f for f in all_files if f not in files_used]
print(f"Files that are not already segmented: {len(files_to_use)}.")

# Create output folder
os.makedirs(output_dir, exist_ok=True)


# Copy files to output_dir
for f in files_to_use:
    shutil.copy(os.path.join(images_dir, f), os.path.join(output_dir, f))

# Now run split_and_tile on the output_dir
split_and_tile(
    images_dir=output_dir,
    output_dir=output_dir,
    tile_size=640,
    stride=440,
    train_ratio=0.8,
    crop_images=False
)

