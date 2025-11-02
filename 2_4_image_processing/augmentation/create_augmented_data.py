import sys
sys.path.append("/user/christoph.wald/u15287/insect_pest_detection/modules")
import os
import cv2
import random
from modules_augmentation import *
from modules_segmentation import create_binary_mask
import ast
import numpy as np

random.seed(43)
pest_types = ["BRAIIM", "LIRIBO", "TRIAVA"]
output_folder = "/user/christoph.wald/u15287/big-scratch/02_splitted_data/augmented_data/"

get_images = False
get_cutout_boxes = False

if get_images:
    #Loading empty YSTs
    empty_image_folder = "/user/christoph.wald/u15287/big-scratch/00_uncropped_dataset/emptyYST"
    empty_image_files = os.listdir(empty_image_folder)
    empty_YSTs = []
    for f in empty_image_files:
        empty_YSTs.append(cv2.imread(os.path.join(empty_image_folder, f)))

    print(f"Loaded {len(empty_YSTs)} empty images.")

    #Shearing to create more data
    sheared_YSTs = []
    for YST in empty_YSTs:
        sheared_YSTs.append(YST)
        sheared_YSTs.append(random_shear(YST))
        sheared_YSTs.append(random_shear(YST))

    print(f"Sheared the empty images twice to create {len(sheared_YSTs)} images.")

    output_folder_sheared = "/user/christoph.wald/u15287/big-scratch/02_splitted_data/augmented_data/sheared_YSTs"
    os.makedirs(output_folder_sheared, exist_ok=True)
    for i, YST in enumerate(sheared_YSTs):
        cv2.imwrite(os.path.join(output_folder_sheared, f"sheared_YST_{i}.jpg"), YST)
    
else:
    empty_image_folder = "/user/christoph.wald/u15287/big-scratch/02_splitted_data/augmented_data/sheared_YSTs"
    filenames = os.listdir(empty_image_folder)
    sheared_YSTs = []
    for filename in filenames:
        sheared_YSTs.append(cv2.imread(os.path.join(empty_image_folder, filename)))
    print(f"Loaded {len(sheared_YSTs)} empty, augmented images.")

'''
#masking to create areas of interest (were to put the insects in)
processed_mask = cv2.imread(
    #"/user/christoph.wald/u15287/insect_pest_detection/2_4_image_processing/masks/06_generated_mask_fat_plus.jpg", 
    "/user/christoph.wald/u15287/insect_pest_detection/2_4_image_processing/masks/04_generated_mask_fat.jpg",
    cv2.IMREAD_GRAYSCALE
)
gridcorners = np.load("/user/christoph.wald/u15287/insect_pest_detection/2_4_image_processing/masks/gridcorners.npy")
mask_h_line = np.load("/user/christoph.wald/u15287/insect_pest_detection/2_4_image_processing/masks/mask_h_line.npy")

masks = []

for YST in sheared_YSTs:
    masks.append(get_mask(YST, processed_mask, gridcorners, mask_h_line))

print(f"Created areas of interest for the empty images.")
'''

if get_cutout_boxes:
#Get the cut out insects
    label_path = "/user/christoph.wald/u15287/big-scratch/02_splitted_data/train_labeled/SSL/05_created_labels"
    image_path = "/user/christoph.wald/u15287/big-scratch/02_splitted_data/train_labeled/SSL/04_images_cropped"
    label_files = os.listdir(label_path)



    all_insects = [[],[], []]
    skipped = 0
    for filename in label_files:
        print(filename)
        cls = [id in filename for id in pest_types].index(True)
        #print(cls)
        abs_boxes = []
        with open(os.path.join(label_path,filename), 'r') as f:
            for line in f:
                if line.strip():
                    x, y, w, h = map(float, ast.literal_eval(line))
                    x1, y1, x2, y2 = x, y, x + w, y + h
                    abs_boxes.append([x1, y1, x2, y2])


        image = cv2.imread(os.path.join(image_path, os.path.splitext(filename)[0] + ".jpg"))
        rectangles = [enlarge_box(box, image.shape, scale=1) for box in abs_boxes]
        
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        insects = []
        for box in rectangles:
            x1, y1, x2, y2 = map(int, box)
            region_bgr = image[y1:y2, x1:x2]
            if region_bgr.size == 0:
                continue
            region_hsv = hsv_image[y1:y2, x1:x2]

            if filename.startswith("TRIAVA"):
                binary_mask = create_binary_mask(region_hsv)

                # Apply mask
                foreground = cv2.bitwise_and(region_bgr, region_bgr, mask=binary_mask)
                background = np.full_like(region_bgr, 255)
                result = np.where(binary_mask[:, :, np.newaxis] == 0, background, foreground)

                #Convert to grayscale to detect white pixels
                gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
                white_mask = (gray >= 250).astype(np.uint8)  # 1 = white, 0 = non-white

                visual = result.copy()

                #Prepare mask for floodFill (2 pixels larger)
                h, w = white_mask.shape
                flood_mask = np.zeros((h+2, w+2), np.uint8)

                #Flood-fill all border-connected white pixels in red
                for x in range(w):
                    if white_mask[0, x]:
                        cv2.floodFill(visual, flood_mask, (x, 0), (0, 0, 255))  # red
                    if white_mask[h-1, x]:
                        cv2.floodFill(visual, flood_mask, (x, h-1), (0, 0, 255))
                for y in range(h):
                    if white_mask[y, 0]:
                        cv2.floodFill(visual, flood_mask, (0, y), (0, 0, 255))
                    if white_mask[y, w-1]:
                        cv2.floodFill(visual, flood_mask, (w-1, y), (0, 0, 255))

                #Find internal white pixels (still white in visual)
                internal_white_mask = np.all(visual == result, axis=2) & (white_mask == 1)

                # Color internal white pixels blue
                visual[internal_white_mask] = (255, 0, 0)  # BGR blue
                
                # Identify blue pixels in visual (internal white)
                blue_mask = (visual[:, :, 0] == 255) & (visual[:, :, 1] == 0) & (visual[:, :, 2] == 0)

                # Step 2: Start from the current result
                merged = result.copy()

                # Step 3: Replace only blue pixels with original colors from region_bgr
                merged[blue_mask] = region_bgr[blue_mask]

                result = merged

            else:

                value = region_hsv[:, :, 2]
                otsu_thresh, binary_mask = cv2.threshold(value, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                binary_mask_inv = cv2.bitwise_not(binary_mask)  # foreground=255, background=0
                foreground = cv2.bitwise_and(region_bgr, region_bgr, mask=binary_mask_inv)
                background = np.full_like(region_bgr, 255)  # white background
                result = np.where(binary_mask_inv[:, :, np.newaxis] == 0, background, foreground)


            result = keep_central_contour(result)
            # --- Compute ratio of non-white area ---
            non_white = np.any(result < 250, axis=2)
            non_white_ratio = np.sum(non_white) / non_white.size
            if non_white_ratio < 0.05:
                skipped += 1
                continue

            # --- Find tight bounding box around non-white region ---
            coords = np.argwhere(non_white)
            if coords.size == 0:
                skipped += 1
                continue

            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0) + 1  # +1 because slicing is exclusive

            # Crop result and update box coordinates (relative to original image)
            cropped = result[y_min:y_max, x_min:x_max]
            new_box = (x1 + x_min, y1 + y_min, x1 + x_max, y1 + y_max)

            insects.append(cropped)
        all_insects[cls].extend(insects)
    print(f"Skipped {skipped}")
    print(f"Loaded {len(all_insects[0])} fungus gnats and {len(all_insects[1])} leaf miner flies.")

    np.save(
        "/user/christoph.wald/u15287/big-scratch/02_splitted_data/augmented_data/all_insects.npy",
        np.array(all_insects, dtype=object),
        allow_pickle=True
    )
else:
    all_insects = np.load(
        "/user/christoph.wald/u15287/big-scratch/02_splitted_data/augmented_data/all_insects.npy",
        allow_pickle=True
    )
    print(f"Loaded {len(all_insects[0])} fungus gnats and {len(all_insects[1])} leaf miner flies.")

n_per_region = 4 #per region
n_per_image = int(5.5 * n_per_region)
images_per_insect = 6
augmentations = 4
n_total = images_per_insect * augmentations * n_per_image
print(n_total)

#create a random selection to take from 

label_path = os.path.join(output_folder, "labels")
image_path = os.path.join(output_folder, "images")
os.makedirs(label_path, exist_ok=True)
os.makedirs(image_path, exist_ok=True)

selected_insects = [[], [], []]
for i in range(len(all_insects)):
    selected_insects[i] = random.sample(all_insects[i], k = n_total)

counter = 0
for i, emptyYST in enumerate(sheared_YSTs):
    imageYST = find_contour(emptyYST)
    x, y, w, h = cv2.boundingRect(imageYST)
    empty_YST_cropped = emptyYST[y:y+h, x:x+w] 
    mask = create_binary_mask(empty_YST_cropped) 
    mask_binary = (mask < 128).astype(np.uint8) #convert to 0/1 mask
    mask_binary = remove_border_connected(mask_binary) #remove borders to place insects only on the YS

    for i in range(4):
        idx_insects = counter % 3
        idx_augmentation = counter % 4
        print(f"Class {pest_types[idx_insects]}, augmentation {idx_augmentation}." )
        
        img = empty_YST_cropped.copy()
        _, placed_boxes = place_insects_by_region(img, mask_binary, selected_insects[idx_insects], n_per_region)

        if idx_augmentation == 1:
            img = cv2.rotate(img, cv2.ROTATE_180)
        if idx_augmentation == 2:
            img = random_light_variation(img)
        if idx_augmentation == 3:
            img = cv2.rotate(img, cv2.ROTATE_180)
            img = random_light_variation(img)
        cv2.imwrite(os.path.join(image_path, f"{pest_types[idx_insects]}_{counter}_YST{i}_aug{idx_augmentation}.jpg"), img)
        with open(os.path.join(label_path, f"{pest_types[idx_insects]}_{counter}_YST{i}_aug{idx_augmentation}.txt"), "w") as f:
            for (x1, y1, x2, y2) in placed_boxes:
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2]) # Convert np.int64 → int
                w = x2 - x1 # Convert (x1, y1, x2, y2) → (x, y, w, h)
                h = y2 - y1
                f.write(f"[{x1}, {y1}, {w}, {h}]\n")

        counter += 1
