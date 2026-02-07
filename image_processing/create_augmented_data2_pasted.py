import sys
sys.path.append("/user/christoph.wald/u15287/insect_pest_detection/modules")
import os
import cv2
import random
from modules_augmentation import *
from modules_segmentation import create_binary_mask
import ast
import numpy as np

'''
creates images with cut-out insects placed on and near background structures
'''

random.seed(43)
pest_types = ["BRAIIM", "LIRIBO", "TRIAVA"]
output_folder = "/user/christoph.wald/u15287/big-scratch/02_splitted_data/augmented_data/"

get_images = False #if False, skips the first step and just load the prepared images
get_cutout_boxes = False #if False, skips the second step and just load the prepared cut-out bounding boxes

#Step 1: randomly shear the images with empty yellow stick traps to create variations
if get_images:
    #Loading empty YSTs
    empty_image_folder = "/user/christoph.wald/u15287/big-scratch/emptyYST"
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

#load already prepared images
else:
    empty_image_folder = "/user/christoph.wald/u15287/big-scratch/02_splitted_data/augmented_data/sheared_YSTs"
    filenames = os.listdir(empty_image_folder)
    sheared_YSTs = []
    for filename in filenames:
        sheared_YSTs.append(cv2.imread(os.path.join(empty_image_folder, filename)))
    print(f"Loaded {len(sheared_YSTs)} empty, augmented images.")

#load mask
processed_mask = cv2.imread(
    #"/user/christoph.wald/u15287/insect_pest_detection/2_4_image_processing/masks/06_generated_mask_fat_plus.jpg", 
    "/user/christoph.wald/u15287/insect_pest_detection/image_processing/masks/04_generated_mask_fat.jpg",
    cv2.IMREAD_GRAYSCALE
)
gridcorners = np.load("/user/christoph.wald/u15287/insect_pest_detection/image_processing/masks/gridcorners.npy")
mask_h_line = np.load("/user/christoph.wald/u15287/insect_pest_detection/image_processing/masks/mask_h_line.npy")


#Step 2: cut out insects according to the given bounding boxes
if get_cutout_boxes:

    label_path = "/user/christoph.wald/u15287/big-scratch/02_splitted_data/train_unlabeled/05_created_labels_reduced"
    image_path = "/user/christoph.wald/u15287/big-scratch/02_splitted_data/train_unlabeled/04_images_cropped_reduced"
    label_files = os.listdir(label_path)

    all_insects = [[],[], []]
    skipped = 0
    for filename in label_files:
        print(filename)
        cls = [id in filename for id in pest_types].index(True)
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

            #for whiteflies, contours are found by color thresholding + finding regions contained inside loops
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

                #Find internal white pixels
                internal_white_mask = np.all(visual == result, axis=2) & (white_mask == 1)

                # Color internal white pixels blue
                visual[internal_white_mask] = (255, 0, 0)  # BGR blue
                
                # Identify blue pixels in visual
                blue_mask = (visual[:, :, 0] == 255) & (visual[:, :, 1] == 0) & (visual[:, :, 2] == 0)

                # Start from the current result
                merged = result.copy()

                #Replace only blue pixels with original colors from region_bgr
                merged[blue_mask] = region_bgr[blue_mask]

                result = merged

            #for leaf miner flies and fungus gnat, color thresholding is done with locally detected value threshold
            else:
                value = region_hsv[:, :, 2]
                otsu_thresh_value, binary_mask = cv2.threshold(value, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                
                '''
                #v1 result
                binary_mask_inv = cv2.bitwise_not(binary_mask)  # foreground=255, background=0
                foreground = cv2.bitwise_and(region_bgr, region_bgr, mask=binary_mask_inv)
                background = np.full_like(region_bgr, 255)  # white background
                result = np.where(binary_mask_inv[:, :, np.newaxis] == 0, background, foreground)
                '''
                
                #v2 result
                lower_yellow = np.array([18, 100, int(otsu_thresh_value)])
                upper_yellow = np.array([30, 255, 255])
                binary_mask = cv2.inRange(region_hsv, lower_yellow, upper_yellow)
                binary_mask_inv = cv2.bitwise_not(binary_mask)
                
                foreground = cv2.bitwise_and(region_bgr, region_bgr, mask=binary_mask_inv)
                background = np.full_like(region_bgr, 255)
                result = np.where(binary_mask_inv[:, :, np.newaxis] == 0, background, foreground)
                

            result = keep_central_contour(result)

            #compute ratio of non-white area
            non_white = np.any(result < 250, axis=2)
            non_white_ratio = np.sum(non_white) / non_white.size
            if non_white_ratio < 0.05:
                skipped += 1
                continue

            #find tight bounding box around non-white region
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


            #extra hsv variation for the whiteflies
            if filename.startswith("TRIAVA"):
                non_white_mask = np.any(cropped < 250, axis=2)  # shape: (h, w)
                # Convert cropped to float32 HSV
                hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV).astype(np.float32)
                
                # Apply random shifts only to insect pixels
                for c in range(3):
                    channel = hsv[:, :, c]
                    if c == 0:  # Hue
                        channel[non_white_mask] += random.uniform(-1, 1)
                    elif c == 1:  # Saturation
                        channel[non_white_mask] += 41 + random.uniform(-7.6, 7.6)
                    elif c == 2:  # Value
                        channel[non_white_mask] += -6 + random.uniform(-2.5, 2.5)
                    hsv[:, :, c] = np.clip(channel, 0, 255 if c != 0 else 179)
    
                augmented = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
                insects.append(augmented)

        all_insects[cls].extend(insects)
    print(f"Skipped {skipped}")
    print(f"Loaded {len(all_insects[0])} fungus gnats, {len(all_insects[1])} leaf miner flies and {len(all_insects[2])} whiteflies.")

    np.save(
        "/user/christoph.wald/u15287/big-scratch/02_splitted_data/augmented_data/all_insects.npy",
        np.array(all_insects, dtype=object),
        allow_pickle=True
    )
#load already prepared insects
else:
    all_insects = np.load(
        "/user/christoph.wald/u15287/big-scratch/02_splitted_data/augmented_data/all_insects.npy",
        allow_pickle=True
    )
    print(f"Loaded {len(all_insects[0])} fungus gnats, {len(all_insects[1])} leaf miner flies and {len(all_insects[2])} whiteflies.")

#total number of insects needed is calculated
n_per_region = 5 #per region
n_per_image = int(5.5 * n_per_region)
images_per_insect = 6
augmentations = 4 #4
placements = 2 #how many times is place-insects by region called 
n_total = images_per_insect * augmentations * n_per_image * placements
print(n_total)

#create a random selection to take from 
selected_insects = [[], [], []]
all_insects[0].extend(all_insects[0])
for i in range(3):
    selected_insects[i] = random.sample(all_insects[i], k=n_total)

label_path = os.path.join(output_folder, "labels")
image_path = os.path.join(output_folder, "images")
os.makedirs(label_path, exist_ok=True)
os.makedirs(image_path, exist_ok=True)

#Step 3: creating the augmented images
counter = 0
for i, emptyYST in enumerate(sheared_YSTs):

    #mask with fat lines defines regions around the background structures
    mask_fat = get_mask(emptyYST, processed_mask, gridcorners, mask_h_line)
    mask_binary_fat = (mask_fat < 128).astype(np.uint8) #convert to 0/1 mask

    #crop image
    imageYST = find_contour(emptyYST)
    x, y, w, h = cv2.boundingRect(imageYST)
    empty_YST_cropped = emptyYST[y:y+h, x:x+w]

    #mask with thin lines defines regions directly on the background structures
    mask = create_binary_mask(empty_YST_cropped) 
    mask_binary = (mask < 128).astype(np.uint8) #convert to 0/1 mask
    mask_binary = remove_border_connected(mask_binary) #remove borders to place insects only on the YS
        
    #for each prepared images four variations with different insects and augmentations are created
    for i in range(4):
        idx_insects = counter % 3
        idx_augmentation = counter % 4
        print(f"Class {pest_types[idx_insects]}, augmentation {idx_augmentation}." )
        
        img = empty_YST_cropped.copy()
        _, placed_boxes = place_insects_by_region(img, mask_binary, selected_insects[idx_insects], n_per_region) #places on the structures
        _, placed_boxes_extra = place_insects_by_region(img, mask_binary_fat, selected_insects[idx_insects], n_per_region) #places also near the structures
        placed_boxes.extend(placed_boxes_extra)
        
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
