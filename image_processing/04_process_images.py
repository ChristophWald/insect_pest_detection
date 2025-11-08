import sys
sys.path.append("/user/christoph.wald/u15287/insect_pest_detection/modules")
import time
import os
import cv2
import numpy as np
from modules_segmentation import *

'''
uses a given mask to produce masked images for segmentation
also saves cropped images and the according labels
'''
def compute_mask_coverage(mask, rectangles, threshold=0.2):
    """
    used only for detecting how many yolo labels are affected by the mask
    mask: np.ndarray (grayscale, same size as original image)
    rectangles: list of rectangles [(x_min, y_min, x_max, y_max), ...]
    Returns: list of (rect_idx, coverage_ratio, is_covered)
    """
    results = []
    for i, (x1, y1, w, h) in enumerate(rectangles):
        x2 = x1 + w
        y2 = y1 + h
        # Ensure bounds within image
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(mask.shape[1], int(x2)), min(mask.shape[0], int(y2))
        if x2 <= x1 or y2 <= y1:
            results.append((i, 0.0, False))
            continue

        # Extract mask region
        region = mask[y1:y2, x1:x2]

        # Count masked pixels (assuming 0=masked, 255=background)
        masked_pixels = np.sum(region == 0)
        total_pixels = region.size
        ratio = masked_pixels / total_pixels

        is_covered = ratio >= threshold
        results.append((i, ratio, is_covered))
    return results


#some flags
inspection = False #if True, saves images of single steps in a test folder
save_images = True #if False, no saving of cropped images, labels and masked images
process_labels = False #if True, also crops given labels
remove_covered_labels = False #if True (and also process_labels is True), which are covered 20% or more by the mask are deleted
remove_and_save_yolo = False #same as process labels & remove covered labels but in yolo format

###Setup
#image_folder = "/user/christoph.wald/u15287/big-scratch/02_splitted_data/train_labeled/SSL/02_images_rotated"
image_folder = "/user/christoph.wald/u15287/big-scratch/02_splitted_data/train_unlabeled/02_images_rotated"
#label_folder = "/user/christoph.wald/u15287/big-scratch/02_splitted_data/train_labeled/SSL/02_labels_rotated"

#output_folder_images_masked = "/user/christoph.wald/u15287/big-scratch/02_splitted_data/train_labeled/SSL/03_images_masked_test6"
output_folder_images_masked = "/user/christoph.wald/u15287/big-scratch/02_splitted_data/train_unlabeled/03_images_masked"
#output_folder_images_cropped = "/user/christoph.wald/u15287/big-scratch/02_splitted_data/train_labeled/SSL/04_images_cropped"
output_folder_images_cropped = "/user/christoph.wald/u15287/big-scratch/02_splitted_data/train_unlabeled/04_images_cropped"
#output_folder_labels = "/user/christoph.wald/u15287/big-scratch/02_splitted_data/train_labeled/SSL/04_labels_cropped"
os.makedirs(output_folder_images_masked, exist_ok=True)
os.makedirs(output_folder_images_cropped, exist_ok=True)
if process_labels or remove_and_save_yolo: os.makedirs(output_folder_labels, exist_ok=True)

test_folder = "/user/christoph.wald/u15287/insect_pest_detection/2_4_image_processing/test"
if inspection: os.makedirs(test_folder, exist_ok=True)

#load mask and corners of the mask YST for alignment

processed_mask = cv2.imread(
    #"/user/christoph.wald/u15287/insect_pest_detection/2_4_image_processing/masks/01_generated_mask_slim.jpg", 
    #"/user/christoph.wald/u15287/insect_pest_detection/2_4_image_processing/masks/02_03_handcrafted_mask_fat.jpg",
    #"/user/christoph.wald/u15287/insect_pest_detection/2_4_image_processing/masks/04_generated_mask_fat.jpg",
    #"/user/christoph.wald/u15287/insect_pest_detection/2_4_image_processing/masks/05_generated_mask_fat_thick_line_right.jpg",
    "/user/christoph.wald/u15287/insect_pest_detection/2_4_image_processing/masks/06_generated_mask_fat_plus.jpg",
    cv2.IMREAD_GRAYSCALE
)


gridcorners = np.load("/user/christoph.wald/u15287/insect_pest_detection/2_4_image_processing/masks/gridcorners.npy")
mask_h_line = np.load("/user/christoph.wald/u15287/insect_pest_detection/2_4_image_processing/masks/mask_h_line.npy")

problems_corners = []
problems_shifts = []
image_files = os.listdir(image_folder)

###Processing

for i, image_file in enumerate(image_files):

    filename= image_file.split(".")[0]

    #Load image file
    print(f"Loading {image_file}, {i}/{len(image_files)}")
    image = cv2.imread(os.path.join(image_folder, image_file))
    if inspection: cv2.imwrite(os.path.join(test_folder, filename + "_01_original.jpg"), image)
    


    #find YST contour of image
    imageYST = find_contour(image)

    #save cropped image
    x, y, w, h = cv2.boundingRect(imageYST)
    cropped_image = image[y:y+h, x:x+w]
    if save_images: cv2.imwrite(os.path.join(output_folder_images_cropped, image_file), cropped_image)

    
    #find corners if possible, if not skip the image
    imagecorners = find_corners(image, imageYST)
    if len(imagecorners) == 0:
        print(f"No YST found for {image_file}")
        problems_corners.append(image_file)
        continue

    
    #find transformation
    H, _ = cv2.findHomography(gridcorners, imagecorners, cv2.RANSAC)

    #first transformation: 
    mask = cv2.warpPerspective(processed_mask, H, (image.shape[1], image.shape[0]))
    if inspection: cv2.imwrite(os.path.join(test_folder, filename + "_02a_aligned_mask.jpg"), mask) 
    
    
    #transform midline of mask as above
    h_line_pts = mask_h_line.reshape(-1,1,2)
    h_line_pts_warped = cv2.perspectiveTransform(h_line_pts, H)
    x1w, y1w, x2w, y2w = h_line_pts_warped.reshape(-1).astype(np.int32)
    mask_h = (x1w, y1w, x2w, y2w)

    #second transformation: correct vertical misalignment

    dy = get_distance_h_mid(create_binary_mask(image), mask_h)
    print(dy)
    H, W = mask.shape[:2]
    M = np.float32([[1, 0, 0], [0, 1, dy]])  # translation matrix
    mask= cv2.warpAffine(mask, M, (W, H), borderValue=255)  # white background
    if inspection: cv2.imwrite(os.path.join(test_folder, filename + "_02b_shifted_mask.jpg"), mask) 
    

    '''
    old shifting
    image_h = get_h_mid(create_binary_mask(image))
    dy = get_midpoint(image_h)- get_midpoint(mask_h)
    print (dy)
    if abs(dy) > 500:
        print(f"Skipped vertical alignment, wrong horizontal line with offset {dy}")
        problems_shifts.append(image_file)
    else:
        if inspection:
            pass
            #function changed
            #check_h_line(mask, mask_h,os.path.join(test_folder, filename + "_02aa_mask_h.jpg") )
            #check_h_line(create_binary_mask(image), image_h, os.path.join(test_folder, filename + "_02ab_image_h.jpg") )
        H, W = mask.shape[:2]
        M = np.float32([[1, 0, 0], [0, 1, dy]])  # translation matrix
        mask= cv2.warpAffine(mask, M, (W, H), borderValue=255)  # white background
        if inspection: cv2.imwrite(os.path.join(test_folder, filename + "_02b_shifted_mask.jpg"), mask) 
    '''


    #replace black background in image with yellow (background color) by using the mask
    yellow_mask = mask == 0 
    image_wo_grid = image.copy()
    image_wo_grid[yellow_mask] = [0,255,255]
    
    #crop the image processed image
    cropped_image_wo_grid = image_wo_grid[y:y+h, x:x+w]
    if inspection: cv2.imwrite(os.path.join(test_folder, filename + "_03_cropped_image.jpg"), cropped_image_wo_grid)
    if save_images: cv2.imwrite(os.path.join(output_folder_images_masked, image_file), cropped_image_wo_grid)
    
    #finding the rectangles given by the yolo labels
    if process_labels:
        label_file = os.path.splitext(image_file)[0] + ".txt"
        label_path = os.path.join(label_folder, label_file)
        with open(label_path, "r") as f:
            yolo_labels = f.read().splitlines()
        yolo_rectangles = yolo_labels_to_rectangles(yolo_labels, image.shape)

        if remove_covered_labels:
            coverage_results = compute_mask_coverage(mask, yolo_rectangles, threshold=0.2)
            yolo_rectangles = [
                rect for i, rect in enumerate(yolo_rectangles)
                if not coverage_results[i][2]
            ]
            
            covered_labels = [i for i, ratio, covered in coverage_results if covered]
            if covered_labels:
                with open("labels_partially_masked.txt", "a") as f:
                    f.write(f"{image_file}: {len(covered_labels)} labels ≥20% covered\n")

        cropped_yolo_rectangles = transform_rectangles_to_cropped(yolo_rectangles, x, y,  w,h)
        if inspection:
            image_cropped = image[y:y+h, x:x+w]
            image_labels = draw_bounding_boxes(image_cropped, cropped_yolo_rectangles, color = (0,255,0))
            cv2.imwrite(os.path.join(test_folder, filename + "_04_w_yolo_labels.jpg"), image_labels)
        
        with open(os.path.join(output_folder_labels, filename + ".txt"), "w") as f:
            for item in cropped_yolo_rectangles:
                f.write(str(item) + "\n") 

    #this is for saving only the yolo-formatted labels that are not covered by the mask
    if remove_and_save_yolo:
        # Load YOLO labels
        label_file = os.path.splitext(image_file)[0] + ".txt"
        label_path = os.path.join(label_folder, label_file)
        with open(label_path, "r") as f:
            yolo_labels = f.read().splitlines()
        
        #print(yolo_labels)
        # Split class and bbox for processing
        yolo_classes = []
        yolo_bboxes = []
        for label in yolo_labels:
            parts = label.split()
            yolo_classes.append(parts[0])
            yolo_bboxes.append([float(x) for x in parts[1:]])  # x_center, y_center, w, h

        #print(yolo_classes)
        #print(yolo_bboxes)
        # Convert YOLO to rectangles to check coverage
        rectangles = yolo_labels_to_rectangles(yolo_labels, image.shape)
        #print(rectangles)
        
        coverage_results = compute_mask_coverage(mask, rectangles, threshold=0.2)
        
        # Keep only rectangles that are not covered
        rectangles = [rect for i, rect in enumerate(rectangles) if not coverage_results[i][2]]
        yolo_classes = [cls for i, cls in enumerate(yolo_classes) if not coverage_results[i][2]]
        
        #print(rectangles)
        covered_labels = [i for i, ratio, covered in coverage_results if covered]
        if covered_labels:
            with open("labels_partially_masked.txt", "a") as f:
                f.write(f"{image_file}: {len(covered_labels)} labels ≥20% covered\n")
        
        # Crop rectangles
        cropped_rectangles = transform_rectangles_to_cropped(rectangles, x, y, w, h)

        # Optional inspection
        if inspection:
            image_cropped = image[y:y+h, x:x+w]
            image_labels = draw_bounding_boxes(image_cropped, cropped_rectangles, color=(0, 255, 0))
            cv2.imwrite(os.path.join(test_folder, filename + "_04_w_yolo_labels.jpg"), image_labels)
        
        # Convert cropped rectangles back to YOLO format and remap classes
        #the remapping is only needed because somehow the labels are different from the ground truth
        #fist discovered in 01_labels_uncropped
        class_map = {"0": "1", "1": "3", "3": "0"}  # mapping old -> new

        yolo_lines = []
        for cls, rect in zip(yolo_classes, cropped_rectangles):
            # remap class if in map, else keep original
            cls_new = class_map.get(cls, cls)
            
            # convert rectangle to YOLO format
            yolo_bbox = rectangle_to_yolo(rect, w, h)
            
            # assemble line
            line = " ".join([cls_new] + [f"{v:.6f}" for v in yolo_bbox])
            yolo_lines.append(line)
        # Save YOLO labels
        with open(os.path.join(output_folder_labels, filename + ".txt"), "w") as f:
            f.write("\n".join(yolo_lines) + "\n")

with open("images_not_processed.txt", "w") as f:
    for p in problems_corners:
        f.write(p + "\n")
with open("images_not_shifted.txt", "w") as f:
    for p in problems_shifts:
        f.write(p + "\n")