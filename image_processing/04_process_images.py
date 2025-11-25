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

#flags
inspection = False #if True, saves images of single steps in a test folder
save_images = True #if False, no saving of cropped images, labels and masked images
process_labels = False #if True, also crops given labels

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
if process_labels: os.makedirs(output_folder_labels, exist_ok=True)

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

        cropped_yolo_rectangles = transform_rectangles_to_cropped(yolo_rectangles, x, y,  w,h)
        if inspection:
            image_cropped = image[y:y+h, x:x+w]
            image_labels = draw_bounding_boxes(image_cropped, cropped_yolo_rectangles, color = (0,255,0))
            cv2.imwrite(os.path.join(test_folder, filename + "_04_w_yolo_labels.jpg"), image_labels)
        
        with open(os.path.join(output_folder_labels, filename + ".txt"), "w") as f:
            for item in cropped_yolo_rectangles:
                f.write(str(item) + "\n") 

with open("images_not_processed.txt", "w") as f:
    for p in problems_corners:
        f.write(p + "\n")
