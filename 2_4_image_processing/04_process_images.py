import time
import os
import cv2
import numpy as np
from modules_segmentation import *

#if True saves images of intermediate steps
inspection = False

###Setup

start = time.time()

image_folder = "/user/christoph.wald/u15287/big-scratch/02_splitted_data/train_labeled/images_uncropped"
image_files = os.listdir(image_folder)
label_folder = "/user/christoph.wald/u15287/big-scratch/02_splitted_data/train_labeled/labels_uncropped"
output_folder_images = "/user/christoph.wald/u15287/big-scratch/02_splitted_data/train_labeled/images_masked"
output_folder_labels = "/user/christoph.wald/u15287/big-scratch/02_splitted_data/train_labeled/labels_for_masked_images"
os.makedirs(output_folder_images, exist_ok=True)
os.makedirs(output_folder_labels, exist_ok=True)
test_folder = "/user/christoph.wald/u15287/insect_pest_detection/image_processing_in_progress/test"
os.makedirs(test_folder, exist_ok=True)

#load mask and corners of the mask YST for alignment
handcrafted_mask = cv2.imread(
    "/user/christoph.wald/u15287/insect_pest_detection/2_4_image_processing/generated_mask_fat.jpg", 
    cv2.IMREAD_GRAYSCALE
)
gridcorners = np.load("/user/christoph.wald/u15287/insect_pest_detection/2_4_image_processing/gridcorners.npy")



problems = []

###Processing

for i, image_file in enumerate(image_files):

    filename= image_file.split(".")[0]

    #Load image file
    print(f"Loading {image_file}, {i}/{len(image_files)}")
    image = cv2.imread(os.path.join(image_folder, image_file))
    if inspection: cv2.imwrite(os.path.join(test_folder, filename + "_01_original.jpg"), image)
    
    #find YST contour of image
    imageYST = find_contour(image)
    imagecorners = find_corners(image, imageYST)
    if len(imagecorners) == 0:
        print(f"No YST found for {image_file}")
        problems.append(image_file)
        continue

    #find transformation
    H, _ = cv2.findHomography(gridcorners, imagecorners, cv2.RANSAC)

    
    #first transformation: 
    mask = cv2.warpPerspective(handcrafted_mask, H, (image.shape[1], image.shape[0]))
    if inspection: cv2.imwrite(os.path.join(test_folder, filename + "_02a_aligned_mask.jpg"), mask) 
    
    
    #second transformation: correct vertical misalignment
    mask_h = get_h_mid(mask)
    image_h = get_h_mid(create_binary_mask(image))
    dy = get_midpoint(image_h)- get_midpoint(mask_h)
    if inspection:
        check_h_line(mask, mask_h,os.path.join(test_folder, filename + "_02aa_mask_h.jpg") )
        check_h_line(create_binary_mask(image), image_h, os.path.join(test_folder, filename + "_02ab_image_h.jpg") )
    H, W = mask.shape[:2]
    M = np.float32([[1, 0, 0], [0, 1, dy]])  # translation matrix
    mask= cv2.warpAffine(mask, M, (W, H), borderValue=255)  # white background
    if inspection: cv2.imwrite(os.path.join(test_folder, filename + "_02b_shifted_mask.jpg"), mask) 
    
   
    #replace black background in image with yellow (background color) by using the mask
    yellow_mask = mask == 0 
    image_wo_grid = image.copy()
    image_wo_grid[yellow_mask] = [0,255,255]
    
    #crop the image
    x, y, w, h = cv2.boundingRect(imageYST)
    cropped_image_wo_grid = image_wo_grid[y:y+h, x:x+w]
    if inspection: cv2.imwrite(os.path.join(test_folder, filename + "_03_cropped_image.jpg"), cropped_image_wo_grid) 
    
    #finding the rectangles given by the yolo labels
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

    #Saving processed images and labels (as rectangles)
    cv2.imwrite(os.path.join(output_folder_images, image_file), cropped_image_wo_grid)
    with open(os.path.join(output_folder_labels, filename + ".txt"), "w") as f:
        for item in cropped_yolo_rectangles:
            f.write(str(item) + "\n") 
    with open("images_without_YST.txt", "w") as f:
        for p in problems:
            f.write(p + "\n")
    