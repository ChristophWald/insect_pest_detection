import cv2
import os
from modules_segmentation import *
import numpy as np

#load background images
grid_folder = "/user/christoph.wald/u15287/big-scratch/00_uncropped_dataset/YSTohneInsekten"
grid_files = os.listdir(grid_folder)

#for all images without insects, collect the corners of the YST and make a noisefree black-and-white image
corners_grids = [] 
cleaned_masks = [] 

#binarize and clean all images, find contour corners
for grid_file in grid_files:
    grid = cv2.imread(os.path.join(grid_folder, grid_file))
    print(f"Processing {grid_file}")     
    mask = create_binary_mask(grid)
    cleaned_mask = denoise_mask(mask)
    cleaned_masks.append(cleaned_mask)
    gridYST = find_contour(grid)
    corners = find_corners(grid, gridYST)
    corners_grids.append(corners)

#add all masks together to close gaps
print("Adding Mask.")
combined_mask = cleaned_masks[0].copy()
for i in range(1, len(cleaned_masks)):
    H, _ = cv2.findHomography(corners_grids[i], corners_grids[0], cv2.RANSAC)
    aligned_mask = cv2.warpPerspective(cleaned_masks[i], H, (cleaned_masks[0].shape[1], cleaned_masks[0].shape[0]))
    combined_mask = cv2.bitwise_and(combined_mask, aligned_mask)

#thicken the lines of the grid, to prevent errors from small misalignments    
print("Thicken lines.")
grown_mask = grow_mask(combined_mask, growth_pixels=25) #image for use in the alignment with images
cv2.imwrite("/user/christoph.wald/u15287/insect_pest_detection/2_4_image_processing/mask.jpg", grown_mask)

#save two points from the horizontal midline
x1, y1, x2, y2 = get_h_mid(create_binary_mask(cv2.imread("/user/christoph.wald/u15287/big-scratch/00_uncropped_dataset/YSTohneInsekten/IMG_5885.JPG")))
h_line_pts = np.array([
    [x1, y1],
    [x2, y2]
], dtype=np.float32).reshape(-1,1,2)

np.save("/user/christoph.wald/u15287/insect_pest_detection/2_4_image_processing/mask_h_line.npy", h_line_pts)

#save the corners of the YST
np.save("/user/christoph.wald/u15287/insect_pest_detection/2_4_image_processing/gridcorners.npy", corners_grids[0])