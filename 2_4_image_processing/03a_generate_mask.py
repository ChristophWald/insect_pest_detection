import cv2
import os
from modules_segmentation import *

grid_folder = "/user/christoph.wald/u15287/big-scratch/00_uncropped_dataset/YSTohneInsekten"
grid_files = os.listdir(grid_folder)

#for all images without insects, collect the corners of the YST and make a noisefree black-and-white image
corners_grids = [] 
cleaned_masks = [] 

for grid_file in grid_files:
    
    grid = cv2.imread(os.path.join(grid_folder, grid_file))
    print(f"Processing {grid_file}")     
    mask = create_binary_mask(grid)
    #Denoising
    mask_inv = cv2.bitwise_not(mask)  #inverting, because morphologyEx expects white foreground
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    cleaned_inv = cv2.morphologyEx(mask_inv, cv2.MORPH_OPEN, kernel)
    cleaned_mask = cv2.bitwise_not(cleaned_inv)  
    cleaned_masks.append(cleaned_mask)

    gridYST = find_contour(grid)
    corners = find_corners(grid, gridYST)
    corners = order_corners(corners)
    corners_grids.append(corners)

#add all masks together, to close gaps
print("Adding Mask.")
combined_mask = cleaned_masks[0].copy()
for i in range(1, len(cleaned_masks)):
    H, _ = cv2.findHomography(corners_grids[i], corners_grids[0], cv2.RANSAC)
    aligned_mask = cv2.warpPerspective(cleaned_masks[i], H, (cleaned_masks[0].shape[1], cleaned_masks[0].shape[0]))
    combined_mask = cv2.bitwise_and(combined_mask, aligned_mask)

#thicken the lines of the grid, to prevent errors from small misalignments    
print("Thicken lines.")
grown_mask = grow_mask(combined_mask, growth_pixels=25) #image for use in the alignment with images
cv2.imwrite("/user/christoph.wald/u15287/insect_pest_detection/image_processing/mask.jpg", grown_mask)

#corners of the yst in the mask for use in the alignment with images
with open("mask_corners.py", "w") as f:
    f.write("import numpy as np\n")
    f.write("gridcorners = np.array(" + repr(corners_grids[0].tolist()) + ")\n")