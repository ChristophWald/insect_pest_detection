import cv2
from modules_segmentation import *


image = cv2.imread("/user/christoph.wald/u15287/big-scratch/00_uncropped_dataset/YSTohneInsekten/gimp3_IMG_5885.JPG")

inspection = False

#calculate and save corners of YST
mask = create_binary_mask(image)
YST = find_contour(image)
corners = find_corners(image, YST)
if inspection: draw_corners(image, corners, "/user/christoph.wald/u15287/insect_pest_detection/2_4_image_processing/check_mask_corners.jpg")
np.save("/user/christoph.wald/u15287/insect_pest_detection/2_4_image_processing/gridcorners.npy", corners)

#denoising
mask_inv = cv2.bitwise_not(mask)  #inverting, because morphologyEx expects white foreground
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
cleaned_inv = cv2.morphologyEx(mask_inv, cv2.MORPH_OPEN, kernel)
cleaned_mask = cv2.bitwise_not(cleaned_inv)

cv2.imwrite("/user/christoph.wald/u15287/insect_pest_detection/2_4_image_processing/mask.jpg", cleaned_mask)

#optional for masks with thin lines
#grown_mask = grow_mask(cleaned_mask, 50)
#cv2.imwrite("/user/christoph.wald/u15287/insect_pest_detection/2_4_image_processing/handcrafted_mask1_grown.jpg", grown_mask)