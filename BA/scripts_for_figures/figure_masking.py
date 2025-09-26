import cv2
import numpy as np
import os
from modules_segmentation import *

def create_masked_image(mask_to_use, image):
    yellow_mask = mask_to_use == 0
    image_wo_grid = image.copy()
    image_wo_grid[yellow_mask] = [0, 255, 255]
    return image_wo_grid

def check_h_line(image, h_mid, color=(0, 0, 255), thickness=5):
    """
    Draws a horizontal line on the image
    """
    x1, y1, x2, y2 = h_mid
    img_with_line = image.copy()
    cv2.line(img_with_line, (x1, y1), (x2, y2), color=color, thickness=thickness)
    return img_with_line

# --- Load data ---
processed_mask = cv2.imread(
    "/user/christoph.wald/u15287/insect_pest_detection/2_4_image_processing/masks/04_generated_mask_fat.jpg",
    cv2.IMREAD_GRAYSCALE
)

original_mask = cv2.imread(
    "/user/christoph.wald/u15287/big-scratch/00_uncropped_dataset/YSTohneInsekten/IMG_5885.JPG"
)

mask_h_line = np.load("/user/christoph.wald/u15287/insect_pest_detection/2_4_image_processing/masks/mask_h_line.npy")
gridcorners = np.load("/user/christoph.wald/u15287/insect_pest_detection/2_4_image_processing/masks/gridcorners.npy")

image_folder = "/user/christoph.wald/u15287/big-scratch/02_splitted_data/train_labeled/images_uncropped"
image_files = os.listdir(image_folder)
image_file = image_files[11]
image = cv2.imread(os.path.join(image_folder, image_file))


# --- Processing ---
imageYST = find_contour(image)
imagecorners = find_corners(image, imageYST)

# Homography & transform mask
H, _ = cv2.findHomography(gridcorners, imagecorners, cv2.RANSAC)
mask = cv2.warpPerspective(processed_mask, H, (image.shape[1], image.shape[0]))
original_mask_transformed = cv2.warpPerspective(create_binary_mask(original_mask), H, (image.shape[1], image.shape[0]))

# figure A
figurea_image = draw_corners(create_masked_image(original_mask_transformed,image), imagecorners)

# horizontal line transformation
h_line_pts = mask_h_line.reshape(-1, 1, 2)
h_line_pts_warped = cv2.perspectiveTransform(h_line_pts, H)
x1w, y1w, x2w, y2w = h_line_pts_warped.reshape(-1).astype(np.int32)
mask_h = (x1w, y1w, x2w, y2w)

image_h = get_h_mid(create_binary_mask(image))
dy = get_midpoint(image_h) - get_midpoint(mask_h)
print(f"Distance: {dy}")

# figure B with midlines
figureb_image = create_masked_image(mask, image)
figureb_image = check_h_line(figureb_image, mask_h, thickness=8)   # red = mask midline
figureb_image = check_h_line(figureb_image, image_h, color = (0,255,0), thickness=8)  # green = image midline

# figure C (shifted mask)
H, W = mask.shape[:2]
M = np.float32([[1, 0, 0], [0, 1, dy]])
shifted_mask = cv2.warpAffine(mask, M, (W, H), borderValue=255)
figurec_image = create_masked_image(shifted_mask, image)

# figure D (binary)
figured_image = cv2.cvtColor(create_binary_mask(figurec_image), cv2.COLOR_GRAY2BGR)

# --- Combine into 2x2 grid ---
row1 = cv2.hconcat([figurea_image, figureb_image])
row2 = cv2.hconcat([figurec_image, figured_image])
final_grid = cv2.vconcat([row1, row2])

# Save four images with quality = 25 (smaller file size)
cv2.imwrite("figureA.jpg", figurea_image, [int(cv2.IMWRITE_JPEG_QUALITY), 25])
cv2.imwrite("figureB.jpg", figureb_image, [int(cv2.IMWRITE_JPEG_QUALITY), 25])
cv2.imwrite("figureC.jpg", figurec_image, [int(cv2.IMWRITE_JPEG_QUALITY), 25])
cv2.imwrite("figureD.jpg", figured_image, [int(cv2.IMWRITE_JPEG_QUALITY), 25])

print("Saved figureA.jpg, figureB.jpg, figureC.jpg, figureD.jpg with quality=25")
