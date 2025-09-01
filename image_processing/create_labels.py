import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import math
from modules_segmentation import *

def coverage(rect1, rect2):
    """
    Compute how much of rect2 is covered by rect1.
    Returns a fraction: intersection_area / area_of_rect2
    rect = (x, y, w, h)
    """
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2

    # Intersection coordinates
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)

    # Compute intersection area
    inter_width = max(0, xi2 - xi1)
    inter_height = max(0, yi2 - yi1)
    inter_area = inter_width * inter_height

    area_rect2 = w2 * h2
    if area_rect2 == 0:
        return 0
    return inter_area / area_rect2


def evaluate_detection_by_coverage(rectangles, yolo_rectangles, coverage_threshold=0.8):
    """
    rectangles: list of detected rectangles (x, y, w, h)
    yolo_rectangles: list of ground-truth rectangles
    coverage_threshold: fraction of YOLO rectangle that must be covered to count as TP

    Returns: TP, FP, FN counts
    """
    tp = 0
    matched_yolo = set()
    matched_detected = set()

    for j, yolo in enumerate(yolo_rectangles):
        for i, det in enumerate(rectangles):
            cov = coverage(det, yolo)
            if cov >= coverage_threshold:
                tp += 1
                matched_yolo.add(j)
                matched_detected.add(i)
                break  # stop after first matching rectangle

    fp = len(rectangles) - len(matched_detected)
    fn = len(yolo_rectangles) - len(matched_yolo)

    return tp, fp, fn


#set paths
image_folder = "/user/christoph.wald/u15287/big-scratch/02_splitted_data/train_labeled/images_uncropped"
image_files = os.listdir(image_folder)
pest_types = ["BRAIIM", "LIRIBO", "TRIAVA"]

gridcorners = np.array( [[ 146.,  576.],
       [5963.,  595.],
       [5857., 2865.],
       [ 201., 2858.]])

grown_mask = cv2.imread(
    "/user/christoph.wald/u15287/insect_pest_detection/image_processing/mask.jpg", 
    cv2.IMREAD_GRAYSCALE
)

for image_file in image_files[:5]:

    #Load image file
    print(f"Loading {image_file}.")
    image = cv2.imread(os.path.join(image_folder, image_file))
    
    #find YST contour 
    imageYST = find_contour(image)
    imagecorners = find_corners(image, imageYST)
    if len(imagecorners) == 0:
        continue
    imagecorners = order_corners(imagecorners)
    H, _ = cv2.findHomography(gridcorners, imagecorners, cv2.RANSAC)
    mask = cv2.warpPerspective(grown_mask, H, (image.shape[1], image.shape[0]))
    
    '''
    #second transformation to secure vertical alignment
    mask_h = get_h_mid(mask)
    image_h = get_h_mid(create_binary_mask(image))
    dy = get_midpoint(image_h)- get_midpoint(mask_h)
    H, W = mask.shape[:2]
    M = np.float32([[1, 0, 0], [0, 1, dy]])  # translation matrix
    shifted_mask= cv2.warpAffine(mask, M, (W, H), borderValue=255)  # white background
    reset to shifted_mask
    '''

    #replace black background with yellow (background color)
    yellow_mask = mask == 0
    image_wo_grid = image.copy()
    image_wo_grid[yellow_mask] = [0,255,255]
    
    x, y, w, h = cv2.boundingRect(imageYST)
    cropped_image_wo_grid = image_wo_grid[y:y+h, x:x+w]
    if "TRIAVA" in image_file:
        min_area = 100  # Define your minimum contour area
        max_area = 1000  # Define your maximum contour area
    else:
        min_area = 1000  # Define your minimum contour area
        max_area = 10000  # Define your maximum contour area
    rectangles= get_list_of_rectangles(cropped_image_wo_grid, min_area, max_area, scale = 2, max_ratio = 2)
    #only for testing from here
    labeled_image = cv2.imread(os.path.join("/user/christoph.wald/u15287/big-scratch/02_splitted_data/train_labeled/images_w_labels_uncropped", image_file))
    labeled_image_cropped = labeled_image[y:y+h, x:x+w]
    cv2.imwrite(os.path.join("/user/christoph.wald/u15287/big-scratch/02_splitted_data/train_labeled/test_segmentation", image_file),
                draw_bounding_boxes(labeled_image_cropped, rectangles))
    label_file = os.path.splitext(image_file)[0] + ".txt"
    label_path = "/user/christoph.wald/u15287/big-scratch/02_splitted_data/train_labeled/labels_uncropped"
    with open(os.path.join(label_path, label_file), "r") as f:
        yolo_labels = f.read().splitlines()
    yolo_rectangles = yolo_labels_to_rectangles(yolo_labels, labeled_image.shape)
    tp, fp, fn = evaluate_detection_by_coverage(rectangles, yolo_rectangles, coverage_threshold=0.8)



    print("True Positives:", tp)
    print("False Positives:", fp)
    print("False Negatives:", fn)

    '''
    tiles, _ = get_tiles_with_rectangles(cropped_image_wo_grid, rectangles) 
    cropped_image = image[y:y+h, x:x+w]
    padded_image = pad_image_to_stride(cropped_image)
    tile_images = extract_tiles(tiles, padded_image)
    class_id = [id in image_file for id in pest_types].index(True)
    labels = generate_yolo_labels(tiles, class_id=class_id)
    i = 0
    for tile, label in zip(tile_images, labels):
        filename = os.path.join("/user/christoph.wald/u15287/insect_pest_detection/registration/test", f"{image_file.rsplit('.',1)[0]}_tile{i:03d}.jpg")
        rectangles = yolo_labels_to_rectangles(label, tile.shape)
        cv2.imwrite(filename,draw_bounding_boxes(tile, rectangles))  
        i += 1
    '''