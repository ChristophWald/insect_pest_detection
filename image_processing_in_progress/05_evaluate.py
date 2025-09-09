import cv2
import os
from modules_segmentation import *
from mask_corners import gridcorners
import pandas as pd
import time

start = time.time()

#set paths
image_folder = "/user/christoph.wald/u15287/big-scratch/02_splitted_data/train_labeled/images_uncropped"
image_files = os.listdir(image_folder)
#pest_types = ["BRAIIM", "LIRIBO", "TRIAVA"]
test_folder = "/user/christoph.wald/u15287/insect_pest_detection/image_processing_in_progress/test_segmentation_generated_mask_fat" \
""

#load mask
grown_mask = cv2.imread(
    "/user/christoph.wald/u15287/insect_pest_detection/2_4_image_processing/generated_mask_fat.jpg", 
    cv2.IMREAD_GRAYSCALE
)

inspection = False
save_label_files = True

#new
results = []
problems = []
predicted_rectangles = []

for i, image_file in enumerate(image_files):

    filename= image_file.split(".")[0]
    entry = []
    entry.append(filename)

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

    '''
    #first transformation
    mask = cv2.warpPerspective(grown_mask, H, (image.shape[1], image.shape[0]))
    if inspection: cv2.imwrite(os.path.join(test_folder, filename + "_02a_aligned_mask.jpg"), mask) 
    
    #second transformation
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
    
    '''
    
    #one combined transformation
    mask_h = get_h_mid(cv2.warpPerspective(grown_mask, H, (image.shape[1], image.shape[0])))
    image_h = get_h_mid(create_binary_mask(image))
    dy = get_midpoint(image_h)- get_midpoint(mask_h)
    # build combined transform
    T = np.array([[1, 0, 0],
                [0, 1, dy],
                [0, 0, 1]], dtype=np.float32)
    H_shifted = T @ H
    mask = cv2.warpPerspective(grown_mask, H_shifted, (image.shape[1], image.shape[0]))
    if inspection: cv2.imwrite(os.path.join(test_folder, filename + "_02_mask.jpg"), mask)
    

    #replace black background in image with yellow (background color) by using the mask
    yellow_mask = mask == 0 #mask or shifted_mask
    image_wo_grid = image.copy()
    image_wo_grid[yellow_mask] = [0,255,255]
    
    #crop the image
    x, y, w, h = cv2.boundingRect(imageYST)
    cropped_image_wo_grid = image_wo_grid[y:y+h, x:x+w]
    if inspection: cv2.imwrite(os.path.join(test_folder, filename + "_03_cropped_image.jpg"), cropped_image_wo_grid) 
    
    #handcrafted features for filtering bounding boxes
    if "TRIAVA" in image_file:
        min_area = 100  
        max_area = 1000 
    else:
        min_area = 1000 
        max_area = 10000 

    #find bounding boxes, filtered by handcrafted features and ratio of w/h, scale them    
    if inspection: cv2.imwrite(os.path.join(test_folder, filename + "_04_binary_mask.jpg"), create_binary_mask(cropped_image_wo_grid)) 
    
    rectangles= get_list_of_rectangles(cropped_image_wo_grid, min_area, max_area, scale = 1.5, max_ratio = 2)
    predicted_rectangles.append(rectangles)
    
    image_cropped = image[y:y+h, x:x+w]
    image_labels = draw_bounding_boxes(image_cropped, rectangles)
    
    #finding the rectangles given by the yolo labels
    label_file = os.path.splitext(image_file)[0] + ".txt"
    labels_path = "/user/christoph.wald/u15287/big-scratch/02_splitted_data/train_labeled/labels_uncropped"
    label_path = os.path.join(labels_path, label_file)
    with open(label_path, "r") as f:
        yolo_labels = f.read().splitlines()
    yolo_rectangles = yolo_labels_to_rectangles(yolo_labels, image.shape)
    cropped_yolo_rectangles = transform_rectangles_to_cropped(yolo_rectangles, x, y,  w,h)
    
    
    image_labels = draw_bounding_boxes(image_labels, cropped_yolo_rectangles, color = (0,255,0))
    if save_label_files: cv2.imwrite(os.path.join(test_folder, filename + "_05_w_labels.jpg"), image_labels)
    rectangles = [(x, y, x + w, y + h) for (x, y, w, h) in rectangles]
    cropped_yolo_rectangles = [(x, y, x + w, y + h) for (x, y, w, h) in cropped_yolo_rectangles]
    entry.append(evaluate_detections(rectangles, cropped_yolo_rectangles))
    print(entry)
    results.append(entry)
    

with open("rectangles.txt", "w") as f:
    for item in predicted_rectangles:
        f.write(str(item) + "\n")

#evaluate
rows = []
for name, metrics in results:
    row = {"image": name, **metrics}  # merge the dict with image name
    rows.append(row)

df = pd.DataFrame(rows)
df['prefix'] = df['image'].str[:6]  # first 6 chars like LIRIBO, BRAIIM, TRIAVA
df.to_csv("results_generated_mask_fat.csv")
# Sum TP, FP, FN over each prefix
grouped = df.groupby('prefix')[['TP', 'FP', 'FN']].sum().reset_index()
grouped['precision'] = grouped['TP'] / (grouped['TP'] + grouped['FP'])
grouped['recall'] = grouped['TP'] / (grouped['TP'] + grouped['FN'])

TP_total = df['TP'].sum()
FP_total = df['FP'].sum()
FN_total = df['FN'].sum()

precision_overall = TP_total / (TP_total + FP_total)
recall_overall = TP_total / (TP_total + FN_total)

print(grouped)
print("Overall precision:", precision_overall)
print("Overall recall:", recall_overall)
grouped.to_csv("metrics_generated_mask_fat.csv")
print("Problems")
for p in problems: 
    print(p)
print(f"Laufzeit {time.time() - start}")