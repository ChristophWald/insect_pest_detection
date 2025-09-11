import cv2
#import numpy as np
import os
from modules_segmentation import *
from mask_corners import gridcorners


#set paths
image_folder = "/user/christoph.wald/u15287/big-scratch/02_splitted_data/train_labeled/images_uncropped"
image_files = os.listdir(image_folder)
pest_types = ["BRAIIM", "LIRIBO", "TRIAVA"]
test_folder = "/user/christoph.wald/u15287/insect_pest_detection/image_processing_in_progress/test_segmentation"

#load mask
grown_mask = cv2.imread(
    "/user/christoph.wald/u15287/insect_pest_detection/2_4_image_processing/mask.jpg", 
    cv2.IMREAD_GRAYSCALE
)

inspection = True

for image_file in image_files[:5]:

    filename= image_file.split(".")[0]

    #Load image file
    print(f"Loading {image_file}.")
    image = cv2.imread(os.path.join(image_folder, image_file))
    if inspection: cv2.imwrite(os.path.join(test_folder, filename + "_original.jpg"), image)
    
    #find YST contour of image
    imageYST = find_contour(image)
    imagecorners = find_corners(image, imageYST)
    if len(imagecorners) == 0:
        print(f"No YST found for {image_file}")
        continue
    imagecorners = order_corners(imagecorners)

    #find transformation and apply to mask
    H, _ = cv2.findHomography(gridcorners, imagecorners, cv2.RANSAC)
    mask = cv2.warpPerspective(grown_mask, H, (image.shape[1], image.shape[0]))
    if inspection: cv2.imwrite(os.path.join(test_folder, filename + "_aligned_mask.jpg"), mask) 
    
    '''
    #second transformation to secure vertical alignment
    mask_h = get_h_mid(mask)
    image_h = get_h_mid(create_binary_mask(image))
    dy = get_midpoint(image_h)- get_midpoint(mask_h)
    if inspection:
        check_h_line(mask, mask_h)
        check_h_line(create_binary_mask(image), image_h)
    H, W = mask.shape[:2]
    M = np.float32([[1, 0, 0], [0, 1, dy]])  # translation matrix
    shifted_mask= cv2.warpAffine(mask, M, (W, H), borderValue=255)  # white background
    reset the following to shifted_mask
    if inspection: cv2.imwrite(os.path.join(test_folder, filename + "_shifted_mask.jpg")) 
    '''

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
    '''

    #replace black background in image with yellow (background color) by using the mask
    yellow_mask = mask == 0
    image_wo_grid = image.copy()
    image_wo_grid[yellow_mask] = [0,255,255]
    if inspection: cv2.imwrite(os.path.join(test_folder, filename + "_wo_grid.jpg"), image_wo_grid) 
    
    #crop the image
    x, y, w, h = cv2.boundingRect(imageYST)
    cropped_image_wo_grid = image_wo_grid[y:y+h, x:x+w]
    if inspection: cv2.imwrite(os.path.join(test_folder, filename + "_cropped.jpg"), cropped_image_wo_grid) 
    
    #handcrafted features for filtering bounding boxes
    if "TRIAVA" in image_file:
        min_area = 100  
        max_area = 1000 
    else:
        min_area = 1000 
        max_area = 10000 

    #find bounding boxes, filtered by handcrafted features and ratio of w/h, scale them    
    if inspection: cv2.imwrite(os.path.join(test_folder, filename + "_binary_mask.jpg"), create_binary_mask(cropped_image_wo_grid)) 
    rectangles= get_list_of_rectangles(cropped_image_wo_grid, min_area, max_area, scale = 2, max_ratio = 2)

    #only for testing: draw the found boxes into the labeled image
    labeled_image = cv2.imread(os.path.join("/user/christoph.wald/u15287/big-scratch/02_splitted_data/train_labeled/images_w_labels_uncropped", image_file))
    labeled_image_cropped = labeled_image[y:y+h, x:x+w]
    #cv2.imwrite(os.path.join("/user/christoph.wald/u15287/big-scratch/02_splitted_data/train_labeled/test_segmentation", image_file),
    #            draw_bounding_boxes(labeled_image_cropped, rectangles))
    cv2.imwrite(os.path.join(test_folder, image_file),
                draw_bounding_boxes(labeled_image_cropped, rectangles))
    
    
    #finding the rectangles given by the yolo labels
    label_file = os.path.splitext(image_file)[0] + ".txt"
    label_path = "/user/christoph.wald/u15287/big-scratch/02_splitted_data/train_labeled/labels_uncropped"
    with open(os.path.join(label_path, label_file), "r") as f:
        yolo_labels = f.read().splitlines()
    yolo_rectangles = yolo_labels_to_rectangles(yolo_labels, labeled_image.shape)
    #todo: compare them


    #create the tiles and labels for training
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