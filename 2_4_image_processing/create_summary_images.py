import sys
sys.path.append("/user/christoph.wald/u15287/insect_pest_detection/modules")
from modules_segmentation import *
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import random

#choose a random image from the processed images
#images_masked = "/user/christoph.wald/u15287/big-scratch/02_splitted_data/train_labeled/SSL/03_images_masked"
images_masked = "/user/christoph.wald/u15287/big-scratch/02_splitted_data/test_set/SSL/03_images_masked"
file_list = os.listdir(images_masked)

#image_file = random.choice(file_list)
#images_folder = "/user/christoph.wald/u15287/big-scratch/02_splitted_data/train_labeled/SSL/04_images_cropped" 
#labels_folder ="/user/christoph.wald/u15287/big-scratch/02_splitted_data/train_labeled/SSL/04_labels_cropped"
images_folder = "/user/christoph.wald/u15287/big-scratch/02_splitted_data/test_set/SSL/04_images_cropped" 
labels_folder ="/user/christoph.wald/u15287/big-scratch/02_splitted_data/test_set/SSL/04_labels_cropped"


for image_file in file_list[:2]:
    image = cv2.imread(os.path.join(images_folder, image_file))


    #1 draw the ground truth
    label_file = os.path.splitext(image_file)[0] + ".txt"
    label_path = os.path.join(labels_folder, label_file)
    with open(label_path, "r") as f:
        yolo_rectangles = [
            tuple(map(int, line.strip("()\n ").split(",")))
            for line in f
        ]
    img_gt = draw_bounding_boxes(image, yolo_rectangles, color = (0,255,0))

    #2 draw contours between min and max size
    #its possible to show the difference between processing the masked and unmasked image
    image_masked = cv2.imread(os.path.join(images_masked, image_file))
    if "TRIAVA" in image_file:
        min_area_contour = 100 
        max_area_contour = 2000
        binary_default = False
    elif "LIRIBO" in image_file: 
        min_area_contour = 1000 
        max_area_contour = 10000 
        binary_default = True
    elif "BRAIIM" in image_file:
        min_area_contour = 2000 
        max_area_contour = 10000
        binary_default = True

    inverted_mask = cv2.bitwise_not(create_binary_mask(image_masked, binary_default), )
    contours, _ = cv2.findContours(inverted_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    img_contours = cv2.cvtColor(inverted_mask, cv2.COLOR_GRAY2BGR)

    for i, c in enumerate(contours):
        if cv2.contourArea(c) > min_area_contour and cv2.contourArea(c) < max_area_contour:
            # Draw contour
            cv2.drawContours(img_contours, [c], -1, (255,100,0), -1)

            # Compute contour center (centroid)
            M = cv2.moments(c)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = 0, 0

            # Label with contour index
            #cv2.putText(img_labeled, str(i), (cx, cy),
            #            cv2.FONT_HERSHEY_SIMPLEX, 5, (255,255,255), 2)

    #3 draw rectangles as result from full image processing
    #its possible to show the difference between processing the masked and unmasked image
    if "TRIAVA" in image_file:
        min_area_contour = 100 
        max_area_contour = 2000
        scale = 1.5
        max_ratio = 2 
        upper_limit_rectangles = None
        value_threshold =97 # 5th percentile
        binary_default = False
    elif "LIRIBO" in image_file: 
        min_area_contour = 1000 
        max_area_contour = 10000 
        scale = 1.5
        max_ratio = 1.76 #95th percentile
        upper_limit_rectangles = 28530 #22340 #95th percentile
        value_threshold = None
        binary_default = True
    elif "BRAIIM" in image_file:
        min_area_contour = 2000 
        max_area_contour = 10000
        scale = 1.5
        max_ratio = 1.73 #1.75 #95the percentile
        upper_limit_rectangles = 42970 #41703 #95th percentil
        value_threshold = None
        binary_default = True

    rectangles, v = get_list_of_rectangles(image_masked, min_area_contour, max_area_contour, scale, max_ratio, upper_limit_rectangles, value_threshold, binary_default)

    if "TRIAVA" in image_file:
        rectangles = remove_smaller_overlaps(rectangles)

    img_rectangles = draw_bounding_boxes(image, rectangles)

    #4 predictions
    from modules_prediction_copy import *

    model = YOLO(f"/user/christoph.wald/u15287/insect_pest_detection/2_5_self_training/runs/detect/train9/weights/best.pt")
    boxes, confs, class_ids = sliding_window_prediction(image, model)
                
    if boxes.numel() > 0:
        boxes, confs, class_ids = nms(boxes, confs, class_ids, iou_threshold=0.4, device=model.device)
        boxes, confs, class_ids = filter_mostly_contained_boxes(boxes, confs, class_ids, threshold=0.5)
        #print(f"Predicted: {boxes.size(0)}")

    yolo_rectangles = [(x, y, x + w, y + h) for (x, y, w, h) in yolo_rectangles]
    label_boxes = torch.tensor(yolo_rectangles, dtype= torch.float32).to("cuda")
    species = ["BRAIIM", "LIRIBO","FRANOC", "TRIAVA"]  # adjust as needed
    row_index = next((i for i, sp in enumerate(species) if image_file.startswith(sp)), len(species))
    label_classes = torch.tensor([row_index]*len(label_boxes), dtype=torch.long).to("cuda")
    
    tp, fp, fn = compare_labels_vectorized(boxes, class_ids, confs, label_boxes, label_classes,
                                                tile_size = 640, iou_threshold=0.5, containment_threshold=0.8, 
                                                convert_to_xyxy=False)

    img_predictions = image.copy()
    for boxes, color in zip([tp[0], fp[0], fn[0]],[(0, 255, 0), (255, 0, 0),  (0, 0, 255)]):
        boxes_xywh = []
        for x1, y1, x2, y2 in boxes:
            x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
            boxes_xywh.append([x, y, w, h])
            img_predictions = draw_bounding_boxes(img_predictions, boxes_xywh, color)
    #cv2.imwrite("test.jpg", img_predictions)




    #save image
    def resize_to(img, size):
        return cv2.resize(img, (size[1], size[0]))

    target_size = image.shape[:2]
    img_gt = resize_to(img_gt, target_size)
    img_contours = resize_to(img_contours, target_size)
    img_predictions = resize_to(img_predictions, target_size)
    img_rectangles = resize_to(img_rectangles, target_size)

    # Optional: add titles to each panel
    def add_label(img, text):
        labeled = img.copy()
        cv2.putText(labeled, text, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 4, cv2.LINE_AA)
        return labeled

    img_gt = add_label(img_gt, "Ground Truth")
    img_contours = add_label(img_contours, "Contours")
    img_predictions = add_label(img_predictions, "Predictions")
    img_rectangles = add_label(img_rectangles, "Rectangles")

    top = np.hstack((img_gt, img_contours))
    bottom = np.hstack((img_predictions, img_rectangles))
    summary = np.vstack((top, bottom))




    # --- Save final combined image only ---
    cv2.imwrite(os.path.join("/user/christoph.wald/u15287/insect_pest_detection/2_4_image_processing/summary_images", 
                             f"summary_grid_{os.path.splitext(image_file)[0]}.jpg"), 
                             summary)