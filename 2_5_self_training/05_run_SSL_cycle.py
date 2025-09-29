print("Importing modules.")
import os
import sys
#import shutil
#import random #for adding empty tiles
#import matplotlib.pyplot as plt #for histogram plot
#import cv2 #for tile prediction
#import numpy as np #for calculation of new predictions confidence std/mean
import json #for saving predictions
from ultralytics import YOLO
from modules_prediction import * #for tile prediction


print("Initializing.")
#make sure, that here is only "train" in the runs-folder
if set(os.listdir("/user/christoph.wald/u15287/insect_pest_detection/2_5_self_training/runs/detect")) != {"train"}:
    sys.exit(f"Error: Directory {base_path} contains {items}, but only {allowed} is allowed.")

training_runs = 1

#Main training loop
for i in range(1, training_runs+1):
    
    print(f"Starting run {i}/{training_runs}")

    #####
    # make new predictions
    #####

    print("Predicting on the tiles.")
    if i == 1:
        model = YOLO(f"/user/christoph.wald/u15287/insect_pest_detection/2_5_self_training/runs/detect/train/weights/best.pt")
    else: 
        model = YOLO(f"/user/christoph.wald/u15287/insect_pest_detection/2_5_self_training/runs/detect/train{i}/weights/best.pt")

    # full images folder (predicting on cropped image contained in train/val)
    image_dirs = [
        "/user/christoph.wald/u15287/big-scratch/02_splitted_data/train_labeled/SSL/split/images/train",
        "/user/christoph.wald/u15287/big-scratch/02_splitted_data/train_labeled/SSL/split/images/val"
    ]

    # tile labels folder
    label_dirs = [
        "/user/christoph.wald/u15287/big-scratch/02_splitted_data/train_labeled/SSL/tiles/train/labels",
        "/user/christoph.wald/u15287/big-scratch/02_splitted_data/train_labeled/SSL/tiles/val/labels"
    ]
    
    #output folder for predictions
    output_dir = f"/user/christoph.wald/u15287/insect_pest_detection/2_5_self_training/predictions"
    os.makedirs(output_dir, exist_ok=True)

    # output dict structured by FN/FP/TP -> species -> image
    results = {"FN": {}, "FP": {}, "TP": {}}

    # cycles through train and val directories
    for image_dir, label_dir in zip(image_dirs, label_dirs):

        # cycles through all images in a directory
        for filename in os.listdir(image_dir)[:1]:
            image_path = os.path.join(image_dir, filename)
            image = cv2.imread(image_path)

            # predicts on the full images (with stride 420)
            boxes, confs, class_ids = sliding_window_prediction(image, model)
            print(f"Number of predicted boxes after thresholding: {boxes.size(0)}")
            
            
            if boxes.numel() > 0:
                boxes, confs, class_ids = nms(boxes, confs, class_ids, iou_threshold=0.4, device=model.device)
                print(f"Number of predicted boxes after NMS: {boxes.size(0)}")
                boxes, confs, class_ids = filter_mostly_contained_boxes(boxes, confs, class_ids, threshold=0.5)
                print(f"Number of predicted boxes after removing contained boxes: {boxes.size(0)}")