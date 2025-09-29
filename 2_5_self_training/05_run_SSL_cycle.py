print("Importing modules.")
import os
import sys
#import shutil
#import random #for adding empty tiles
#import matplotlib.pyplot as plt #for histogram plot
#import cv2 #for tile prediction
#import numpy as np #for calculation of new predictions confidence std/mean
#import json #for saving predictions
#from ultralytics import YOLO
#from modules_prediction import * #for tile prediction


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
        for filename in os.listdir(image_dir):
            image_path = os.path.join(image_dir, filename)
            image = cv2.imread(image_path)


####################################

            # predicts on the full images (with stride 420)
            boxes, scores, classes = sliding_window_prediction(image, model, conf_threshold=0)
            if boxes:
                boxes, scores, classes = nms(boxes, scores, classes, iou_threshold=0.4)
                boxes, scores, classes = filter_mostly_contained_boxes(boxes, scores, classes, threshold=0.5)
            #remapping the labels to the tiles (with stride 440)
            pred_tiles_data = get_labels_per_tile(image, boxes, classes, scores)
            #question: what about overlaps?

            # loads given labels
            label_files = [f for f in os.listdir(label_dir) if f.startswith(os.path.splitext(filename)[0])]
            label_files = sorted(label_files, key=extract_tile_number)
            label_tiles_data = []
            for f in label_files:
                file_path = os.path.join(label_dir, f)
                with open(file_path, 'r') as file:
                    lines = file.read().splitlines()
                    tile_labels = [list(map(float, line.split())) for line in lines]
                    label_tiles_data.append(tile_labels)

            # collect FN, FP, TP
            missing_labels = []  # FN
            extra_preds = []     # FP
            true_positives = []  # TP

            for tile_id, (tile_pred, tile_lab) in enumerate(zip(pred_tiles_data, label_tiles_data)):

                # Check for missing labels (FN)
                for label in tile_lab:
                    _, xl, yl, wl, hl = label
                    matched = False
                    for pred in tile_pred:
                        _, x, y, w, h, *rest = pred
                        if compute_iop([x, y, w, h], [xl, yl, wl, hl]) > 0.8:
                            matched = True
                            break
                    if not matched:
                        missing_labels.append({"tile_id": tile_id, "label": label})

                # Check for extra predictions and true positives
                for pred in tile_pred:
                    _, x, y, w, h, conf = pred
                    matched = False
                    for label in tile_lab:
                        _, xl, yl, wl, hl = label
                        if compute_iop([x, y, w, h], [xl, yl, wl, hl]) > 0.8:
                            matched = True
                            true_positives.append({"tile_id": tile_id, "prediction": pred})
                            break
                    if not matched:
                        extra_preds.append({"tile_id": tile_id, "prediction": pred})

            # determine species name (prefix before first underscore)
            species = filename.split("_")[0]

            # store results grouped by FN / FP / TP -> species -> image
            if missing_labels:
                results["FN"].setdefault(species, {})[filename] = missing_labels
            if extra_preds:
                results["FP"].setdefault(species, {})[filename] = extra_preds
            if true_positives:
                results["TP"].setdefault(species, {})[filename] = true_positives

    # save JSON
    with open(os.path.join(output_dir, f'predictions_{i+2}.json'), 'w') as f:
        json.dump(results, f, indent=4)