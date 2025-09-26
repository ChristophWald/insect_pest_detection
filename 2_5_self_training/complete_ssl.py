print("Importing modules.")
import os
import shutil
import random
import yaml
from ultralytics import YOLO
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import torch
import math
import numpy as np
import json
import cv2
from modules_complete import * #lazy solution

#do before starting:
#setup folder tiles/labels_training_run to write onto with all label files copied from labels
#make sure, that here is only "train" in the runs-folder

print("Initializing.")
tiles_folder = "/user/christoph.wald/u15287/big-scratch/02_splitted_data/train_labeled/SSL/tiles"
training_folder = "/user/christoph.wald/u15287/big-scratch/04_SSL_training_data/training_data"

training_runs = 10
empty_file_percentages = [0,5,5,0,0,0,0,0,5,0]
epochs = [10,10,10,10,10,10,10,10,10,20]
mosaic_epochs = [10,10,10,10,8,6,6,6,6,16]
thresholds = [
    {"BRAIIM": 0.85, "LIRIBO": 0.85, "FRANOC": 0, "TRIAVA": 0.75},  # run 0
    {"BRAIIM": 0.85, "LIRIBO": 0.85, "FRANOC": 0, "TRIAVA": 0.75},
    {"BRAIIM": 0.85, "LIRIBO": 0.85, "FRANOC": 0, "TRIAVA": 0.75},
    {"BRAIIM": 0.9, "LIRIBO": 0.9, "FRANOC": 0, "TRIAVA": 0.8},
    {"BRAIIM": 0.9, "LIRIBO": 0.9, "FRANOC": 0, "TRIAVA": 0.8},
    {"BRAIIM": 0.9, "LIRIBO": 0.9, "FRANOC": 0, "TRIAVA": 0.8},
        {"BRAIIM": 0.85, "LIRIBO": 0.85, "FRANOC": 0, "TRIAVA": 0.75}, 
           {"BRAIIM": 0.80, "LIRIBO": 0.80, "FRANOC": 0, "TRIAVA": 0.70},
              {"BRAIIM": 0.85, "LIRIBO": 0.85, "FRANOC": 0, "TRIAVA": 0.75},
                 {"BRAIIM": 0.9, "LIRIBO": 0.9, "FRANOC": 0, "TRIAVA": 0.8}   # run 1
]

for i in range(training_runs):
    
    print(f"Starting run {i+1}/{training_runs}")
    if i == 0:
        print("Creating the training set.")
    else:
        print("Actualizing the training set.")
        print(f"Adding {empty_file_percentages[i]}% empty tiles.")
    ########
    #copy the training set / the new labels into the training set)
    #######

    # Setup file structure for destination
    file_types = ["images", "labels"]
    use_types = ["train", "val"]

    if i == 0:
        for f in file_types:
            for u in use_types:
                path = os.path.join(training_folder, f, u)
                os.makedirs(path, exist_ok=True)

        data = {
            "train": os.path.join(training_folder, "images/train"),
            "val": os.path.join(training_folder, "images/val"),
            "nc": 4,
            "names": {
                0: "FungusGnats",
                1: "LeafMinerFlies",
                2: "Thrips",
                3: "WhiteFlies"
                }
        }

        with open(os.path.join(training_folder,"data.yaml"), "w") as f:
            yaml.dump(data, f, sort_keys=False)


    # Copy all tiles and labels from train/val source folders
    for u in use_types:

        print(f"Copying {u} data.")
        skipped = 0
        empty_files = []

        label_path = os.path.join(tiles_folder, u, "labels_training_run")
        img_path = os.path.join(tiles_folder, u, "images")
        label_dest_path = os.path.join(training_folder, "labels", u)
        img_dest_path = os.path.join(training_folder, "images", u)
        label_files = os.listdir(label_path)

        # First pass: copy non-empty files and track empty files
        for file in label_files:
            with open(os.path.join(label_path, file), "r") as f:
                if f.read().strip() == "":
                    skipped += 1
                    empty_files.append(file)
                else:
                    img_file = os.path.splitext(file)[0] + ".jpg"
                    img_src = os.path.join(img_path, img_file)
                    img_dest = os.path.join(img_dest_path, img_file)

                    # only copy if the image file isn’t already there
                    if not os.path.exists(img_dest):
                        shutil.copy2(img_src, img_dest)

                    # copy label every time (might have changed)
                    shutil.copy2(
                        os.path.join(label_path, file),
                        os.path.join(label_dest_path, file)
                    )

        total_files = len(label_files)
        copied_non_empty = total_files - skipped

        # Calculate number of empty files allowed
        max_empty_allowed = int(total_files * empty_file_percentages[i] / 100)
        empty_to_copy = min(max_empty_allowed, skipped)

        # Second pass: copy allowed empty files randomly
        copied_empty = 0
        random_empty_files = random.sample(empty_files, empty_to_copy)  # <-- random selection
        for file in random_empty_files:
            img_file = os.path.splitext(file)[0] + ".jpg"
            shutil.copy2(os.path.join(img_path, img_file), os.path.join(img_dest_path, img_file))
            shutil.copy2(os.path.join(label_path, file), os.path.join(label_dest_path, file))
            copied_empty += 1

        total_copied = copied_non_empty + copied_empty
        
        
    #####
    #training
    #####
    print("Starting the training.")
    print(f"{epochs[i]} epochs, {epochs[i]-mosaic_epochs[i]} with mosaic.")

    if i == 0:
        model = YOLO('/user/christoph.wald/u15287/insect_pest_detection/2_5_self_training/runs/detect/train/weights/best.pt')
    else:
        model = YOLO(f'/user/christoph.wald/u15287/insect_pest_detection/2_5_self_training/runs/detect/train{i+1}/weights/best.pt')
    model.train(data=os.path.join(training_folder,"data.yaml"),
                epochs=epochs[i],
                imgsz=640,
                close_mosaic=mosaic_epochs[i], #40% mosaic
                scale=0.3, #instead of 0.5
                mosaic= 0.25, #instead of 1.0
                mixup=0.05, #instead of 0.0
                erasing=0.4, #default (increase when oberving false positives)
                auto_augment="randaugment", #default, maybe try augmix
                )
      
    ####
    #evaluate
    ####

    print("Evaluate the training.")
    print("Plotting curves.")

    # Load the CSV file
    file_path = f'/user/christoph.wald/u15287/insect_pest_detection/2_5_self_training/runs/detect/train{i+2}/results.csv'
    df = pd.read_csv(file_path)

    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    # Left plot: Precision and Recall vs Epoch
    axs[0].plot(df['epoch'], df['metrics/precision(B)'], label='Precision (B)', marker='o')
    axs[0].plot(df['epoch'], df['metrics/recall(B)'], label='Recall (B)', marker='o')
    axs[0].plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP50 (B)', color='red', marker='o')
    axs[0].axhline(y=0.9, color='blue', linestyle=':', label='Precision Threshold')
    axs[0].axhline(y=0.8, color='orange', linestyle=':', label = "Recall Threshold")
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Score')
    axs[0].set_title('Precision and Recall vs Epoch')
    axs[0].legend()
    axs[0].grid(True)

    # Right plot: Training and Validation losses vs Epoch
    # Training losses
    axs[1].plot(df['epoch'], df['train/box_loss'], label='Train Box Loss', marker='o')
    axs[1].plot(df['epoch'], df['train/cls_loss'], label='Train Cls Loss', marker='o')
    #axs[1].plot(df['epoch'], df['train/dfl_loss'], label='Train DFL Loss', marker='o')

    # Validation losses (dashed lines)
    axs[1].plot(df['epoch'], df['val/box_loss'], label='Val Box Loss', linestyle='--', marker='o')
    axs[1].plot(df['epoch'], df['val/cls_loss'], label='Val Cls Loss', linestyle='--', marker='o')
    #axs[1].plot(df['epoch'], df['val/dfl_loss'], label='Val DFL Loss', linestyle='--', marker='o')

    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Loss')
    axs[1].set_title('Training and Validation Losses vs Epoch')
    axs[1].legend()
    axs[1].grid(True)

    # Adjust layout and save the figure
    plt.tight_layout()
    save_path = f'/user/christoph.wald/u15287/insect_pest_detection/2_5_self_training/runs/detect/train{i+2}/results.jpg'
    plt.savefig(save_path)
    plt.close()

    #####
    # make new predictions
    #####

    print("Predicting on the tils.")

    model = YOLO(f"/user/christoph.wald/u15287/insect_pest_detection/2_5_self_training/runs/detect/train{i+2}/weights/best.pt")

    #predict on cropped image contained in train/val
    image_dirs = ["/user/christoph.wald/u15287/big-scratch/02_splitted_data/train_labeled/SSL/split/images/train",
                "/user/christoph.wald/u15287/big-scratch/02_splitted_data/train_labeled/SSL/split/images/val" ]
    #compare to per tile labels
    label_dirs = ["/user/christoph.wald/u15287/big-scratch/02_splitted_data/train_labeled/SSL/tiles/train/labels", 
                "/user/christoph.wald/u15287/big-scratch/02_splitted_data/train_labeled/SSL/tiles/val/labels"]
    ### also possible: use the labels by segmentation info and recreate the tile labels afterwards

    output_dir = f"/user/christoph.wald/u15287/insect_pest_detection/2_5_self_training/predictions"
    os.makedirs(output_dir, exist_ok=True)

    #counts for boxes
    sum_predictions = 0
    sum_labels = 0
    sum_missing = 0
    sum_extra = 0

    #to collect infos on confidences
    conf_true_positives = []
    conf_new_labels = []

    #output
    new_labels = {}

    #cycles through train and val directories
    for image_dir, label_dir in zip(image_dirs, label_dirs):
        print(image_dir)
        #cycles through all images in a directory
        for filename in os.listdir(image_dir):
            image_path = os.path.join(image_dir, filename)
            image = cv2.imread(image_path)
            print(filename)
            
            #predicts on an image
            boxes, scores, classes = sliding_window_prediction(image, model, conf_threshold=0)
            if boxes:
                boxes, scores, classes = nms(boxes, scores, classes, iou_threshold=0.4)
                boxes, scores, classes = filter_mostly_contained_boxes(boxes, scores, classes, threshold=0.5)
            pred_tiles_data = get_labels_per_tile(image, boxes, classes, scores)
            #predictions are save as lists of tile
            #each tile is a list of labels, given as [class_id, x,y,w,h,conf]
            
            #loads given labels
            label_files = [f for f in os.listdir(label_dir) if f.startswith(os.path.splitext(filename)[0])]
            label_files = sorted(label_files, key=extract_tile_number)
            label_tiles_data = []
            for f in label_files:
                file_path = os.path.join(label_dir, f)
                with open(file_path, 'r') as file:
                    lines = file.read().splitlines()
                    tile_labels = [list(map(float, line.split())) for line in lines]
                    label_tiles_data.append(tile_labels)
            #same structure as above, but without conf

            #give number of labels/predictions
            total_pred_boxes = sum(len(tile_labels) for tile_labels in pred_tiles_data)
            total_gt_boxes = sum(len(tile_labels) for tile_labels in label_tiles_data)
            sum_predictions += total_pred_boxes
            sum_labels += total_gt_boxes
            print(f"Total predicted boxes: {total_pred_boxes}")
            print(f"Total ground-truth boxes: {total_gt_boxes}")


            missing_labels = []  # ground-truth not matched by predictions
            extra_preds = []     # predictions not matched by ground-truth

            #find missing labels and potential new labels
            for tile_id, (tile_pred, tile_lab) in enumerate(zip(pred_tiles_data, label_tiles_data)):

                # Check for missing labels
                for label in tile_lab:
                    _, xl, yl, wl, hl = label
                    matched = False
                    for pred in tile_pred:
                        _, x, y, w, h, *rest = pred  # allow extra info if exists
                        if compute_iop([x, y, w, h], [xl, yl, wl, hl]) > 0.8:
                            matched = True
                            break
                    if not matched:
                        missing_labels.append({'tile_id': tile_id, 'label': label})

                # Check for extra predictions
                for pred in tile_pred:
                    _, x, y, w, h, conf = pred
                    matched = False
                    for label in tile_lab:
                        _, xl, yl, wl, hl = label
                        if compute_iop([x, y, w, h], [xl, yl, wl, hl]) > 0.8:
                            matched = True
                            conf_true_positives.append(conf)
                            break
                    if not matched:
                        conf_new_labels.append(conf)
                        extra_preds.append({'tile_id': tile_id, 'prediction': pred})


            print(f"Total missing ground-truth labels: {len(missing_labels)}")
            print(f"Total extra predictions: {len(extra_preds)}")
            sum_missing += len(missing_labels)
            sum_extra += len(extra_preds)
            new_labels[filename] = extra_preds

    print("### Total statistics ###")
    print(f"{sum_labels} boxes were given.")
    print(f"{sum_predictions} boxes were predicted.")
    print(f"{sum_missing} labels were missed.")
    print(f"{sum_extra} new labels were found.")




    conf_true_positives = np.array(conf_true_positives)
    print(f"Mean conf of correct predictions {np.round(np.mean(conf_true_positives),2)} with standard deviation {np.round(np.std(conf_true_positives),2)}")
    conf_new_labels = np.array(conf_new_labels)
    print(f"Mean conf of new predictions {np.round(np.mean(conf_new_labels),2)} with standard deviation {np.round(np.std(conf_new_labels),2)}")

    with open(os.path.join(output_dir,f'predictions_{i+2}.json'), 'w') as f:
        json.dump(new_labels, f, indent=4)

    ###
    #suggested by ChatGPT
    ###

    # ---- Plot histograms of pseudo-label confidences ----
    plt.figure(figsize=(10,5))

    # True positives
    if len(conf_true_positives) > 0:
        plt.hist(conf_true_positives, bins=20, alpha=0.6, label='True Positive Confidences', color='green')
    else:
        print("No true positive confidences to plot.")

    # New predictions (pseudo-labels)
    if len(conf_new_labels) > 0:
        plt.hist(conf_new_labels, bins=20, alpha=0.6, label='New Pseudo-Label Confidences', color='orange')
    else:
        print("No new pseudo-label confidences to plot.")

    plt.title(f"Confidence Distribution for Run {i+1}")
    plt.xlabel("Confidence")
    plt.ylabel("Number of Boxes")
    plt.legend()
    plt.grid(True)

    # Save figure
    hist_save_path = f'/user/christoph.wald/u15287/insect_pest_detection/2_5_self_training/runs/detect/train{i+2}/confidence_histogram.jpg'
    plt.tight_layout()
    plt.savefig(hist_save_path)
    plt.close()

    #####
    # adding the labels
    ###

    print("Adding new labels to the tiles.")
    print("Using thresholds:")
    print(thresholds[i])

    folder_paths = [
        "/user/christoph.wald/u15287/big-scratch/02_splitted_data/train_labeled/SSL/tiles/train/labels",
        "/user/christoph.wald/u15287/big-scratch/02_splitted_data/train_labeled/SSL/tiles/val/labels"
    ]

    #make this a list 
    # threshold depends on species prefix
    #thresholds = {
    #    "BRAIIM": 0.8,
    #    "LIRIBO": 0.8,
    #    "FRANOC": 0,
    #    "TRIAVA": 0.7,
    #}
    species = list(thresholds[i].keys())

    # --- Flag to choose behavior ---
    # True: correct class ID according to filename
    # False: skip incorrect class predictions
    correct_class = True

    data = new_labels

    for folder_path in folder_paths:
        print(f"\nFolder {folder_path}")

        # --- Prepare output directory ---
        parent_dir = os.path.dirname(folder_path)
        out_dir = os.path.join(parent_dir, "labels_training_run")
        os.makedirs(out_dir, exist_ok=True)

        total_preds_appended = 0

        # --- Process files ---
        for filename in os.listdir(folder_path):

            in_path = os.path.join(folder_path, filename)
            out_path = os.path.join(out_dir, filename)

            # Extract base image name and tile id
            parts = filename.split("_tile_")
            base_name = parts[0] + ".jpg"
            tile_id = int(parts[1].split(".")[0])

            # Determine ground truth class from filename
            gt_class_str = next((s for s in species if filename.startswith(s)), None)
            gt_class_id = species.index(gt_class_str)

            # Pick threshold for this species
            threshold = thresholds[i][gt_class_str]

            # Default: copy file as-is
            shutil.copy(in_path, out_path)

            # Append predictions above threshold
            if base_name in data:
                for entry in data[base_name]:
                    if entry['tile_id'] != tile_id:
                        continue
                    pred = entry['prediction'][:5]
                    conf = entry['prediction'][5]

                    if conf < threshold:
                        continue

                    pred_class_id = pred[0]

                    if pred_class_id != gt_class_id:
                        if correct_class:
                            # Correct class according to filename
                            pred[0] = gt_class_id
                        else:
                            # Skip this prediction
                            continue

                    # Append to file
                    with open(out_path, "a") as f:
                        f.write(" ".join(map(str, pred)) + "\n")
                    total_preds_appended += 1

        print(f"  Total predictions appended: {total_preds_appended}")