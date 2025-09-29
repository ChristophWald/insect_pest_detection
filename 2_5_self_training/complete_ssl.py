print("Importing modules.")
import os
import shutil
import sys
import random #for adding empty tiles
import matplotlib.pyplot as plt #for histogram plot
import cv2 #for tile prediction
import numpy as np #for calculation of new predictions confidence std/mean
import json #for saving predictions
from ultralytics import YOLO
from modules_prediction import * #for tile prediction


print("Initializing.")




tiles_folder = "/user/christoph.wald/u15287/big-scratch/02_splitted_data/train_labeled/SSL/tiles"
training_folder = "/user/christoph.wald/u15287/big-scratch/04_SSL_training_data/training_data"

training_runs = 1
empty_file_percentages = [0,5,5,0,0,0,0,0,5,0]
epochs = [1,10,10,10,10,10,10,10,10,20]
thresholds = [
    {"BRAIIM": 0.85, "LIRIBO": 0.85, "FRANOC": 0, "TRIAVA": 0.75},  
    {"BRAIIM": 0.85, "LIRIBO": 0.85, "FRANOC": 0, "TRIAVA": 0.75},
    {"BRAIIM": 0.85, "LIRIBO": 0.85, "FRANOC": 0, "TRIAVA": 0.75},
    {"BRAIIM": 0.9, "LIRIBO": 0.9, "FRANOC": 0, "TRIAVA": 0.8},
    {"BRAIIM": 0.9, "LIRIBO": 0.9, "FRANOC": 0, "TRIAVA": 0.8},
    {"BRAIIM": 0.9, "LIRIBO": 0.9, "FRANOC": 0, "TRIAVA": 0.8},
    {"BRAIIM": 0.85, "LIRIBO": 0.85, "FRANOC": 0, "TRIAVA": 0.75}, 
    {"BRAIIM": 0.80, "LIRIBO": 0.80, "FRANOC": 0, "TRIAVA": 0.70},
    {"BRAIIM": 0.85, "LIRIBO": 0.85, "FRANOC": 0, "TRIAVA": 0.75},
    {"BRAIIM": 0.9, "LIRIBO": 0.9, "FRANOC": 0, "TRIAVA": 0.8}   
]

for i in range(training_runs):
    
    print(f"Starting run {i+1}/{training_runs}")
    
    if i == 0:
        print("Creating the training set.")
    else:
        print("Actualizing the training set.")
        print(f"Adding {empty_file_percentages[i]}% empty tiles.")
    
    ########
    #create or update the new yolo training set
    ########

    # Setup file structure for destination
    file_types = ["images", "labels"]
    use_types = ["train", "val"]

    #create folders and data.yaml first time
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
        
       
    #######
    #training
    #######
    print("Starting the training.")
    print(f"{epochs[i]} epochs")

    #always load the model from before
    if i == 0:
        model = YOLO('/user/christoph.wald/u15287/insect_pest_detection/2_5_self_training/runs/detect/train/weights/best.pt')
    else:
        model = YOLO(f'/user/christoph.wald/u15287/insect_pest_detection/2_5_self_training/runs/detect/train{i+1}/weights/best.pt')
    model.train(data=os.path.join(training_folder,"data.yaml"),
                epochs=epochs[i],
                imgsz=640,
                close_mosaic=epochs[i], #no mosaic
                scale=0.3, #instead of 0.5
                mosaic= 0.25, #instead of 1.0
                mixup=0.05, #instead of 0.0
                erasing=0.4, #default (increase when oberving false positives)
                auto_augment="randaugment", #default, maybe try augmix
                )
      

       
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

      species = list(thresholds[i].keys())

    # --- Flag to choose behavior ---
    # True: correct class ID according to filename
    # False: skip incorrect class predictions
    correct_class = True

    data = new_labels

    for folder_path in folder_paths:
        #print(f"\nFolder {folder_path}")

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

    

from plot_prec_recall_combined import *


plot_prec_recall()

from plot_prediction_histograms import * 

plot_histograms()

