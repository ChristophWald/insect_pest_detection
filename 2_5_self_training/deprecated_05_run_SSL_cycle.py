print("Importing modules.")
import sys
sys.path.append("/user/christoph.wald/u15287/insect_pest_detection/modules")
import os
#for tile predicition
from ultralytics import YOLO
import cv2 
import torch
import math
from modules_prediction import * 
import json
import shutil
import random #for adding empty tiles
import time
from modules_testing import check_data_yaml, delete_cache_files

print("Initializing.")
#make sure, that there is only "train" in the runs-folder
if set(os.listdir("/user/christoph.wald/u15287/insect_pest_detection/2_5_self_training/runs/detect")) != {"train"}:
    sys.exit(f"Error: Training directory contains already train.")


#for metrics
output_folder = "/user/christoph.wald/u15287/insect_pest_detection/2_5_self_training/metrics_ssl"
os.makedirs(output_folder, exist_ok=True)

#folder with images and labels in tiles to write onto and to copy from
tiles_folder = "/user/christoph.wald/u15287/big-scratch/02_splitted_data/train_labeled/SSL/tiles"
#folder with images and labels in yolo-format
training_folder = "/user/christoph.wald/u15287/big-scratch/04_SSL_training_data/training_data"

check_data_yaml(os.path.join(training_folder, "data.yaml"))
delete_cache_files(os.path.join(training_folder, "labels"))

training_runs = 3
thresholds = [
    {"BRAIIM": 0.4, "LIRIBO": 0.4, "FRANOC": 0, "TRIAVA": 0.4}  ,
     {"BRAIIM": 0.4, "LIRIBO": 0.4, "FRANOC": 0, "TRIAVA": 0.4}  , 
     {"BRAIIM": 0.4, "LIRIBO": 0.4, "FRANOC": 0, "TRIAVA": 0.4}  
      
]
empty_file_percentages = [0,0,0]
epochs = [5,5,10]

#Main training loop
for i in range(1,4):
    

    print(f"Starting run {i}/{training_runs}")
    #####
    # make new predictions
    #####
    start = time.time()

    print("Predicting on the tiles.")
    if i == 1:
        model_path = f"/user/christoph.wald/u15287/insect_pest_detection/2_5_self_training/runs/detect/train/weights/best.pt"
    else:
        #for weighted training
        #model_path = f"/user/christoph.wald/u15287/ultralytics/runs/detect/train{i}/weights/best.pt" 
        model_path = f"/user/christoph.wald/u15287/insect_pest_detection/2_5_self_training/runs/detect/train{i}/weights/best.pt"
    
    if not os.path.exists(model_path):
        print(model_path)
        sys.exit("Model path not correct.")

    model = YOLO(model_path)

    # full images folder (predicting on cropped image contained in train/val)
    image_dir = "/user/christoph.wald/u15287/big-scratch/02_splitted_data/train_labeled/SSL/split/images/train"
    
    # tile labels folder
    label_dir = "/user/christoph.wald/u15287/big-scratch/02_splitted_data/train_labeled/SSL/tiles/train/labels"
    
    #output folder for predictions
    output_dir = f"/user/christoph.wald/u15287/insect_pest_detection/2_5_self_training/predictions"
    os.makedirs(output_dir, exist_ok=True)

    # output dict structured by FN/FP/TP -> species -> image
    json_results = {"FN": {}, "FP": {}, "TP": {}}

    # cycles through all images in a directory
    for filename in os.listdir(image_dir):
        print(f"Predicting on {filename}.")
        image_path = os.path.join(image_dir, filename)
        image = cv2.imread(image_path)

        # predicts on the full images (with stride 420)
        boxes, confs, class_ids = sliding_window_prediction(image, model)
        
        
        
        if boxes.numel() > 0:
            boxes, confs, class_ids = nms(boxes, confs, class_ids, iou_threshold=0.4, device=model.device)
            
            boxes, confs, class_ids = filter_mostly_contained_boxes(boxes, confs, class_ids, threshold=0.5)
            #print(f"Predicted: {boxes.size(0)}")
        
        pred_tiles_data = get_labels_per_tile_tensor(image, boxes, class_ids, confs)
        label_tiles_data = load_label_tiles(label_dir, filename)

        results = []

        # This is the per-tile loop:
        for tile_id, (pred, label) in enumerate(zip(pred_tiles_data, label_tiles_data)):
            # pred = predictions for tile i
            # label = labels for tile i

            if pred.numel() == 0:
                pred_boxes = torch.empty((0, 4), device='cuda')
                pred_classes = torch.empty((0,), dtype=torch.long, device='cuda')
                pred_scores = torch.empty((0,), device='cuda')
            else:
                pred_boxes = pred[:, 1:5]
                pred_classes = pred[:, 0].long()
                pred_scores = pred[:, 5]

            if label.numel() == 0:
                gt_boxes = torch.empty((0, 4), device='cuda')
                gt_classes = torch.empty((0,), dtype=torch.long, device='cuda')
            else:
                gt_boxes = label[:, 1:5]
                gt_classes = label[:, 0].long()

            tp, fp, fn = compare_labels_vectorized(
                pred_boxes, pred_classes, pred_scores, gt_boxes, gt_classes
            )
            species = filename.split("_")[0]  # extract species from filename

            # --- Add entries to JSON with tile_id ---
            for category, items in zip(["TP", "FP", "FN"], [tp, fp, fn]):
                boxes, classes, scores = items if category != "FN" else (*items, [None]*len(items[0]))
                if len(classes) > 0:
                    json_results.setdefault(category, {}).setdefault(species, {}).setdefault(filename, [])
                    for cls, box, score in zip(classes, boxes, scores):
                        entry = {"tile_id": tile_id}
                        if category != "FN":
                            entry["prediction"] = [cls, *box, score]
                        else:
                            entry["prediction"] = [cls, *box]
                        #print(category,entry)
                        json_results[category][species][filename].append(entry)

    with open(os.path.join(output_dir, f'predictions{i}.json'), 'w') as f:
        json.dump(json_results, f, indent=4)

    end = time.time()
    print(f"Predicting took {end-start:.2f} seconds.")
    start = end


    ####
    # Adding newfound labels (FP) and update the yolo training set
    ####

    print("Adding new labels to the tiles (in-place).")
    print("Using thresholds:")
    print(thresholds[i-1])

    #folder with images and labels in tiles to write onto and to copy from
    tiles_folder = "/user/christoph.wald/u15287/big-scratch/02_splitted_data/train_labeled/SSL/tiles"
    #   folder with images and labels in yolo-format
    training_folder = "/user/christoph.wald/u15287/big-scratch/04_SSL_training_data/training_data"


        

    # Only FP predictions from the JSON
    data = json_results["FP"]

    total_preds_appended = 0

    add_weights = True
    correct_labels = True
    corrections = []

    fp_log_path = os.path.join(output_folder, f"fp_labels_added_run{i}.txt")
    fp_log = open(fp_log_path, "w")
    fp_log.write("species,base_name,tile_id,class_id,conf,xyxy,yolo\n")


    for species_name, images in data.items():
        for base_name, entries in images.items():

            # Determine if the image belongs to train or val
            #if base_name in os.listdir("/user/christoph.wald/u15287/big-scratch/02_splitted_data/train_labeled/SSL/split/images/train"):
            #    file_usage = "train"
            #elif base_name in os.listdir("/user/christoph.wald/u15287/big-scratch/02_splitted_data/train_labeled/SSL/split/images/val"):
            #    file_usage = "val"
            file_usage = "train"

            for entry in entries:
                tile_id = entry['tile_id']
                pred = entry['prediction']  # [class_id, x1, y1, x2, y2, conf]

                # Unpack prediction
                class_id, x1, y1, x2, y2, conf = pred

                # Apply threshold
                threshold = thresholds[i-1][species_name]
                if conf < threshold:
                    continue

                # Correct class according to filename if needed
                gt_class_id = list(thresholds[i-1].keys()).index(species_name)
                if correct_labels and class_id != gt_class_id:
                    corrections.append([base_name, tile_id, pred, class_id, gt_class_id])
                    class_id = gt_class_id  # enforce correct class
                    

                # Path to the existing tile label file
                base_filename = os.path.splitext(base_name)[0]  # e.g., "LIRIBO_0629"
                full_filename = f"{base_filename}_tile_{tile_id}.txt"
                tile_file = os.path.join(tiles_folder,file_usage,"labels_training_run",full_filename)
                label_file = os.path.join(training_folder, "labels", file_usage,full_filename)

                # Append the FP prediction directly
                yolo_box = xyxy_to_yolo(x1, y1, x2, y2, tile_size=640)
                #print(f"Writing to {tile_file} and {label_file}")
                if os.path.exists(tile_file):
                    is_empty = os.path.getsize(tile_file) == 0
                else:
                    is_empty = True

                if add_weights:
                    line = f"{class_id} {yolo_box[0]} {yolo_box[1]} {yolo_box[2]} {yolo_box[3]} {conf}\n"
                else:
                    line = f"{class_id} {yolo_box[0]} {yolo_box[1]} {yolo_box[2]} {yolo_box[3]}\n"

                with open(tile_file, "a") as f:
                    f.write(line)
                with open(label_file, "a") as f:
                    f.write(line)
                
                if is_empty:
                    src_image = os.path.join(tiles_folder,file_usage, "images", os.path.splitext(full_filename)[0]+".jpg")
                    dest_image = os.path.join(training_folder, "images", file_usage, os.path.splitext(full_filename)[0]+".jpg")
                    shutil.copy(src_image, dest_image)

                if i == 1 and add_weights:
                    with open(label_file, "r") as f:
                        existing_lines = [l.strip().split() for l in f.readlines() if l.strip()]
                    normalized_lines = []
                    for line in existing_lines:
                        if len(line) == 5:
                            line.append("1.0")  # add default weight
                        normalized_lines.append(" ".join(line))
                    with open(label_file, "w") as f:
                        f.write("\n".join(normalized_lines) + "\n")

                # ---- LOG HERE ----
                fp_log.write(
                    f"{species_name},{base_name},{tile_id},{class_id},{conf:.3f},"
                    f"[{x1},{y1},{x2},{y2}],[{yolo_box[0]:.4f},{yolo_box[1]:.4f},{yolo_box[2]:.4f},{yolo_box[3]:.4f}]\n"
                )
                total_preds_appended += 1
                #print(base_name, tile_id)

    with open(os.path.join(output_folder, f"class_corrections{i}.txt"), "w") as f:
    # First line: variable names
        f.write("base_name,tile_id,pred,class_id,gt_class_id\n")
        for entry in corrections:
            # Convert pred list to string to fit in one column
            pred_str = "[" + ",".join(map(str, entry[2])) + "]"
            f.write(f"{entry[0]},{entry[1]},{pred_str},{entry[3]},{entry[4]}\n")
    
    fp_log.close()

    print(f"Total FP predictions appended: {total_preds_appended}")

    end = time.time()
    print(f"Adding the new labels took {end-start:.2f} seconds.")
    start = end

    ########
    # Optional adding of empty files
    ########

    if empty_file_percentages[i-1] == 0:
        print("Skipping empty tiles addition (0%).")
    else:
        print(f"Adding {empty_file_percentages[i-1]}% empty tiles.")
        
        # Setup file structure for destination
        file_types = ["images", "labels"]
        use_types = ["train"] #only for train

        for u in use_types:

            empty_files = []

            label_path = os.path.join(tiles_folder, u, "labels_training_run")
            img_path = os.path.join(tiles_folder, u, "images")
            label_dest_path = os.path.join(training_folder, "labels", u)
            img_dest_path = os.path.join(training_folder, "images", u)
            os.makedirs(label_dest_path, exist_ok=True)
            os.makedirs(img_dest_path, exist_ok=True)

            label_files = os.listdir(label_path)

            for file in label_files:
                if os.path.getsize(os.path.join(label_path, file)) == 0:
                    empty_files.append(file)

            # Calculate number of empty files allowed
            max_empty_allowed = int(len(label_files) * empty_file_percentages[i-1] / 100)
            empty_to_copy = min(max_empty_allowed, len(empty_files))
            print(f"Copying {empty_to_copy} files.")

            if empty_to_copy > 0:
                random_empty_files = random.sample(empty_files, empty_to_copy)  # random selection
                for file in random_empty_files:
                    img_file = os.path.splitext(file)[0] + ".jpg"
                    #print(img_file)
                    shutil.copy2(os.path.join(img_path, img_file), os.path.join(img_dest_path, img_file))
                    shutil.copy2(os.path.join(label_path, file), os.path.join(label_dest_path, file))

    end = time.time()
    print(f"Adding empty files took {end-start:.2f} seconds.")
    start = end

    #######
    #training
    #######


    print("Starting the training.")
    print(f"{epochs[i-1]} epochs")

    #always load the model from before
    if i == 1:
        model = YOLO('/user/christoph.wald/u15287/insect_pest_detection/2_5_self_training/runs/detect/train/weights/best.pt')
    else:
        #for weighted training
        model = YOLO(f"/user/christoph.wald/u15287/ultralytics/runs/detect/train{i}/weights/best.pt" )
        #model = YOLO(f'/user/christoph.wald/u15287/insect_pest_detection/2_5_self_training/runs/detect/train{i}/weights/best.pt')
    model.train(data=os.path.join(training_folder,"data.yaml"),
                epochs=epochs[i-1],
                imgsz=640,
                close_mosaic=epochs[i-1], #no mosaic
                scale=0.3, #instead of 0.5
                mosaic= 0.0,#25, #instead of 1.0
                mixup=0.0,#5, #instead of 0.0
                erasing=0.4, #default (increase when oberving false positives)
                auto_augment="randaugment", #default, maybe try augmix
                )
    
    end = time.time()
    print(f"Training took {end-start:.2f} seconds.")
    start = end
