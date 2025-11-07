import sys
sys.path.append("/user/christoph.wald/u15287/insect_pest_detection/modules")
#from modules_testing import *
import time
import os
import json
import shutil
import cv2

def xyxy_to_yolo(x1, y1, x2, y2, width, height):
    """
    Convert bounding box from [x1, y1, x2, y2] to YOLO [x_center, y_center, w, h]
    normalized by image width and height.
    """
    x_center = (x1 + x2) / 2 / width
    y_center = (y1 + y2) / 2 / height
    w = (x2 - x1) / width
    h = (y2 - y1) / height
    return [x_center, y_center, w, h]


def add_labels(pred_file, 
               thresholds, 
               run_number = "not_needed", 
               correct_labels = True, 
               threshold_steps = False, 
               write = True, 
               write_into_tiles = False,
               add_weights = False):
    '''
    pred_file = json file with predicitions on the tiles, should be in the "predictions folder"
    thresholds = dict, example: {"BRAIIM": 0.8, "LIRIBO": 0.8, "FRANOC": 0, "TRIAVA": 0.8}
    run_number = to identify the files, e.g. "08"
    correct_labels = if True, class_ids are adapted to the expected species, identified by the filename
    threshold_steps = if True, only predictions from between threshold and threshold + 0.1 are logged (useful for testing)
    write = if True, new labels are written into the YOLO-training data
    write_into_tile = if True, new labels are written into the tiles data (for the next iteration)
    
    
    '''
    start = time.time()

    print(f"Adding labels from prediction file {pred_file} with thresholds {thresholds}.")

    output_folder = "/user/christoph.wald/u15287/insect_pest_detection/training/metrics"
    
    pred_file = os.path.join("/user/christoph.wald/u15287/insect_pest_detection/training/predictions", pred_file)

    # Load predictions
    with open(pred_file, "r") as f:
        json_results = json.load(f)
    data = json_results["FP"]

    print("Adding new labels training data.")
    print("Using thresholds:")
    print(thresholds)

    #folder with images and labels in tiles to write onto (optional) and to copy from
    tiles_folder = "dummy"
    # folder with images and labels in yolo-format
    training_folder = "/user/christoph.wald/u15287/big-scratch/04_SSL_training_data/training_data"


    # Only FP predictions from the JSON
    data = json_results["FP"]

    total_preds_appended = 0
    corrections = []

    fp_log_path = os.path.join(output_folder, f"fp_labels_added_run{run_number}.txt")
    with open(fp_log_path, "w") as fp_log:
        fp_log.write("species,base_name,tile_id,class_id,conf,xyxy,yolo\n")  # header line

        for species_name, images in data.items():
            for base_name, entries in images.items():

                #added for unsegemented images
                file_usage = "train"

                for entry in entries:
                    tile_id = entry['tile_id']
                    pred = entry['prediction']  # [class_id, x1, y1, x2, y2, conf]

                    # Unpack prediction
                    class_id, x1, y1, x2, y2, conf = pred

                    # Apply threshold
                    threshold = thresholds[species_name]
                    if threshold_steps:
                        if not (threshold <= conf < threshold + 0.1):
                            continue
                    else: 
                        if conf < threshold:
                            continue


                    # Correct class according to filename if needed
                    gt_class_id = list(thresholds.keys()).index(species_name)
                    if correct_labels and class_id != gt_class_id:
                        corrections.append([base_name, tile_id, pred, class_id, gt_class_id])
                        class_id = gt_class_id  # enforce correct class
                        

                    # Path to the existing tile label file
                    base_filename = os.path.splitext(base_name)[0]  # e.g., "LIRIBO_0629"
                    full_filename = f"{base_filename}.txt"
                    # full_filename = f"{base_filename}_tile_{tile_id}.txt"
                    tile_file = os.path.join(tiles_folder, "images",full_filename) 
                    label_file = os.path.join(training_folder, "labels", file_usage,full_filename)

                    # Append the FP prediction directly

                    src_image = os.path.join("/user/christoph.wald/u15287/big-scratch/02_splitted_data/train_labeled/split/images/train", os.path.splitext(full_filename)[0]+".jpg")

                    # Load image with cv2
                    img = cv2.imread(src_image)
                    if img is None:
                        raise FileNotFoundError(f"Image {src_image} not found or cannot be opened.")

                    height, width = img.shape[:2]  # note: OpenCV returns (height, width, channels)

                    # Convert prediction to YOLO format
                    yolo_box = xyxy_to_yolo(x1, y1, x2, y2, width=width, height=height)

                    #yolo_box = xyxy_to_yolo(x1, y1, x2, y2, tile_size=640) #shape is hardcoded

                    # ---- LOG HERE ----
                    fp_log.write(
                        f"{species_name},{base_name},{tile_id},{class_id},{conf:.3f},"
                        f"[{x1},{y1},{x2},{y2}],[{yolo_box[0]:.4f},{yolo_box[1]:.4f},{yolo_box[2]:.4f},{yolo_box[3]:.4f}]\n"
                    )

                    if write:
                        print(f"Writing to {label_file}")
                        is_empty = os.path.exists(label_file) == False
                        
                        if not is_empty and add_weights:
                            with open(label_file, "r") as f:
                                existing_lines = [l.strip().split() for l in f.readlines() if l.strip()]
                            normalized_lines = []
                            for line in existing_lines:
                                if len(line) == 5:
                                    line.append("1.0")  # add default weight
                                normalized_lines.append(" ".join(line))
                            with open(label_file, "w") as f:
                                f.write("\n".join(normalized_lines) + "\n")

                        with open(label_file, "a") as f:
                            if add_weights:
                                f.write(f"{class_id} {yolo_box[0]} {yolo_box[1]} {yolo_box[2]} {yolo_box[3]} {conf}\n")
                            else:
                                f.write(f"{class_id} {yolo_box[0]} {yolo_box[1]} {yolo_box[2]} {yolo_box[3]}\n") 

                                            

                        if is_empty:
                            src_image = os.path.join(tiles_folder, "images", os.path.splitext(full_filename)[0]+".jpg")
                            dest_image = os.path.join(training_folder, "images", file_usage, os.path.splitext(full_filename)[0]+".jpg")
                            shutil.copy(src_image, dest_image)

                    if write_into_tiles:
                        with open(tile_file, "a") as f:
                            f.write(f"{class_id} {yolo_box[0]} {yolo_box[1]} {yolo_box[2]} {yolo_box[3]}\n")

                    total_preds_appended += 1
                    #print(base_name, tile_id)

    with open(os.path.join(output_folder, f"class_corrections{run_number}.txt"), "w") as f:
    # First line: variable names
        f.write("base_name,tile_id,pred,class_id,gt_class_id\n")
        for entry in corrections:
            # Convert pred list to string to fit in one column
            pred_str = "[" + ",".join(map(str, entry[2])) + "]"
            f.write(f"{entry[0]},{entry[1]},{pred_str},{entry[3]},{entry[4]}\n")

    print(f"Total FP predictions appended: {total_preds_appended}")

    end = time.time()
    print(f"Adding the new labels took {end-start:.2f} seconds.")
    start = end

add_labels(
    pred_file = "predictions_fullimage_2.json", #change if needed!
    thresholds = {"BRAIIM": 0.2, "LIRIBO": 0.2, "FRANOC": 0, "TRIAVA": 0.2},
    run_number = "1",
    correct_labels = True,
    threshold_steps = False,
    write = True,
    write_into_tiles = False,
    add_weights=False
)

