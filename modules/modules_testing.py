import sys
sys.path.append("/user/christoph.wald/u15287/insect_pest_detection/modules")
#sys.path.append("/user/christoph.wald/u15287/ultralytics")

from ultralytics import YOLO
import time
import os
import json
import cv2
import shutil
from modules_prediction import *
from modules import load_yolo_labels
from modules_evaluation import *
import random

import os
#import sys
import yaml

def delete_cache_files(folder: str) -> None:
    """
    Deletes all files ending with '.cache' in the specified folder (non-recursively).

    Args:
        folder (str): The folder path to search for .cache files
    """
    if not os.path.exists(folder):
        print(f"Folder does not exist: {folder}")
        return

    deleted_count = 0
    for file in os.listdir(folder):
        if file.endswith(".cache"):
            file_path = os.path.join(folder, file)
            if os.path.isfile(file_path):
                try:
                    os.remove(file_path)
                    deleted_count += 1
                except Exception as e:
                    print(f"Failed to delete {file_path}: {e}")

    print(f"Deleted {deleted_count} .cache files from {folder}")


def check_data_yaml(yaml_path: str) -> None:
    """
    Checks whether the 'train' and 'val' paths in a YOLO-style data.yaml file:
      1. Exist on disk
      2. Are located within the parent directory of the YAML file

    Exits with an error message if a check fails.
    """
    if not os.path.exists(yaml_path):
        sys.exit(f"YAML file not found: {yaml_path}")

    # Load YAML
    with open(yaml_path, "r") as f:
        try:
            data = yaml.safe_load(f)
        except Exception as e:
            sys.exit(f"Failed to parse YAML: {e}")

    train_path = data.get("train")
    val_path = data.get("val")
    yaml_parent = os.path.dirname(os.path.abspath(yaml_path))

    if not train_path or not val_path:
        sys.exit("Missing 'train' or 'val' entries in YAML.")

    def check_path(label, path):
        abs_path = os.path.abspath(path)
        if not os.path.exists(abs_path):
            sys.exit(f"{label} path does not exist: {abs_path}")
        if not abs_path.startswith(yaml_parent):
            sys.exit(
                f"{label} path is not inside YAML parent directory:\n"
                f"  path:   {abs_path}\n"
                f"  parent: {yaml_parent}"
            )

    check_path("train", train_path)
    check_path("val", val_path)

    print("All paths are valid and within the YAML parent directory.")


def train(train_data_dir, model_number, epochs):
        
    print(f"Starting the training with {epochs} epochs.")

    
    if model_number == None:
        print("Using pretrained YOLO.")
        model = YOLO('yolov8s.pt')
    else:
        model = YOLO(f'/user/christoph.wald/u15287/insect_pest_detection/2_5_self_training/runs/detect/train{model_number}/weights/best.pt')


    model.train(data=os.path.join(train_data_dir,"data.yaml"),
                epochs=epochs,
                imgsz=640,
                close_mosaic=epochs, #no mosaic
                scale=0.3, #instead of 0.5
                mosaic= 0.25, #0.25, #instead of 1.0
                mixup=0.05, #0.05 #instead of 0.0
                erasing=0.4, #default (increase when oberving false positives)
                auto_augment="randaugment", #default, maybe try augmix
                degrees=90.0
                )


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

    output_folder = "/user/christoph.wald/u15287/insect_pest_detection/2_5_self_training/metrics"
    
    pred_file = os.path.join("/user/christoph.wald/u15287/insect_pest_detection/2_5_self_training/predictions", pred_file)

    # Load predictions
    with open(pred_file, "r") as f:
        json_results = json.load(f)
    data = json_results["FP"]

    print("Adding new labels training data.")
    print("Using thresholds:")
    print(thresholds)

    #folder with images and labels in tiles to write onto (optional) and to copy from
    tiles_folder = "/user/christoph.wald/u15287/big-scratch/02_splitted_data/train_labeled/SSL/tiles"
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
                    full_filename = f"{base_filename}_tile_{tile_id}.txt"
                    tile_file = os.path.join(tiles_folder,file_usage,"labels_training_run",full_filename) 
                    label_file = os.path.join(training_folder, "labels", file_usage,full_filename)

                    # Append the FP prediction directly
                    yolo_box = xyxy_to_yolo(x1, y1, x2, y2, tile_size=640)

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
                            src_image = os.path.join(tiles_folder,file_usage, "images", os.path.splitext(full_filename)[0]+".jpg")
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


def evaluate_on_test_set(conf_threshold, model_number = "", save_images = False, save_results = True,  skip_FRANOC = True):
    start = time.time()

    results = []

    print(f"Testing model {model_number}.")
    model = YOLO(f"/user/christoph.wald/u15287/insect_pest_detection/2_5_self_training/runs/detect/train{model_number}/weights/best.pt")
    base_output_path = f"/user/christoph.wald/u15287/insect_pest_detection/2_5_self_training/metrics/train{model_number}"
    os.makedirs(base_output_path, exist_ok=True)

    base_input_path = "/user/christoph.wald/u15287/big-scratch/02_splitted_data/test_set"
    base_image_path = os.path.join(base_input_path, "images")
    base_label_path = os.path.join(base_input_path, "labels")   

    #added for testing on masked test set
    #base_image_path = "/user/christoph.wald/u15287/big-scratch/02_splitted_data/test_set/SSL/03_images_masked"
    #base_label_path = "/user/christoph.wald/u15287/big-scratch/02_splitted_data/test_set/SSL/04_labels_cropped_and_filtered_yolo"

    #collecting test files
    filenames = os.listdir(base_image_path)
    filenames.sort()

    if save_images:
        image_output_path = os.path.join(base_output_path, "images_w_bboxes")
        os.makedirs(image_output_path, exist_ok=True)

    for filename in filenames:
        if skip_FRANOC and filename.startswith("FRANOC"):
            #print("skipping " + filename)
            continue

        #added for testing on masked test set
        label_path = os.path.join(base_label_path, os.path.splitext(filename)[0] + ".txt")
        if not os.path.exists(label_path):
            continue

        #print(f"Processing {filename}...")
        image = cv2.imread(os.path.join(base_image_path, filename))
        boxes, confs, class_ids = sliding_window_prediction(image, model, conf_threshold)
        
        if len(boxes) > 0:
            boxes, confs, class_ids = nms(boxes, confs, class_ids, iou_threshold=0.4) 
            boxes, confs, class_ids = filter_mostly_contained_boxes(boxes, confs, class_ids, threshold=0.5)    
        #print(f"Predicted: {boxes.size(0)}")

        label_path = os.path.join(base_label_path, os.path.splitext(filename)[0] + ".txt")
        label_boxes, label_classes_ids = load_yolo_labels(label_path, image.shape[1], image.shape[0])
 
        label_boxes = torch.tensor(label_boxes).to("cuda")
        label_classes_ids = torch.tensor(label_classes_ids).to("cuda")
 
        tp, fp, fn = compare_labels_vectorized(boxes, class_ids, confs, label_boxes, label_classes_ids,
                                               tile_size = 640, iou_threshold=0.5, containment_threshold=0.8, 
                                               convert_to_xyxy=False)
        
        results.append([filename, tp, fp, fn])
        if save_images: make_image_with_boxes(image, tp, fp, fn, image_output_path, filename)    
        metrics = compute_metrics(results)
        if save_results: 
            with open(os.path.join(base_output_path, "metrics.json"), "w") as f:
                json.dump(metrics, f, indent=4)
            save_results_to_json(base_output_path, results)
    
    end = time.time()
    print(f"Predicting took {end-start:.2f} seconds.")
    start = end


def evaluate_on_test_set_image_proc(conf_threshold, model_number = "", save_images = True, save_results = True,  skip_FRANOC = True):

    def parse_coords(line):
        line = line.strip()
        if not line:
            return None
        # Remove surrounding brackets or parentheses
        line = line.strip("[]()")
        # Split by comma and convert each element to int after stripping whitespace
        coords = [int(x.strip()) for x in line.split(",")]
        return coords
    
    start = time.time()

    results = []

    print(f"Testing model {model_number}.")
    model = YOLO(f"/user/christoph.wald/u15287/insect_pest_detection/2_5_self_training/runs/detect/train{model_number}/weights/best.pt")
    base_output_path = f"/user/christoph.wald/u15287/insect_pest_detection/2_5_self_training/metrics/train{model_number}_pseudolabeltest"
    os.makedirs(base_output_path, exist_ok=True)

    base_input_path = "/user/christoph.wald/u15287/big-scratch/02_splitted_data/test_set"
    base_image_path = os.path.join(base_input_path, "SSL/04_images_cropped")
    base_label_path = os.path.join(base_input_path, "SSL/05_created_labels")   

    #collecting test files
    filenames = os.listdir("/user/christoph.wald/u15287/big-scratch/02_splitted_data/test_set/SSL/03_images_masked")
    filenames.sort()

    if save_images:
        image_output_path = os.path.join(base_output_path, "images_w_bboxes")
        os.makedirs(image_output_path, exist_ok=True)

    for filename in filenames:
        if skip_FRANOC and filename.startswith("FRANOC"):
            #print("skipping " + filename)
            continue
        print(f"Processing {filename}...")
        image = cv2.imread(os.path.join(base_image_path, filename))
        boxes, confs, class_ids = sliding_window_prediction(image, model, conf_threshold)
        
        if len(boxes) > 0:
            boxes, confs, class_ids = nms(boxes, confs, class_ids, iou_threshold=0.4) 
            boxes, confs, class_ids = filter_mostly_contained_boxes(boxes, confs, class_ids, threshold=0.5)    
        

        label_path = os.path.join(base_label_path, os.path.splitext(filename)[0] + ".txt")
        label_boxes = []
        with open(label_path, "r") as f:
              for line in f:
                coords = parse_coords(line)
                if coords is not None:
                    label_boxes.append(coords)

        if len(label_boxes) == 0:
            label_boxes = torch.empty((0,4), dtype=torch.float32).to("cuda")
            label_classes_ids = torch.empty((0,), dtype=torch.long).to("cuda")
        else:
            label_boxes = torch.tensor(label_boxes, dtype=torch.float32).to("cuda")


            # Derive class ID from filename or from content if available
            species = ["BRAIIM", "LIRIBO","FRANOC", "TRIAVA"]
            row_index = next((i for i, sp in enumerate(species) if filename.startswith(sp)), len(species))
            label_classes_ids = torch.tensor([row_index]*len(label_boxes), dtype=torch.long).to("cuda")

        tp, fp, fn = compare_labels_vectorized(boxes, class_ids, confs, label_boxes, label_classes_ids,
                                               tile_size = 640, iou_threshold=0.5, containment_threshold=0.8, 
                                               convert_to_xyxy=False)
        print(tp)
        results.append([filename, tp, fp, fn])
        if save_images: make_image_with_boxes(image, tp, fp, fn, image_output_path, filename)    
        metrics = compute_metrics(results)
        if save_results: 
            with open(os.path.join(base_output_path, "metrics.json"), "w") as f:
                json.dump(metrics, f, indent=4)
            save_results_to_json(base_output_path, results)
    
    end = time.time()
    print(f"Predicting took {end-start:.2f} seconds.")
    start = end

def predict_on_tiles(model_number = "", output_number = "x"):
    start = time.time()

    print("Predicting on the tiles.")
    model = YOLO(f"/user/christoph.wald/u15287/insect_pest_detection/2_5_self_training/runs/detect/train{model_number}/weights/best.pt")
    
    # full images folder (predicting on cropped image contained in train/val)
    image_dirs = [
        "/user/christoph.wald/u15287/big-scratch/02_splitted_data/train_labeled/SSL/split/images/train",
        #"/user/christoph.wald/u15287/big-scratch/02_splitted_data/train_labeled/SSL/split/images/val"
    ]

    # tile labels folder
    label_dirs = [
        "/user/christoph.wald/u15287/big-scratch/02_splitted_data/train_labeled/SSL/tiles/train/labels",
        #"/user/christoph.wald/u15287/big-scratch/02_splitted_data/train_labeled/SSL/tiles/val/labels"
    ]
    
    #output folder for predictions
    output_dir = f"/user/christoph.wald/u15287/insect_pest_detection/2_5_self_training/predictions"
    os.makedirs(output_dir, exist_ok=True)

    # output dict structured by FN/FP/TP -> species -> image
    json_results = {"FN": {}, "FP": {}, "TP": {}}

    # cycles through train and val directories
    for image_dir, label_dir in zip(image_dirs, label_dirs):

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

    with open(os.path.join(output_dir, f'predictions_{output_number}.json'), 'w') as f:
        json.dump(json_results, f, indent=4)

    end = time.time()
    print(f"Predicting took {end-start:.2f} seconds.")
    start = end


def create_labels(image_folder, model_number = ""):
    #does not work if folders are in the image folder
    start = time.time()

    print("Predicting on the tiles.")
    model = YOLO(f"/user/christoph.wald/u15287/insect_pest_detection/2_5_self_training/runs/detect/train{model_number}/weights/best.pt")
    

       
    #output folder for predictions
    output_dir = f"/user/christoph.wald/u15287/insect_pest_detection/2_5_self_training/predictions"
    os.makedirs(output_dir, exist_ok=True)

    # output dict structured by FN/FP/TP -> species -> image
    json_results = {"FN": {}, "FP": {}, "TP": {}}


    # cycles through all images in a directory
    for filename in os.listdir(image_folder):
        #print(f"Predicting on {filename}.")
        image_path = os.path.join(image_folder, filename)
        image = cv2.imread(image_path)

        # predicts on the full images (with stride 420)
        boxes, confs, class_ids = sliding_window_prediction(image, model)
        
        
        
        if boxes.numel() > 0:
            boxes, confs, class_ids = nms(boxes, confs, class_ids, iou_threshold=0.4, device=model.device)
            
            boxes, confs, class_ids = filter_mostly_contained_boxes(boxes, confs, class_ids, threshold=0.5)
            #print(f"Predicted: {boxes.size(0)}")
        
        pred_tiles_data = get_labels_per_tile_tensor(image, boxes, class_ids, confs)

        fps_per_tile = []

        for pred in pred_tiles_data:
            if pred.numel() == 0:
                boxes = torch.empty((0, 4))
                classes = torch.empty((0,), dtype=torch.long)
                scores = torch.empty((0,))
            else:
                boxes = yolo_to_xyxy_tensor(pred[:, 1:5].cpu(), tile_size=640)     # move to CPU immediately
                classes = pred[:, 0].long().cpu()
                scores = pred[:, 5].cpu()

            fps_per_tile.append((boxes, classes, scores))

        species = filename.split("_")[0]
        
        print(filename)
        for tile_id, (boxes_tile, classes_tile, scores_tile) in enumerate(fps_per_tile):
            if len(classes_tile) == 0:
                continue
            json_results.setdefault("FP", {}).setdefault(species, {}).setdefault(filename, [])
            for cls, box, score in zip(classes_tile.tolist(),
                                    boxes_tile.tolist(),
                                    scores_tile.tolist()):
                entry = {"tile_id": tile_id, "prediction": [cls, *box, score]}
                json_results["FP"][species][filename].append(entry)

    # Save JSON
    output_json_path = os.path.join(output_dir,"predictions_for_unsegmented_images.json")
    

    with open(output_json_path, "w") as f:
        json.dump(json_results, f, indent=4)

    end = time.time()
    print(f"Predicting took {end-start:.2f} seconds.")

def add_empty_tiles():
    print(f"Adding 5% empty tiles.")

    tiles_folder = "/user/christoph.wald/u15287/big-scratch/02_splitted_data/train_labeled/SSL/tiles"
    training_folder = "/user/christoph.wald/u15287/big-scratch/04_SSL_training_data/training_data"
        
    # Setup file structure for destination
    file_types = ["images", "labels"]
    u = "train"

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
    max_empty_allowed = int(len(label_files) *5 / 100)
    empty_to_copy = min(max_empty_allowed, len(empty_files))
    print(f"Copying {empty_to_copy} files.")

    if empty_to_copy > 0:
        random_empty_files = random.sample(empty_files, empty_to_copy)  # random selection
        for file in random_empty_files:
            img_file = os.path.splitext(file)[0] + ".jpg"
            #print(img_file)
            shutil.copy2(os.path.join(img_path, img_file), os.path.join(img_dest_path, img_file))
            shutil.copy2(os.path.join(label_path, file), os.path.join(label_dest_path, file))
