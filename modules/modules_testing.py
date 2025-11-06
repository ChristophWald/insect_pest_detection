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
        

        label_path = os.path.sjoin(base_label_path, os.path.splitext(filename)[0] + ".txt")
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
                                               tile_size = 640, iou_threshold=0.5, containment_threshold=1.1, 
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
    model = YOLO(f"/user/christoph.wald/u15287/insect_pest_detection/2_3_supervised_training/runs/detect/train{model_number}/weights/best.pt")
    
    # full images folder (predicting on cropped image, because these are all rotated
    image_dirs = [
        "/user/christoph.wald/u15287/big-scratch/02_splitted_data/train_labeled/SSL/04_images_cropped",
        #"/user/christoph.wald/u15287/big-scratch/02_splitted_data/train_labeled/SSL/split/images/val"
    ]

    # tile labels folder
    label_dirs = [
        "/user/christoph.wald/u15287/big-scratch/02_splitted_data/train_labeled/SSL/tiles_mininside08/labels",
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
            if filename not in os.listdir("/user/christoph.wald/u15287/big-scratch/02_splitted_data/train_labeled/split/images/train"):
                continue
            if filename.startswith("FRANOC"):
                continue
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
