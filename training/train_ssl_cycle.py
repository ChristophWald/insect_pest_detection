import sys
sys.path.append("/user/christoph.wald/u15287/insect_pest_detection/modules")
from modules_evaluation import plot_histograms
from modules_prediction import *
from modules import load_yolo_labels
from ultralytics import YOLO
import os
import cv2
import torch
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import random
import shutil

def raw_predictions_on_val(model_number, base_output_path, skip_FRANOC = False, predict_on_tiles = False):

    all_results = []

    print(f"Testing model {model_number}.")
    model_path = f"{path}runs/detect/train{model_number}/weights/best.pt"

    if os.path.exists(model_path):
        #print(f"File exists: {model_path}")
        # You can load the model safely
        model = YOLO(model_path)
    else:
        print(f"File does not exist: {model_path}")
        # Handle the missing file case

    model = YOLO(model_path)
    base_input_path = "/user/christoph.wald/u15287/big-scratch/02_splitted_data/train_labeled/split"
    base_image_path = os.path.join(base_input_path, "images/val")  
    
    #collecting test files
    filenames = os.listdir(base_image_path)
    filenames.sort()

    for filename in filenames:
        #print(f"Predicting on {filename}.")
        if skip_FRANOC and filename.startswith("FRANOC"):
            #print("skipping " + filename)
            continue

        #print(f"Processing {filename}...")
        image = cv2.imread(os.path.join(base_image_path, filename))
        if predict_on_tiles:
            boxes, confs, class_ids = sliding_window_prediction(image, model, conf_threshold = 0.0)
               
        else:
            results = model(image,conf = 0.0, iou = 0.0,  verbose=False, augment=True)
            predictions = results[0].boxes

            if predictions is None or len(predictions) == 0:
                boxes, confs, class_ids = [], [], []
            else:
                boxes = predictions.xyxy
                confs = predictions.conf
                class_ids = predictions.cls
        if len(boxes) > 0:
            boxes, confs, class_ids = nms(boxes, confs, class_ids, iou_threshold=0.4) 
            boxes, confs, class_ids = filter_mostly_contained_boxes(boxes, confs, class_ids, threshold=0.5) 

        pred_output = {
            "filename": filename,
            "boxes": boxes.cpu().tolist(),
            "scores": confs.cpu().tolist(),
            "classes": class_ids.cpu().tolist(),
        }
        all_results.append(pred_output)
    
    with open(os.path.join(base_output_path, "predictions_for_pr.json"), "w") as f:
        json.dump(all_results, f, indent=4)

def load_predictions(pred_json_path):
    with open(pred_json_path, "r") as f:
        return json.load(f)

def compare_labels_single(pred_box, pred_class, pred_score, gt_boxes, gt_classes, iou_threshold=0.5, containment_threshold=0.9, convert_to_xyxy=True):
    """
    Compare a single YOLO prediction box to all GT boxes and return TP/FP flags.
    pred_box: Tensor of shape (1,4)
    pred_class: Tensor of shape (1,)
    gt_boxes: Tensor of shape (N,4)
    gt_classes: Tensor of shape (N,)
    Returns: tp_flag (bool), fp_flag (bool)
    """

    if gt_boxes.numel() == 0:
        # No GT, automatically FP
        return False, True  # TP=False, FP=True

    if convert_to_xyxy:
        pred_xyxy = yolo_to_xyxy_tensor(pred_box)
        gt_xyxy = yolo_to_xyxy_tensor(gt_boxes)
    else:
        pred_xyxy = pred_box
        gt_xyxy = gt_boxes

    # Compute intersection
    xx1 = torch.max(pred_xyxy[:, 0], gt_xyxy[:, 0])
    yy1 = torch.max(pred_xyxy[:, 1], gt_xyxy[:, 1])
    xx2 = torch.min(pred_xyxy[:, 2], gt_xyxy[:, 2])
    yy2 = torch.min(pred_xyxy[:, 3], gt_xyxy[:, 3])
    w = (xx2 - xx1).clamp(min=0)
    h = (yy2 - yy1).clamp(min=0)
    inter = w * h

    pred_area = (pred_xyxy[:, 2] - pred_xyxy[:, 0]) * (pred_xyxy[:, 3] - pred_xyxy[:, 1])
    gt_area = (gt_xyxy[:, 2] - gt_xyxy[:, 0]) * (gt_xyxy[:, 3] - gt_xyxy[:, 1])
    iou = inter / (pred_area + gt_area - inter + 1e-6)
    containment_pred_in_gt = inter / (pred_area + 1e-6)
    containment_gt_in_pred = inter / (gt_area + 1e-6)
    containment_match = (containment_pred_in_gt >= containment_threshold) | (containment_gt_in_pred >= containment_threshold)

    # Match class
    class_match = pred_class[:, None] == gt_classes[None, :]
    match_mask = class_match & ((iou >= iou_threshold) | containment_match)

    if match_mask.any():
        return True, False  # TP
    else:
        return False, True  # FP

def compute_pr_curves_fixed(pred_json_path, class_names, iou_threshold=0.5, containment_threshold=0.9):
    """
    Compute per-class PR curves without double-counting GT boxes.
    Uses compare_labels_single for single-prediction evaluation.
    """

    print("Load predictions.")
    predictions = load_predictions(pred_json_path)

    pr_results = {cls: {"precision": [], "recall": []} for cls in class_names}

    base_input_path = "/user/christoph.wald/u15287/big-scratch/02_splitted_data/train_labeled/split"
    base_image_path = os.path.join(base_input_path, "images/val")  
    base_label_path = os.path.join(base_input_path, "labels/val") 

    # Load ground truth
    print("Load ground truth.")
    gt_dict = {}
    matched_gt_dict = {}  # track matched GT per image
    for filename in os.listdir(base_label_path):
  
        jpg_filename = os.path.splitext(filename)[0] + ".jpg"
        img = cv2.imread(os.path.join(base_image_path, jpg_filename))
        img_height, img_width = img.shape[:2]
        boxes, classes = load_yolo_labels(os.path.join(base_label_path, filename), img_width, img_height)
        gt_dict[jpg_filename] = {"boxes": torch.tensor(boxes).to("cuda"), "classes": torch.tensor(classes).to("cuda")}
        matched_gt_dict[jpg_filename] = torch.zeros(len(classes), dtype=torch.bool).to("cuda")

    # Track GT match counts for debugging
    gt_match_counter = defaultdict(lambda: torch.zeros(1, dtype=torch.int).to("cuda"))

    for cls_id, cls_name in enumerate(class_names):
        print(f"Processing class '{cls_name}'...")

        # Collect all predictions for this class globally
        class_preds = []
        for p in predictions:
            for box, score, c in zip(p["boxes"], p["scores"], p["classes"]):
                if c == cls_id:
                    class_preds.append({
                        "filename": p["filename"],
                        "box": torch.tensor(box).unsqueeze(0).to("cuda"),
                        "score": score,
                        "class": torch.tensor([c]).to("cuda")
                    })

        # Sort descending by confidence
        class_preds.sort(key=lambda x: x["score"], reverse=True)

        tp_cum, fp_cum = [], []
        tp_total, fp_total = 0, 0
        total_gt = sum([gt_dict[f]["classes"].tolist().count(cls_id) for f in gt_dict])

        for d in class_preds:
            filename = d["filename"]
            pred_box = d["box"]
            pred_class = d["class"]
            pred_score = d["score"]

            gt_boxes = gt_dict[filename]["boxes"]
            gt_classes = gt_dict[filename]["classes"]
            matched_gt = matched_gt_dict[filename]

            # Only consider unmatched GT boxes
            unmatched_mask = ~matched_gt
            unmatched_gt_boxes = gt_boxes[unmatched_mask]
            unmatched_gt_classes = gt_classes[unmatched_mask]

            # Compare with single GT set
            if unmatched_gt_boxes.numel() == 0:
                # No unmatched GT to match => prediction is FP
                is_tp, is_fp = False, True
            else:
                is_tp, is_fp = compare_labels_single(
                    pred_box, pred_class, pred_score,
                    unmatched_gt_boxes, unmatched_gt_classes,
                    iou_threshold=iou_threshold,
                    containment_threshold=containment_threshold,
                    convert_to_xyxy=False
                )

                if is_tp:
                    # Mark the best-matched GT as used
                    xx1 = torch.max(pred_box[:, 0], unmatched_gt_boxes[:, 0])
                    yy1 = torch.max(pred_box[:, 1], unmatched_gt_boxes[:, 1])
                    xx2 = torch.min(pred_box[:, 2], unmatched_gt_boxes[:, 2])
                    yy2 = torch.min(pred_box[:, 3], unmatched_gt_boxes[:, 3])
                    w = (xx2 - xx1).clamp(min=0)
                    h = (yy2 - yy1).clamp(min=0)
                    inter = w * h
                    pred_area = (pred_box[:, 2] - pred_box[:, 0]) * (pred_box[:, 3] - pred_box[:, 1])
                    gt_area = (unmatched_gt_boxes[:, 2] - unmatched_gt_boxes[:, 0]) * (unmatched_gt_boxes[:, 3] - unmatched_gt_boxes[:, 1])
                    iou = inter / (pred_area + gt_area - inter + 1e-6)
                    best_idx = torch.argmax(iou)
                    # Update global matched_gt
                    matched_gt[torch.where(unmatched_mask)[0][best_idx]] = True

                    # Increment match counter
                    if filename not in gt_match_counter:
                        gt_match_counter[filename] = torch.zeros(len(gt_boxes), dtype=torch.int).to("cuda")
                    gt_match_counter[filename][torch.where(unmatched_mask)[0][best_idx]] += 1

            tp_total += int(is_tp)
            fp_total += int(is_fp)
            tp_cum.append(tp_total)
            fp_cum.append(fp_total)

        # Compute precision and recall
        precision = [tp/(tp+fp) if (tp+fp) > 0 else 1.0 for tp, fp in zip(tp_cum, fp_cum)]
        recall = [tp/total_gt if total_gt > 0 else 0.0 for tp in tp_cum]

        pr_results[cls_name] = {
            "tp_cum": tp_cum,
            "fp_cum": fp_cum,
            "scores": [d["score"] for d in class_preds],
            "total_gt": total_gt,
            "precision": precision,
            "recall": recall
        }

    # Debug: check for multiple GT matches
    for fname, counts in gt_match_counter.items():
        multiple_matches = (counts > 1).sum().item()
        if multiple_matches > 0:
            print(f"{fname}: {multiple_matches} GT boxes matched more than once (counts={counts.tolist()})")

    total_gt_matches = sum([c.sum().item() for c in gt_match_counter.values()])
    total_gt_boxes = sum([len(gt_dict[f]["classes"]) for f in gt_dict])
    print(f"Total GT boxes: {total_gt_boxes}, total GT matched: {total_gt_matches}")

    return pr_results

def compute_pr_curves_with_all_fixed(pred_json_path, base_output_path, class_names, iou_threshold=0.5, containment_threshold=0.9):
    """
    Compute per-class PR curves and a macro-averaged PR curve including interpolated scores.
    Uses compute_pr_curves_fixed to prevent double-counting of GT boxes.
    """

    # Compute per-class PR curves with GT single-match enforcement
    pr_results = compute_pr_curves_fixed(pred_json_path, class_names, iou_threshold, containment_threshold)

    # Standard recall points for interpolation
    recall_points = np.linspace(0, 1, 101)  # 0.0, 0.01, ..., 1.0
    avg_precision = np.zeros_like(recall_points)
    avg_scores = np.zeros_like(recall_points)
    valid_class_count = 0  # count classes with predictions

    for cname in class_names:
        data = pr_results.get(cname)
        if data is None or len(data["precision"]) == 0:
            print(f"⚠️ Skipping class '{cname}' — no predictions.")
            continue

        recall = np.array(data['recall'])
        precision = np.array(data['precision'])
        scores = np.array(data['scores'])

        # Skip empty classes safely
        if recall.size == 0 or precision.size == 0:
            continue

        # Interpolate precision and scores at fixed recall points
        prec_interp = np.interp(recall_points, recall, precision, left=1.0, right=0.0)
        score_interp = np.interp(recall_points, recall, scores, left=1.0, right=0.0)

        avg_precision += prec_interp
        avg_scores += score_interp
        valid_class_count += 1

    if valid_class_count > 0:
        avg_precision /= valid_class_count
        avg_scores /= valid_class_count

    # Store macro-averaged results
    pr_results["all_classes"] = {
        "recall": recall_points.tolist(),
        "precision": avg_precision.tolist(),
        "scores": avg_scores.tolist()
    }

    with open(os.path.join(base_output_path, "pr_results.json"), "w") as f:
        json.dump(pr_results, f, indent=4)

    return pr_results

def plot_pr_curves(
    pr_results, 
    best_points=None, 
    second_points=None, 
    base_output_path=None, 
    metrics=None
): 
    """
    Plot PR curves for each class and optionally highlight best points 
    and overlay metrics (per-class and summary).
    
    Args:
        pr_results: dict from compute_pr_curves_with_all
        best_points: dict of points to highlight with circles (optional)
        second_points: dict of points to highlight with squares (optional)
        base_output_path: folder to save the plot (optional)
        metrics: dict containing 'per_class' and 'summary' precision/recall (optional)
    """
    plt.figure(figsize=(8, 6)) 
    
    # keep color mapping for consistent plotting
    class_colors = {}
    
    # Plot each class curve
    for cls_name, data in pr_results.items(): 
        line, = plt.plot(data["recall"], data["precision"], label=cls_name)
        class_colors[cls_name] = line.get_color()
        
        # Highlight best_points (circles)
        if best_points and cls_name in best_points:
            bp = best_points[cls_name]
            plt.scatter(
                bp["recall"], bp["precision"], 
                marker='o', s=20, edgecolors='k', facecolors='none', zorder=5
            )
        
        # Highlight second_points (squares)
        if second_points and cls_name in second_points:
            sp = second_points[cls_name]
            plt.scatter(
                sp["recall"], sp["precision"], 
                marker='s', s=20, edgecolors='k', facecolors='none', zorder=5
            )
    
    # Plot metrics points (from old test)
    if metrics:
        per_class = metrics.get("per_class", {})
        for cls_name, vals in per_class.items():
            recall = vals["recall"]
            precision = vals["precision"]
            color = class_colors.get(cls_name, "gray")
            plt.scatter(
                recall, precision,
                marker='s', s=20, edgecolors='k', facecolors='none', color=color, zorder=6
            )

    
    # Highlight target region
    plt.fill_betweenx([0.9, 1.0], 0.8, 1.0, color='gray', alpha=0.3)
    
    # Labels and styling
    plt.xlabel("Recall") 
    plt.ylabel("Precision") 
    plt.title("Precision-Recall Curve") 
    plt.legend(loc="lower left", fontsize=8) 
    plt.grid(True) 

    # Save plot if path provided
    if base_output_path:
        os.makedirs(base_output_path, exist_ok=True)
        plt.savefig(os.path.join(base_output_path, "pr_curve.jpg"), dpi=300, bbox_inches="tight")
    
    plt.show()
    plt.close()

def predict_on_images(model_number="", output_number="x"):
    print("Predicting on full images.")

    # Load YOLO model
    model = YOLO(f"/user/christoph.wald/u15287/insect_pest_detection/training/runs/detect/train{model_number}/weights/best.pt")

    # Full images folder
    image_dir = "/user/christoph.wald/u15287/big-scratch/04_SSL_training_data/training_data/images/train"

    # Ground-truth label folder (for full images)
    label_dir = "/user/christoph.wald/u15287/big-scratch/04_SSL_training_data/training_data/labels/train"

    # Output folder for predictions
    output_dir = f"/user/christoph.wald/u15287/insect_pest_detection/training/predictions{i}"
    os.makedirs(output_dir, exist_ok=True)

    # Structured output: FN / FP / TP -> species -> image
    json_results = {"FN": {}, "FP": {}, "TP": {}}

    # Loop through images
    for filename in sorted(os.listdir(image_dir)):
        if filename.startswith("FRANOC"):
            continue
        #print(f"Predicting on {filename}...")
        image_path = os.path.join(image_dir, filename)
        image = cv2.imread(image_path)

        # Skip if no GT label file
        label_path = os.path.join(label_dir, os.path.splitext(filename)[0] + ".txt")
        label_path = os.path.join(label_dir, os.path.splitext(filename)[0] + ".txt")
        if not os.path.exists(label_path):
            print(f"No label file for {filename}, treating as empty GT.")
            gt_boxes = torch.empty((0, 4), device='cuda')
            gt_classes = torch.empty((0,), dtype=torch.long, device='cuda')
        else:
            # --- Load ground-truth labels ---
            gt_boxes_list, gt_classes_list = load_yolo_labels(label_path, image.shape[1], image.shape[0])
            if len(gt_boxes_list) == 0:
                gt_boxes = torch.empty((0, 4), device='cuda')
                gt_classes = torch.empty((0,), dtype=torch.long, device='cuda')
            else:
                gt_boxes = torch.tensor(gt_boxes_list, device='cuda', dtype=torch.float32)
                gt_classes = torch.tensor(gt_classes_list, device='cuda', dtype=torch.long)

        # --- Prediction on the full image ---
        result = model(image, conf=0.0, iou=0.0, verbose=False, augment=True)
        predictions = result[0].boxes

        if predictions is None or len(predictions) == 0:
            boxes, confs, class_ids = torch.empty((0, 4)), torch.empty((0,)), torch.empty((0,))
        else:
            boxes = predictions.xyxy
            confs = predictions.conf
            class_ids = predictions.cls

        # Apply NMS and containment filtering
        if len(boxes) > 0:
            boxes, confs, class_ids = nms(boxes, confs, class_ids, iou_threshold=0.4)
            boxes, confs, class_ids = filter_mostly_contained_boxes(boxes, confs, class_ids, threshold=0.5)

        # --- Compare predictions vs. ground truth ---
        tp, fp, fn = compare_labels_vectorized(
            boxes, class_ids, confs, gt_boxes, gt_classes,
            tile_size=640, iou_threshold=0.5, containment_threshold=0.5,
            convert_to_xyxy=False
        )
        
        species = filename.split("_")[0]

        # --- Fill the JSON structure ---
        for category, items in zip(["TP", "FP", "FN"], [tp, fp, fn]):
            if category != "FN":
                det_boxes, det_classes, det_scores = items
              
            else:
                det_boxes, det_classes = items
                det_scores = [None] * len(det_boxes)

            if len(det_classes) > 0:
                json_results.setdefault(category, {}).setdefault(species, {}).setdefault(filename, [])
                for cls, box, score in zip(det_classes, det_boxes, det_scores):
                    entry = {"tile_id": None}  # full image, so no tile
                    if category != "FN":
                        entry["prediction"] = [int(cls)] + [float(x) for x in box] + [float(score)]
                    else:
                        entry["prediction"] = [int(cls)] + [float(x) for x in box]
                    json_results[category][species][filename].append(entry)

    # --- Save JSON output ---
    output_path = os.path.join(output_dir, f"predictions_fullimage_{output_number}.json")
    with open(output_path, "w") as f:
        json.dump(json_results, f, indent=4)

    print(f"Results saved to: {output_path}")

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

    print(f"Adding labels from prediction file {pred_file} with thresholds {thresholds}.")

    output_folder = "/user/christoph.wald/u15287/insect_pest_detection/training/metrics"

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

    #add newline character at end of file if missing
    label_path = os.path.join(training_folder, "labels/train")
    for file in os.listdir(label_path):
        if file.endswith(".txt"):
            path = os.path.join(label_path, file)
            with open(path, "rb+") as f:
                f.seek(0, os.SEEK_END)
                if f.tell() > 0:
                    f.seek(-1, os.SEEK_END)
                    if f.read(1) != b"\n":
                        f.write(b"\n")

    # Only FP predictions from the JSON
    data = json_results["FP"]

    total_preds_appended = 0
    corrections = []

    fp_log_path = os.path.join(output_folder, f"fp_labels_added_run{run_number}.txt")
    with open(fp_log_path, "w") as fp_log:
        fp_log.write("species,base_name,tile_id,class_id,conf,xyxy,yolo\n")  # header line

        for species_name, images in data.items():
            for base_name, entries in images.items():
                #print(base_name)
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
                        #print(f"Writing to {label_file}")
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

def find_best_pr_points(pr_results, base_output_path, prec_thresh=0.9, rec_thresh=0.8):
    """
    For each class in pr_results (including 'all_classes'), find the best (precision, recall, score) point.
    """
    best_points = {}

    for cname, data in pr_results.items():
        # Skip malformed or empty entries
        if "scores" not in data:
            continue

        precision = np.array(data["precision"])
        recall = np.array(data["recall"])
        scores = np.array(data["scores"])

        # skip empty arrays safely
        if precision.size == 0 or recall.size == 0 or scores.size == 0:
            print(f"Skipping class '{cname}' — no valid PR data for best-point computation.")
            continue

        # Compute F1 safely
        denom = precision + recall
        f1_scores = np.zeros_like(denom)
        np.divide(
            2 * precision * recall,
            denom,
            out=f1_scores,
            where=denom > 0
        )

        # --- Step 1: inside the "good" zone ---
        inside_mask = (precision >= prec_thresh) & (recall >= rec_thresh)
        if np.any(inside_mask):
            idx_inside = np.argmax(f1_scores[inside_mask])
            best_idx = np.where(inside_mask)[0][idx_inside]
            method = "inside"
        else:
            candidates = []

            mask1 = precision >= prec_thresh
            if np.any(mask1):
                i1 = np.argmax(recall[mask1])
                idx1 = np.where(mask1)[0][i1]
                dist1 = abs(recall[idx1] - rec_thresh)
                candidates.append(("near_high_prec", idx1, dist1))

            mask2 = recall >= rec_thresh
            if np.any(mask2):
                i2 = np.argmax(precision[mask2])
                idx2 = np.where(mask2)[0][i2]
                dist2 = abs(precision[idx2] - prec_thresh)
                candidates.append(("near_high_rec", idx2, dist2))

            mask3 = (precision < prec_thresh) & (recall < rec_thresh)
            if np.any(mask3):
                dists3 = np.sqrt((recall[mask3] - rec_thresh)**2 + (precision[mask3] - prec_thresh)**2)
                i3 = np.argmin(dists3)
                idx3 = np.where(mask3)[0][i3]
                candidates.append(("closest_below", idx3, dists3[i3]))

            if candidates:
                method, best_idx, _ = min(candidates, key=lambda x: x[2])
            else:
                #fallback for empty candidate lists
                best_idx = np.argmax(f1_scores) if f1_scores.size > 0 else None
                method = "fallback"

        if best_idx is not None:
            best_points[cname] = {
                "precision": float(precision[best_idx]),
                "recall": float(recall[best_idx]),
                "score": float(scores[best_idx]),
                "f1": float(f1_scores[best_idx]),
                "method": method
            }
        else:
            print(f"Skipping class '{cname}' — no valid best point found.")

        with open(os.path.join(base_output_path, "operating_points.json"), "w") as f:
            json.dump(best_points, f, indent=4)

    return best_points

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

thresholds = {"BRAIIM": 0.7, "LIRIBO": 0.7, "FRANOC": 0, "TRIAVA": 0.7}
class_names = ["fungus gnats", "leaf miner flies", "thrips", "whiteflies"]  # replace with your classes
path = "/user/christoph.wald/u15287/insect_pest_detection/training/"
             
for i in range(2,5):
   
    #######
    #eval model on val
    #######

    model_number = i
    base_output_path = f"{path}metrics/train{model_number}"
    os.makedirs(base_output_path, exist_ok=True)
   
    
    #makes prediction with conf=0 and saves them
    raw_predictions_on_val(model_number = model_number,
                        base_output_path=base_output_path,
                        skip_FRANOC = True, 
                        predict_on_tiles = False)

    #reload the raw predictions and creates precision recall curves
    pred_json_path = os.path.join(base_output_path, "predictions_for_pr.json")
    pr_results = compute_pr_curves_with_all_fixed(pred_json_path, base_output_path, class_names, iou_threshold=0.5, containment_threshold=0.5)

    #plots the curves
    best_points = find_best_pr_points(pr_results, base_output_path)
    plot_pr_curves(pr_results, best_points=best_points, base_output_path=base_output_path)

    #####
    #collects prediction on train
    #####

    files = os.listdir("/user/christoph.wald/u15287/big-scratch/04_SSL_training_data/training_data/images/unprocessed")
    files_to_copy = random.sample(files, min(3, len(files)))
    for f in files_to_copy:
        src = os.path.join("/user/christoph.wald/u15287/big-scratch/04_SSL_training_data/training_data/images/unprocessed", f)
        dst = os.path.join("/user/christoph.wald/u15287/big-scratch/04_SSL_training_data/training_data/images/train", f)
        shutil.copy2(src, dst)
        print(f"Copied: {f}")

    
    predictions_path  = f"/user/christoph.wald/u15287/insect_pest_detection/training/predictions{i}" 
    predict_on_images(model_number=model_number, output_number = model_number)

    #plot histograms
    intersections = plot_histograms(predictions_path, predictions_path)
    '''
    # Update thresholds using the intersection_conf values
    for (json_name, species), (conf, prec) in intersections.items():
        # Only update species that exist in your dictionary
        if species in thresholds:
            thresholds[species] = conf  # or round(conf, 2) if you want cleaner numbers

    print(thresholds)
    '''
    ####
    # add labels
    ####

    add_labels(          
        pred_file = os.path.join(f"/user/christoph.wald/u15287/insect_pest_detection/training/predictions{i}/predictions_fullimage_{i}.json"),
        thresholds = thresholds,
        run_number = str(i),
        correct_labels = True,
        threshold_steps = False,
        write = True,
        write_into_tiles = False,
        add_weights=False
    )
    #thresholds = {k: max(0, v - 0.05) for k, v in thresholds.items()}

    #####
    # train
    #####

    delete_cache_files("/user/christoph.wald/u15287/big-scratch/04_SSL_training_data/training_data/labels")

    model = YOLO(f"/user/christoph.wald/u15287/insect_pest_detection/training/runs/detect/train{i}/weights/best.pt")
    model.train(data = "/user/christoph.wald/u15287/big-scratch/04_SSL_training_data/training_data/data.yaml", 
                epochs=3, 
                #patience = 10, 
                imgsz=1280,
                #save_period=1,
                scale=0.3, #instead of 0.5
                mosaic= 0.25, #instead of 1.0
                mixup=0.05, #instead of 0.0
                erasing=0.4, #default (increase when oberving false positives)
                auto_augment="randaugment", #default, maybe try augmix
                flipud = 0.5,

            
                )

model_number = str(i+1)
base_output_path = f"{path}metrics/train{model_number}"
os.makedirs(base_output_path, exist_ok=True)

#makes prediction with conf=0 and saves them
raw_predictions_on_val(model_number = model_number,
                    base_output_path=base_output_path,
                    skip_FRANOC = True, 
                    predict_on_tiles = False)

#reload the raw predictions and creates precision recall curves
pred_json_path = os.path.join(base_output_path, "predictions_for_pr.json")
pr_results = compute_pr_curves_with_all_fixed(pred_json_path, base_output_path, class_names, iou_threshold=0.5, containment_threshold=0.5)

#plots the curves
best_points = find_best_pr_points(pr_results, base_output_path)
plot_pr_curves(pr_results, best_points=best_points, base_output_path=base_output_path)
