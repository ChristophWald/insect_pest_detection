import sys
sys.path.append("/user/christoph.wald/u15287/insect_pest_detection/modules")
#sys.path.append("/user/christoph.wald/u15287/ultralytics")

from ultralytics import YOLO
import time
import os
import json
import cv2
import torch
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from modules_prediction import *
from modules import load_yolo_labels


def get_predictions(training_path, #path the parent folder of runs/detect
                    model_number, #number of the run
                    base_output_path, 
                    skip_FRANOC = False, #skips thrips if Ture
                    predict_on_tiles = False, #predicts on full images if False
                    test_set = False #predicts on validation set if False
                    ):
    start = time.time()

    '''
    gets all predictions for a model on the test/val set and saves them as a json    
    '''

    all_results = []

    #loading the model
    print(f"Testing model {model_number}.")
    model_path = f"{training_path}runs/detect/train{model_number}/weights/best.pt"


    if os.path.exists(model_path):
        print(f"File exists: {model_path}")
        model = YOLO(model_path)
    else:
        print(f"File does not exist: {model_path}")

    model = YOLO(model_path)

    #setting the path to images/labels
    if not test_set:
        base_input_path = "/user/christoph.wald/u15287/big-scratch/02_splitted_data/train_labeled/split"
        base_image_path = os.path.join(base_input_path, "images/val")  
    else: 
        base_input_path = "/user/christoph.wald/u15287/big-scratch/02_splitted_data/test_set/test_set_w_new_labels"
        base_image_path = os.path.join(base_input_path, "images") 

    #collecting test files
    filenames = os.listdir(base_image_path)
    filenames.sort()

    for filename in filenames:
        print(f"Predicting on {filename}.")
        if skip_FRANOC and filename.startswith("FRANOC"):
            print("skipping " + filename)
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
        
    end = time.time()
    print(f"Predicting took {end-start:.2f} seconds.")
    start = end

def load_predictions(pred_json_path):
    with open(pred_json_path, "r") as f:
        return json.load(f)

def compare_labels_single(pred_box, 
                          pred_class, 
                          pred_score, 
                          gt_boxes, 
                          gt_classes, 
                          iou_threshold=0.5, 
                          containment_threshold=0.9, 
                          convert_to_xyxy=True):
    """
    Compare a single YOLO prediction box to all GT boxes and return TP/FP flags.
    pred_box: Tensor of shape (1,4)
    pred_class: Tensor of shape (1,)
    gt_boxes: Tensor of shape (N,4)
    gt_classes: Tensor of shape (N,)
    Returns: tp_flag (bool), fp_flag (bool)
    """
    device = pred_box.device

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


def compute_pr_curves(pred_json_path, 
                    class_names, 
                    iou_threshold=0.5, 
                    containment_threshold=0.9, 
                    test_set = False):
    """
    Compute per-class PR curves
    Uses compare_labels_single for single-prediction evaluation.
    """

    print("Load predictions.")
    predictions = load_predictions(pred_json_path)

    pr_results = {cls: {"precision": [], "recall": []} for cls in class_names}

    if not test_set:
        base_input_path = "/user/christoph.wald/u15287/big-scratch/02_splitted_data/train_labeled/split"
        base_image_path = os.path.join(base_input_path, "images/val")  
        base_label_path = os.path.join(base_input_path, "labels/val") 
    else:
        base_input_path = "/user/christoph.wald/u15287/big-scratch/02_splitted_data/test_set/test_set_w_new_labels"
        base_image_path = os.path.join(base_input_path, "images") 
        base_label_path = os.path.join(base_input_path, "labels")

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

def compute_pr_curves_with_all(pred_json_path, class_names, iou_threshold=0.5, containment_threshold=0.9, test_set = False):
    """
    Compute per-class PR curves and a macro-averaged PR curve including interpolated scores.
    """

    # Compute per-class PR curves with GT single-match enforcement
    pr_results = compute_pr_curves(pred_json_path, class_names, iou_threshold, containment_threshold, test_set = test_set)

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

    return pr_results

def find_best_pr_points(pr_results, prec_thresh=0.9, rec_thresh=0.8):
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

    return best_points

def plot_pr_curves(
    pr_results, 
    best_points=None,  
    base_output_path=None, 
    metrics=None,
    title=None,
): 
    """
    Plot PR curves for each class
    if best points are given, plots them as circles
    if metrics are given (saved by predict_on_fixed_thresholds), plots pr points as squares
    """

    label_map = {
        "fungus gnats": "Fungus gnats",
        "leaf miner flies": "Leaf miner flies",
        "thrips": "Thrips",
        "whiteflies": "Whiteflies",
        "all_classes": "All classes"
    }

    plt.figure(figsize=(8, 6))

    # Assign a fixed color to each class based on the label map order
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    class_colors = {}
    for i, cls_name in enumerate(label_map.keys()):
        class_colors[cls_name] = color_cycle[i % len(color_cycle)]

    #plot the pr curves
    for cls_name, data in pr_results.items():
        # Skip classes with no data
        if len(data.get("recall", [])) == 0 or len(data.get("precision", [])) == 0:
            continue

        display_name = label_map.get(cls_name, cls_name)
        color = class_colors[cls_name]

        plt.plot(data["recall"], data["precision"], label=display_name, color=color)

        # Best points (circles)
        if best_points and cls_name in best_points:
            bp = best_points[cls_name]
            plt.scatter(
                bp["recall"], bp["precision"],
                marker='o', s=40, edgecolors='k', facecolors='none', zorder=5, color=color
            )

    #if metrics are given, plots pr points as squares
    if metrics:
        per_class = metrics.get("per_class", {})
        for cls_name, vals in per_class.items():
            recall = vals.get("recall")
            precision = vals.get("precision")

            # Skip metric points if values are None or class has no valid data
            if recall is None or precision is None:
                continue
            if recall == 0 and precision == 0:
                # Optionally skip zero points for classes with no curve
                if cls_name not in pr_results or len(pr_results[cls_name].get("recall", [])) == 0:
                    continue

            color = class_colors.get(cls_name, "gray")
            plt.scatter(
                recall, precision,
                marker='s', s=40, edgecolors='k', facecolors='none',
                color=color, zorder=6
            )

    #results image processing
    '''
    coords = [(0.5045945945945945, 0.784453781512605), (0.6048242237618681, 0.9210629152012505), (0.20974015870546134, 0.8685567010309279)]  # replace with your actual coordinates
    species_to_show = ["fungus gnats", "leaf miner flies", "whiteflies"]

    for (x, y), sp in zip(coords, species_to_show):
        plt.scatter(x, y, color=class_colors[sp], s=50, label=sp, zorder=10)
    '''
        
    #plt.legend(loc='upper right', fontsize=12, frameon=False)
    #target_zone
    plt.fill_betweenx([0.9, 1.0], 0.8, 1.0, color='gray', alpha=0.3)

    plotted_classes = [cls_name for cls_name in pr_results if len(pr_results[cls_name].get("recall", [])) > 0]
    if plotted_classes:
        handles = [plt.Line2D([], [], color=class_colors[cls], label=label_map.get(cls, cls)) for cls in plotted_classes]
        class_legend = plt.legend(handles=handles, loc="lower left", fontsize=12,title_fontsize=12, title="Classes", bbox_to_anchor=(0.1, 0.23) )
        plt.gca().add_artist(class_legend)

    # Custom legend
    custom_handles = []
    custom_labels = []

    # Grey patch
    target_patch = plt.Rectangle((0, 0), 1, 1, color='gray', alpha=0.3)
    custom_handles.append(target_patch)
    custom_labels.append("Target zone")

    #prec_rec_handle = plt.Line2D([], [], color='k', marker='o', linestyle='None', markersize=6)
    #custom_handles.append(prec_rec_handle)
    #custom_labels.append("Precision/recall image processing")

    # Threshold markers
    if best_points:
        handle = plt.Line2D([], [], marker='o', linestyle='None', markersize=6, markeredgecolor='k', markerfacecolor='none')
        custom_handles.append(handle)
        custom_labels.append("Optimal confidence threshold")
    elif metrics:
        handle = plt.Line2D([], [], marker='s', linestyle='None', markersize=6, markeredgecolor='k', markerfacecolor='none')
        custom_handles.append(handle)
        custom_labels.append("Optimal confidence threshold from validation set")

    plt.legend(custom_handles, custom_labels, loc="lower left", fontsize=12, bbox_to_anchor=(0.1, 0.05) )
    
    plt.xlabel("Recall", fontsize = 12)
    plt.ylabel("Precision", fontsize = 12)
    plt.title(title, fontsize = 12)
    plt.xticks(fontsize = 12)
    plt.yticks(fontsize = 12)
    plt.grid(True)

    if base_output_path:
        import os
        os.makedirs(base_output_path, exist_ok=True)
        out_path = os.path.join(base_output_path, "pr_curve.jpg")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")

    plt.show()
    plt.close()


#setup
test_set= True #if False, tests on validation set
path = "/user/christoph.wald/u15287/insect_pest_detection/results/self_training_evaluation/"


model_number = "4"
if test_set:
    base_output_path = f"{path}metrics/train{model_number}_test_set"
else:
    base_output_path = f"{path}metrics/train{model_number}"
os.makedirs(base_output_path, exist_ok=True)

class_names = ["fungus gnats", "leaf miner flies", "thrips", "whiteflies"]

#makes prediction with conf=0 and saves them
get_predictions(training_path = path,
                model_number = model_number,
                    base_output_path=base_output_path,
                    skip_FRANOC = True,
                    predict_on_tiles = False,
                    test_set = test_set)

#reload the raw predictions and creates precision recall curves
pred_json_path = os.path.join(base_output_path, "predictions_for_pr.json")
pr_results = compute_pr_curves_with_all(pred_json_path, class_names, iou_threshold=0.5, containment_threshold=0.5, test_set = test_set)

# saves pr results
with open(os.path.join(base_output_path, "pr_results.json"), "w") as f:
    json.dump(pr_results, f, indent=4)
#reloads the pr results
with open(os.path.join(base_output_path, "pr_results.json"), 'r') as f:
        pr_results = json.load(f)

#calculates and saves operating points
best_points = find_best_pr_points(pr_results)

#loads saved operating points
with open(os.path.join(base_output_path, "operating_points.json"), "w") as f:
        json.dump(best_points, f, indent=4)

#plots curves
#for validation set
if not test_set:
    plot_pr_curves(pr_results, 
                   best_points=best_points, 
                   base_output_path=base_output_path, 
                   title = "Model trained supervised on tiles and evaluated on validation set")
#for test set
else:
    #loads metrics given by predict_on_fixed_thresholds
    with open(os.path.join(base_output_path, "metrics.json"), "r") as f:
        metrics_file = json.load(f)

    plot_pr_curves(
        pr_results,
        best_points=None,
        base_output_path=base_output_path,
        metrics=metrics_file,
        title = "Model trained supervised on tiles and evaluated on test set"
    )
