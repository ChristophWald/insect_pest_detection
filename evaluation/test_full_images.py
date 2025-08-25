from ultralytics import YOLO
import os
import cv2
import torch
import numpy as np
from collections import defaultdict
import json
from modules.modules import draw_box, load_yolo_labels, save_cropped_boxes, compute_intersection_area

'''
testing prediction on full images
returns general metrics, full prediction info 
can print color coded boxes into images (green: true positive, blue: false positive, red: false negative)
'''

def sliding_window_prediction(image, model, conf_threshold=0.365):
    
    height, width, _ = image.shape
    tile_size = 640
    stride = 420

    num_windows_y = (height - tile_size) // stride + 1
    num_windows_x = (width - tile_size) // stride + 1

    all_boxes = []
    all_scores = []
    all_classes = []

    for i in range(num_windows_y):
        for j in range(num_windows_x):
            y_start = i * stride
            x_start = j * stride
            y_end = y_start + tile_size
            x_end = x_start + tile_size

            window = image[y_start:y_end, x_start:x_end]
            pad_bottom = max(0, y_end - height)
            pad_right = max(0, x_end - width)
            window_padded = cv2.copyMakeBorder(window, 0, pad_bottom, 0, pad_right, cv2.BORDER_CONSTANT, value=(0, 0, 0))

            window_rgb = cv2.cvtColor(window_padded, cv2.COLOR_BGR2RGB)
            window_tensor = torch.tensor(window_rgb / 255.0).permute(2, 0, 1).unsqueeze(0).float().to(model.device)

            results = model(window_tensor, verbose=False, augment=True)
            predictions = results[0].boxes
            valid = predictions.conf > conf_threshold
            predictions = predictions[valid]

            for box, cls, conf in zip(predictions.xyxy.cpu().numpy(), predictions.cls.cpu().numpy(), predictions.conf.cpu().numpy()):
                xmin, ymin, xmax, ymax = box
                # Convert to global coordinates
                xmin += x_start
                xmax += x_start
                ymin += y_start
                ymax += y_start

                all_boxes.append([xmin, ymin, xmax, ymax])
                all_scores.append(conf)
                all_classes.append(int(cls))

    return [all_boxes, all_scores, all_classes]

def nms(boxes, scores, classes, iou_threshold = 0.5):
    
    boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
    scores_tensor = torch.tensor(scores, dtype=torch.float32)
    classes_tensor = torch.tensor(classes, dtype=torch.int)

    keep_nms = torch.ops.torchvision.nms(boxes_tensor, scores_tensor, iou_threshold)
    boxes_nms = boxes_tensor[keep_nms]
    scores_nms = scores_tensor[keep_nms]
    classes_nms = classes_tensor[keep_nms]

    return [boxes_nms, scores_nms, classes_nms]

def filter_mostly_contained_boxes(boxes, scores, classes, threshold=0.9):

    keep = []
    for i, box_a in enumerate(boxes):
        xa1, ya1, xa2, ya2 = box_a
        area_a = (xa2 - xa1) * (ya2 - ya1)
        if area_a == 0:
            continue

        mostly_contained = False
        for j, box_b in enumerate(boxes):
            if i == j or scores[j] < scores[i]:
                continue
            inter_area = compute_intersection_area(box_a, box_b)
            if area_a > 0 and (inter_area / area_a) >= threshold:
                mostly_contained = True
                break
        if not mostly_contained:
            keep.append(i)

    return [boxes[keep].tolist(), scores[keep].tolist(), classes[keep].tolist()]

def compute_iou(box1, box2):
    inter_area = compute_intersection_area(box1, box2)
    if inter_area == 0:
        return 0.0
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    iou = inter_area / (box1_area + box2_area - inter_area)
    return iou

def compute_containment_ratio(inner_box, outer_box):
    inter_area = compute_intersection_area(inner_box, outer_box)
    inner_area = (inner_box[2] - inner_box[0]) * (inner_box[3] - inner_box[1])
    return inter_area / inner_area if inner_area > 0 else 0.0


def compare_labels(
    pred_boxes,   # list of predicted boxes: [[xmin, ymin, xmax, ymax], ...]
    pred_classes, # list of predicted classes: [cls_id, ...]
    gt_boxes,     # list of ground truth boxes: [[xmin, ymin, xmax, ymax], ...]
    gt_classes,   # list of ground truth classes: [cls_id, ...]
    iou_threshold=0.5,
    containment_threshold=0.9
):
    matched_gt = set()

    tp_boxes = []
    tp_classes = []

    fp_boxes = []
    fp_classes = []

    fn_boxes = []
    fn_classes = []

    # Match predictions to GT
    for pred_box, pred_cls in zip(pred_boxes, pred_classes):
        matched = False
        for i, (gt_box, gt_cls) in enumerate(zip(gt_boxes, gt_classes)):
            if i in matched_gt:
                continue
            if pred_cls == gt_cls:
                iou = compute_iou(pred_box, gt_box)
                containment_pred_in_gt = compute_containment_ratio(pred_box, gt_box)
                containment_gt_in_pred = compute_containment_ratio(gt_box, pred_box)
                if (iou >= iou_threshold or 
                    containment_pred_in_gt >= containment_threshold or 
                    containment_gt_in_pred >= containment_threshold):
                    tp_boxes.append(pred_box)
                    tp_classes.append(pred_cls)
                    matched_gt.add(i)
                    matched = True
                    break
        if not matched:
            fp_boxes.append(pred_box)
            fp_classes.append(pred_cls)

    # Find GT boxes not matched → false negatives
    for i, (gt_box, gt_cls) in enumerate(zip(gt_boxes, gt_classes)):
        if i not in matched_gt:
            fn_boxes.append(gt_box)
            fn_classes.append(gt_cls)

    return (tp_boxes, tp_classes), (fp_boxes, fp_classes), (fn_boxes, fn_classes)

def make_image_with_boxes(
    image,
    tp_data,  # (boxes, classes)
    fp_data,  # (boxes, classes)
    fn_data,  # (boxes, classes)
    output_image_path=None,
    filename=None
):
    
    drawn = image.copy()

    tp_boxes, tp_classes = tp_data
    fp_boxes, fp_classes = fp_data
    fn_boxes, fn_classes = fn_data

    # Draw true positives in green
    for box, cls in zip(tp_boxes, tp_classes):
        draw_box(drawn, box, (0, 255, 0), f"TP: {cls}")

    # Draw false positives in blue
    for box, cls in zip(fp_boxes, fp_classes):
        draw_box(drawn, box, (255, 0, 0), f"FP: {cls}")

    # Draw false negatives in red
    for box, cls in zip(fn_boxes, fn_classes):
        draw_box(drawn, box, (0, 0, 255), f"FN: {cls}")

    if output_image_path:
        cv2.imwrite(os.path.join(output_image_path,os.path.splitext(filename)[0] + "_w_boxes.jpg"), drawn)

from collections import defaultdict

def compute_metrics(results):
    """
    Compute total and per-class metrics from a list of detection results.

    Args:
        results: list of [filename, tp, fp, fn] where each of tp, fp, fn is (boxes, class_ids)

    Returns:
        Dictionary with overall 'summary' and per-class 'per_class' metrics.
    """
    class_stats = defaultdict(lambda: {"TP": 0, "FP": 0, "FN": 0})

    # Aggregate counts
    for _, tp, fp, fn in results:
        for cls in tp[1]:
            class_stats[cls]["TP"] += 1
        for cls in fp[1]:
            class_stats[cls]["FP"] += 1
        for cls in fn[1]:
            class_stats[cls]["FN"] += 1

    # Compute per-class precision and recall
    for cls, stats in class_stats.items():
        tp = stats["TP"]
        fp = stats["FP"]
        fn = stats["FN"]
        stats["precision"] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        stats["recall"] = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # Compute totals
    total_tp = sum(stats["TP"] for stats in class_stats.values())
    total_fp = sum(stats["FP"] for stats in class_stats.values())
    total_fn = sum(stats["FN"] for stats in class_stats.values())

    total_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    total_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0

    return {
        "summary": {
            "TP": total_tp,
            "FP": total_fp,
            "FN": total_fn,
            "precision": total_precision,
            "recall": total_recall
        },
        "per_class": dict(class_stats)
    }

import json

def save_results_to_json(output_path,results):
    """
    Save the list of detection results to a JSON file.

    Args:
        results: List of [filename, tp, fp, fn] entries.
        output_path: Path to the output JSON file.
    """
    formatted = []
    for filename, tp, fp, fn in results:
        entry = {
            "filename": filename,
            "true_positives": {
                "boxes": tp[0],
                "classes": tp[1],
            },
            "false_positives": {
                "boxes": fp[0],
                "classes": fp[1],
            },
            "false_negatives": {
                "boxes": fn[0],
                "classes": fn[1],
            }
        }
        formatted.append(entry)

    with open(os.path.join(output_path, "results.json"), "w") as f:
        json.dump(formatted, f, indent=4)

#load model
model = YOLO('/user/christoph.wald/u15287/insect_pest_detection/runs/detect/train7/weights/best.pt')

#set in- & output path
base_input_path = "/user/christoph.wald/u15287/big-scratch/02_splitted_data/test_set"
base_output_path = "/user/christoph.wald/u15287/insect_pest_detection/train7_test_prediction_568"
base_image_path = os.path.join(base_input_path, "images")
base_label_path = os.path.join(base_input_path, "labels")
image_output_path = os.path.join(base_output_path, "images_w_bboxes")
boxes_output_path = os.path.join(base_output_path, "boxes")
os.makedirs(base_output_path, exist_ok=True)
os.makedirs(image_output_path, exist_ok=True)
os.makedirs(boxes_output_path, exist_ok=True)

filenames = os.listdir(base_image_path)
filenames.sort()

save_images = False #explain
save_boxes = False #
save_results = True #

conf_threshold=0.568

results = []

#make a block with the thresholds here

for filename in filenames:
    print(f"Processing {filename}...")
    image = cv2.imread(os.path.join(base_image_path, filename))
    boxes, confs, class_ids = sliding_window_prediction(image, model, conf_threshold) #returns lists
    print(f"Number of predicted boxes after thresholding: {len(boxes)}")
    boxes, confs, class_ids = nms(boxes, confs, class_ids, iou_threshold=0.4) #transform the lists to tensors
    print(f"Number of predicted boxes after NMS: {len(boxes)}")
    boxes, confs, class_ids = filter_mostly_contained_boxes(boxes, confs, class_ids, threshold=0.5) #operates on tensors and returns lists again
    print(f"Number of predicted boxes after removing contained boxes: {len(boxes)}")
    
    label_path = os.path.join(base_label_path, os.path.splitext(filename)[0] + ".txt")
    label_boxes, label_classes_ids = load_yolo_labels(label_path, image.shape[1], image.shape[0])
    tp, fp, fn = compare_labels(pred_boxes=boxes, pred_classes=class_ids, gt_boxes=label_boxes, gt_classes=label_classes_ids, 
                                iou_threshold = 0.5,
                                containment_threshold = 0.9)
    results.append([filename, tp, fp, fn])
    if save_images: make_image_with_boxes(image, tp, fp, fn, image_output_path, filename)
    if save_boxes: save_cropped_boxes(image, fp[0],filename, boxes_output_path)

metrics = compute_metrics(results)
if save_results: 
    with open(os.path.join(base_output_path, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=4)
    save_results_to_json(base_output_path, results)