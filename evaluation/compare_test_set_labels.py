import os
import torch
from collections import defaultdict
import sys
import cv2
sys.path.append("/user/christoph.wald/u15287/insect_pest_detection/modules")
from modules import load_yolo_labels
from modules_prediction import compare_labels_vectorized

'''
compares two set of labels with the same logic as evaluating predictions
'''


def compare_label_folders_vectorized(folder_a, folder_b,
                                     iou_threshold=0.5, containment_threshold=0.9):

    per_class_stats = defaultdict(lambda: {"TP": 0, "FP": 0, "FN": 0})
    missing_files = []

    
    for root, _, files in os.walk(folder_a):
        for file in files:
            if not file.endswith('.txt'):
                continue

            rel_path = os.path.relpath(os.path.join(root, file), folder_a)
            path_a = os.path.join(folder_a, rel_path)
            path_b = os.path.join(folder_b, rel_path)

            #get image width and height, needed for loading yolo labels
            base, _ = os.path.splitext(file)
            labels_parent = os.path.dirname(root) 
            img_path = os.path.join(labels_parent, "images", base + ".jpg")
            img = cv2.imread(img_path)
            img_height, img_width = img.shape[:2]

            if not os.path.exists(path_b):
                missing_files.append(rel_path)
                continue

            # Load YOLO labels
            gt_boxes, gt_classes = load_yolo_labels(path_a, img_width, img_height)
            pred_boxes, pred_classes = load_yolo_labels(path_b, img_width, img_height)

            # Convert to tensors
            gt_boxes = torch.tensor(gt_boxes, dtype=torch.float32)
            gt_classes = torch.tensor(gt_classes, dtype=torch.int64)
            pred_boxes = torch.tensor(pred_boxes, dtype=torch.float32)
            pred_classes = torch.tensor(pred_classes, dtype=torch.int64)
            pred_scores = torch.ones(len(pred_classes), dtype=torch.float32)

            (tp, fp, fn) = compare_labels_vectorized(
                pred_boxes, pred_classes, pred_scores,
                gt_boxes, gt_classes,
                iou_threshold=iou_threshold,
                containment_threshold=containment_threshold,
                convert_to_xyxy=False
            )

            # Aggregate TP/FP/FN counts
            for c in tp[1]: per_class_stats[c]["TP"] += 1
            for c in fp[1]: per_class_stats[c]["FP"] += 1
            for c in fn[1]: per_class_stats[c]["FN"] += 1

    return per_class_stats, missing_files



folder_gt = "/user/christoph.wald/u15287/big-scratch/02_splitted_data/test_set/test_set_w_old_labels/labels"
folder_new = "/user/christoph.wald/u15287/big-scratch/02_splitted_data/test_set/test_set_w_new_labels/labels"

stats, missing = compare_label_folders_vectorized(folder_gt, folder_new)

print("\n=== Per-Class Results ===")
print("Class | TP | FP | FN")
for cls, vals in sorted(stats.items()):
    print(f"{cls:5d} | {vals['TP']:3d} | {vals['FP']:3d} | {vals['FN']:3d}")

if missing:
    print("Missing files in new folder:")
    for m in missing:
        print(" -", m)
