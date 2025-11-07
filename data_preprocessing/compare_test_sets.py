import os
import torch
from collections import defaultdict


'''
compares two set of labels with the same logic as evaluating predictions

'''


def read_labels(file_path):
    """Read YOLO labels and return list of parsed (class_id, cx, cy, w, h) floats."""
    labels = []
    if not os.path.exists(file_path):
        return labels
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                try:
                    class_id = int(parts[0])
                    values = tuple(float(x) for x in parts[1:])
                    labels.append((class_id, *values))
                except ValueError:
                    continue
    return labels


def yolo_to_tensor(labels):
    """Convert list of YOLO labels into PyTorch tensors (boxes, classes)."""
    if len(labels) == 0:
        return (
            torch.empty((0, 4), dtype=torch.float32),
            torch.empty((0,), dtype=torch.int64),
        )
    class_ids = torch.tensor([l[0] for l in labels], dtype=torch.int64)
    boxes = torch.tensor([l[1:] for l in labels], dtype=torch.float32)
    return boxes, class_ids


def yolo_to_xyxy_tensor(boxes):
    """Convert YOLO-format boxes [xc, yc, w, h] → [x1, y1, x2, y2]."""
    if boxes.numel() == 0:
        return boxes
    x1 = boxes[:, 0] - boxes[:, 2] / 2
    y1 = boxes[:, 1] - boxes[:, 3] / 2
    x2 = boxes[:, 0] + boxes[:, 2] / 2
    y2 = boxes[:, 1] + boxes[:, 3] / 2
    return torch.stack([x1, y1, x2, y2], dim=1)

def compare_labels_vectorized(
    pred_boxes,
    pred_classes,
    pred_scores,
    gt_boxes,
    gt_classes,
    tile_size=640,
    iou_threshold=0.5,
    containment_threshold=0.5,
    convert_to_xyxy=True
):
    device = pred_boxes.device

    # Sort by confidence
    sort_idx = torch.argsort(pred_scores, descending=True)
    pred_boxes = pred_boxes[sort_idx]
    pred_classes = pred_classes[sort_idx]
    pred_scores = pred_scores[sort_idx]

    # Convert to xyxy if needed
    if convert_to_xyxy:
        pred_boxes_xyxy = yolo_to_xyxy_tensor(pred_boxes)
        gt_boxes_xyxy   = yolo_to_xyxy_tensor(gt_boxes)
    else:
        pred_boxes_xyxy = pred_boxes
        gt_boxes_xyxy   = gt_boxes

    if pred_boxes_xyxy.numel() == 0 and gt_boxes_xyxy.numel() == 0:
        return ([], [], []), ([], [], []), ([], [])

    # Compute areas
    pred_areas = (pred_boxes_xyxy[:, 2] - pred_boxes_xyxy[:, 0]).clamp(min=0) * \
                 (pred_boxes_xyxy[:, 3] - pred_boxes_xyxy[:, 1]).clamp(min=0)
    gt_areas = (gt_boxes_xyxy[:, 2] - gt_boxes_xyxy[:, 0]).clamp(min=0) * \
               (gt_boxes_xyxy[:, 3] - gt_boxes_xyxy[:, 1]).clamp(min=0)

    # Compute intersections
    xx1 = torch.max(pred_boxes_xyxy[:, None, 0], gt_boxes_xyxy[None, :, 0])
    yy1 = torch.max(pred_boxes_xyxy[:, None, 1], gt_boxes_xyxy[None, :, 1])
    xx2 = torch.min(pred_boxes_xyxy[:, None, 2], gt_boxes_xyxy[None, :, 2])
    yy2 = torch.min(pred_boxes_xyxy[:, None, 3], gt_boxes_xyxy[None, :, 3])

    w = (xx2 - xx1).clamp(min=0)
    h = (yy2 - yy1).clamp(min=0)
    inter = w * h
    union = pred_areas[:, None] + gt_areas[None, :] - inter
    iou = inter / (union + 1e-6)

    # Containment check
    containment_pred_in_gt = inter / (pred_areas[:, None] + 1e-6)
    containment_gt_in_pred = inter / (gt_areas[None, :] + 1e-6)
    containment_match = (containment_pred_in_gt >= containment_threshold) | \
                        (containment_gt_in_pred >= containment_threshold)

    # Class matching
    class_match = pred_classes[:, None] == gt_classes[None, :]
    match_matrix = class_match & ((iou >= iou_threshold) | containment_match)

    matched_pred = torch.zeros(pred_boxes.size(0), dtype=torch.bool, device=device)
    matched_gt   = torch.zeros(gt_boxes.size(0), dtype=torch.bool, device=device)

    tp_boxes, tp_classes, tp_scores = [], [], []
    fp_boxes, fp_classes, fp_scores = [], [], []

    # Greedy matching
    for i in range(pred_boxes.size(0)):
        possible = torch.where(match_matrix[i] & ~matched_gt)[0]
        if possible.numel() > 0:
            j = possible[torch.argmax(iou[i, possible])].item()
            tp_boxes.append(pred_boxes_xyxy[i].cpu().tolist())
            tp_classes.append(int(pred_classes[i].cpu()))
            tp_scores.append(float(pred_scores[i].cpu()))
            matched_gt[j] = True
            matched_pred[i] = True
        else:
            fp_boxes.append(pred_boxes_xyxy[i].cpu().tolist())
            fp_classes.append(int(pred_classes[i].cpu()))
            fp_scores.append(float(pred_scores[i].cpu()))

    fn_boxes = [gt_boxes_xyxy[i].cpu().tolist() for i in range(gt_boxes.size(0)) if not matched_gt[i]]
    fn_classes = [int(gt_classes[i].cpu()) for i in range(gt_classes.size(0)) if not matched_gt[i]]

    return (tp_boxes, tp_classes, tp_scores), (fp_boxes, fp_classes, fp_scores), (fn_boxes, fn_classes)



def compare_label_folders_vectorized(folder_a, folder_b, iou_threshold=0.5, containment_threshold=0.9):
    """
    Compare two YOLO label folders and compute TP/FP/FN per class.
    folder_a = ground truth, folder_b = new/predicted labels
    """
    per_class_stats = defaultdict(lambda: {"TP": 0, "FP": 0, "FN": 0})
    missing_files = []

    for root, _, files in os.walk(folder_a):
        for file in files:
            if not file.endswith('.txt'):
                continue

            rel_path = os.path.relpath(os.path.join(root, file), folder_a)
            path_a = os.path.join(folder_a, rel_path)
            path_b = os.path.join(folder_b, rel_path)

            if not os.path.exists(path_b):
                missing_files.append(rel_path)
                continue

            gt_labels = read_labels(path_a)
            pred_labels = read_labels(path_b)

            gt_boxes, gt_classes = yolo_to_tensor(gt_labels)
            pred_boxes, pred_classes = yolo_to_tensor(pred_labels)
            pred_scores = torch.ones(len(pred_classes), dtype=torch.float32)  # assume all predictions have score=1

            (tp, fp, fn) = compare_labels_vectorized(
                pred_boxes, pred_classes, pred_scores,
                gt_boxes, gt_classes,
                iou_threshold=iou_threshold,
                containment_threshold=containment_threshold
            )

            # Aggregate counts per class
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
