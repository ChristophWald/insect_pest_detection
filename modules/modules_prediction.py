from ultralytics import YOLO
import cv2
import torch
import math
import os
import torchvision
from modules import pad_to_multiple

'''
sliding_window_prediction: collects predictions for an image by sliding over it
filter_by_prediction: filters prediction by class-specific confidence threholds
nms: filters predictions by NMS
filter_mostly_contained_boxes: filters precited boxes contained inside larger boxes
compute_intersection_area_tensor: vectorized calculation of intersection
compare_labels_vectorized: fast comparision to ground truth
yolo_to_xyxy_tensor: loads yolo labels and returns tensor with absolute xyxy-coordinates

'''


def sliding_window_prediction(image, model, conf_threshold=0):
    height, width, _ = image.shape
    tile_size = 1280 #640
    stride = 1080 #420

    num_windows_y = (height - tile_size) // stride + 1
    num_windows_x = (width - tile_size) // stride + 1

    all_boxes, all_scores, all_classes = [], [], []

    for i in range(num_windows_y):
        for j in range(num_windows_x):
            y_start, x_start = i * stride, j * stride
            y_end, x_end = y_start + tile_size, x_start + tile_size

            window = image[y_start:y_end, x_start:x_end]
            pad_bottom, pad_right = max(0, y_end - height), max(0, x_end - width)
            window_padded = cv2.copyMakeBorder(
                window, 0, pad_bottom, 0, pad_right, cv2.BORDER_CONSTANT, value=(0, 0, 0)
            )

            window_rgb = cv2.cvtColor(window_padded, cv2.COLOR_BGR2RGB)
            window_tensor = (
                torch.tensor(window_rgb / 255.0)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .float()
                .to(model.device)
            )

            results = model(window_tensor, verbose=False, augment=True)
            predictions = results[0].boxes

            valid = predictions.conf > conf_threshold
            boxes = predictions.xyxy[valid]
            scores = predictions.conf[valid]
            classes = predictions.cls[valid]
            
            #shifting the coordinates back to full image
            boxes[:, [0, 2]] += x_start
            boxes[:, [1, 3]] += y_start

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_classes.append(classes)

    if len(all_boxes) == 0:
        return torch.zeros((0, 4), device=model.device), torch.zeros((0,), device=model.device), torch.zeros((0,), device=model.device)

    return (
        torch.cat(all_boxes, dim=0),
        torch.cat(all_scores, dim=0),
        torch.cat(all_classes, dim=0),
    )

def filter_by_class_confidence(boxes, scores, classes, class_conf_thresholds):
    """
    Filter predictions per class based on a minimum confidence threshold for each class.

    Args:
        boxes (torch.Tensor): Tensor of shape (N, 4)
        scores (torch.Tensor): Tensor of shape (N,)
        classes (torch.Tensor): Tensor of shape (N,)
        class_conf_thresholds (dict): e.g., {0: 0.5, 1: 0.6, 2: 0.4} mapping class_id to min confidence

    Returns:
        boxes, scores, classes filtered tensors
    """
    if boxes.numel() == 0:
        return boxes, scores, classes

    keep_mask = torch.zeros_like(scores, dtype=torch.bool)

    for cls_id, min_conf in class_conf_thresholds.items():
        class_mask = classes == cls_id
        keep_mask |= class_mask & (scores >= min_conf)

    return boxes[keep_mask], scores[keep_mask], classes[keep_mask]

def nms(boxes, scores, classes, iou_threshold=0.5, device="cuda"):
    if boxes.numel() == 0:
        return boxes, scores, classes
    keep = torch.ops.torchvision.nms(
        boxes.to(device), scores.to(device), iou_threshold
    )
    return boxes[keep], scores[keep], classes[keep]

def filter_mostly_contained_boxes(boxes, scores, classes, threshold=0.5):
    if boxes.numel() == 0:
        return boxes, scores, classes

    boxes = boxes.float()
    scores = scores.float()

    areas = (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)

    xx1 = torch.max(boxes[:, None, 0], boxes[None, :, 0])
    yy1 = torch.max(boxes[:, None, 1], boxes[None, :, 1])
    xx2 = torch.min(boxes[:, None, 2], boxes[None, :, 2])
    yy2 = torch.min(boxes[:, None, 3], boxes[None, :, 3])

    w = (xx2 - xx1).clamp(min=0)
    h = (yy2 - yy1).clamp(min=0)
    inter = w * h

    containment_ratio = inter / (areas[:, None] + 1e-6)

    # j can suppress i if score[j] >= score[i]
    higher_or_equal_conf_mask = scores[None, :] >= scores[:, None]

    containment_mask = (containment_ratio >= threshold) & higher_or_equal_conf_mask

    # remove self-comparisons
    containment_mask.fill_diagonal_(False)

    mostly_contained = containment_mask.any(dim=1)
    keep = ~mostly_contained

    return boxes[keep], scores[keep], classes[keep]

def compute_intersection_area_tensor(box_a, box_b):
    """
    box_a: Tensor[N, 4] (x1, y1, x2, y2)
    box_b: Tensor[M, 4] (x1, y1, x2, y2)
    Returns: Tensor[N, M] of intersection areas
    """
    N = box_a.shape[0]
    M = box_b.shape[0]

    # Expand for broadcasting
    a = box_a[:, None, :]  # [N, 1, 4]
    b = box_b[None, :, :]  # [1, M, 4]

    xi1 = torch.maximum(a[..., 0], b[..., 0])
    yi1 = torch.maximum(a[..., 1], b[..., 1])
    xi2 = torch.minimum(a[..., 2], b[..., 2])
    yi2 = torch.minimum(a[..., 3], b[..., 3])

    inter_w = torch.clamp(xi2 - xi1, min=0)
    inter_h = torch.clamp(yi2 - yi1, min=0)
    return inter_w * inter_h  # [N, M]

def compare_labels_vectorized(
    pred_boxes,
    pred_classes,
    pred_scores,
    gt_boxes,
    gt_classes,
    tile_size=640,
    iou_threshold=0.5,
    containment_threshold=0.9, convert_to_xyxy = True
):
    """
    Compare YOLO predictions to ground-truth boxes and return TP, FP, FN.

    pred_boxes, gt_boxes: Tensor of shape (N,4) in YOLO format [xc, yc, w, h] normalized (0-1)
    pred_classes, gt_classes: Tensor of shape (N,) integer class IDs
    pred_scores: Tensor of shape (N,) confidence scores
    tile_size: tile size to scale normalized boxes to pixels
    Returns:
        (tp_boxes, tp_classes, tp_scores), (fp_boxes, fp_classes, fp_scores), (fn_boxes, fn_classes)
        All lists with XYXY coordinates in pixels
    """
    device = pred_boxes.device

    #sorting by confidence
    sort_idx = torch.argsort(pred_scores, descending=True)
    pred_boxes = pred_boxes[sort_idx]
    pred_classes = pred_classes[sort_idx]
    pred_scores = pred_scores[sort_idx]

    if convert_to_xyxy:
        pred_boxes_xyxy = yolo_to_xyxy_tensor(pred_boxes)
        gt_boxes_xyxy   = yolo_to_xyxy_tensor(gt_boxes)
    else:
        pred_boxes_xyxy = pred_boxes
        gt_boxes_xyxy = gt_boxes

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

    '''old
    min_area = torch.min(pred_areas[:, None], gt_areas[None, :])
    containment = inter / (min_area + 1e-6)
    '''

    
    containment_pred_in_gt = inter / (pred_areas[:, None] + 1e-6)
    containment_gt_in_pred = inter / (gt_areas[None, :] + 1e-6)
    containment_match = (containment_pred_in_gt >= containment_threshold) | \
                        (containment_gt_in_pred >= containment_threshold)
    

    # Class matching
    class_match = pred_classes[:, None] == gt_classes[None, :]
    match_matrix = class_match & ((iou >= iou_threshold) | containment_match)
    #old
    #match_matrix = class_match & ((iou >= iou_threshold) | (containment >= containment_threshold))

    matched_pred = torch.zeros(pred_boxes.size(0), dtype=torch.bool, device=device)
    matched_gt   = torch.zeros(gt_boxes.size(0), dtype=torch.bool, device=device)

    tp_boxes, tp_classes, tp_scores = [], [], []
    fp_boxes, fp_classes, fp_scores = [], [], []

    # Greedy matching
    for i in range(pred_boxes.size(0)):
        possible = torch.where(match_matrix[i] & ~matched_gt)[0]
        if possible.numel() > 0:
            j = possible[torch.argmax(iou[i, possible])].item() #picks box with highest iou
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

def yolo_to_xyxy_tensor(boxes, tile_size=640):
    """
    Convert YOLO boxes normalized to a tile into absolute pixel coordinates in the tile.
    Vectorized for torch tensors.

    boxes: Tensor of shape (N, 4) with [xc, yc, w, h] in normalized format (0-1).
    tile_size: size of the tile (default: 640)

    Returns: Tensor of shape (N, 4) with [xmin, ymin, xmax, ymax] in pixel coords.
    """
    boxes = boxes.clone()  # avoid modifying input

    # Scale normalized values by tile size
    boxes[:, 0] = boxes[:, 0] * tile_size  # xc
    boxes[:, 1] = boxes[:, 1] * tile_size  # yc
    boxes[:, 2] = boxes[:, 2] * tile_size  # w
    boxes[:, 3] = boxes[:, 3] * tile_size  # h

    # Compute corners
    x_min = boxes[:, 0] - boxes[:, 2] / 2
    y_min = boxes[:, 1] - boxes[:, 3] / 2
    x_max = boxes[:, 0] + boxes[:, 2] / 2
    y_max = boxes[:, 1] + boxes[:, 3] / 2

    return torch.stack([x_min, y_min, x_max, y_max], dim=1)
