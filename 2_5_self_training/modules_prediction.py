from ultralytics import YOLO
import cv2
import torch
import math
import os


def sliding_window_prediction(image, model, conf_threshold=0):
    height, width, _ = image.shape
    tile_size = 640
    stride = 420

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

def pad_image_to_multiple(image, tile_size=640, pad_value=(114,114,114)):
    h, w = image.shape[:2]
    pad_w = math.ceil(w / tile_size) * tile_size - w
    pad_h = math.ceil(h / tile_size) * tile_size - h
    padded = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=pad_value)
    return padded, w, h

def save_tile_labels_to_tensor(boxes, classes, scores, tile_box, tile_size, min_inside_ratio=0.8):
    """
    boxes: [N, 4], classes: [N], scores: [N]
    tile_box: (x1, y1, x2, y2)
    Returns: Tensor[K, 6] = [class, nx, ny, nw, nh, score]
    """
    x_tile, y_tile, x_tile2, y_tile2 = tile_box
    tile_tensor = torch.tensor([x_tile, y_tile, x_tile2, y_tile2], device=boxes.device, dtype=boxes.dtype)

    # Filter zero-area boxes
    box_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    valid = box_area > 0
    boxes, classes, scores, box_area = boxes[valid], classes[valid], scores[valid], box_area[valid]

    if boxes.shape[0] == 0:
        return torch.zeros((0, 6), device=boxes.device, dtype=boxes.dtype)

    # Compute intersection
    inter_area = compute_intersection_area_tensor(tile_tensor.unsqueeze(0), boxes).squeeze(0)
    inside_ratio = inter_area / box_area
    keep = inside_ratio >= min_inside_ratio
    boxes, classes, scores = boxes[keep], classes[keep], scores[keep]

    if boxes.shape[0] == 0:
        return torch.zeros((0, 6), device=boxes.device, dtype=boxes.dtype)

    # Clip boxes to tile
    cx1 = torch.maximum(boxes[:, 0], torch.tensor(x_tile, device=boxes.device))
    cy1 = torch.maximum(boxes[:, 1], torch.tensor(y_tile, device=boxes.device))
    cx2 = torch.minimum(boxes[:, 2], torch.tensor(x_tile2, device=boxes.device))
    cy2 = torch.minimum(boxes[:, 3], torch.tensor(y_tile2, device=boxes.device))

    box_w = cx2 - cx1
    box_h = cy2 - cy1
    box_xc = cx1 + box_w / 2
    box_yc = cy1 + box_h / 2

    nx = (box_xc - x_tile) / tile_size
    ny = (box_yc - y_tile) / tile_size
    nw = box_w / tile_size
    nh = box_h / tile_size

    return torch.stack([classes, nx, ny, nw, nh, scores], dim=1)  # [K, 6]

def get_labels_per_tile_tensor(image, boxes, classes, scores, tile_size=640, stride=440, min_inside_ratio=0.8):
    padded_img, orig_w, orig_h = pad_image_to_multiple(image, tile_size=tile_size)
    h, w = padded_img.shape[:2]

    tiles_labels = []
    for y in range(0, h - tile_size + 1, stride):
        for x in range(0, w - tile_size + 1, stride):
            tile_box = (x, y, x + tile_size, y + tile_size)
            tile_labels = save_tile_labels_to_tensor(boxes, classes, scores, tile_box, tile_size, min_inside_ratio)
            tiles_labels.append(tile_labels)

    # Return as a list of tensors (one per tile)
    return tiles_labels


def load_label_tiles(label_dir, filename, tile_size=640, device='cuda'):
    """
    Loads YOLO label tiles for a given image.

    Returns a list of tensors [M,6] per tile: [class, xc, yc, w, h, score]
    (YOLO format, normalized to the tile).
    """
    def extract_tile_number(f):
        # Extract tile number from filename like "example_image_tile_3.txt"
        num_part = f.split("_tile_")[-1].replace('.txt','')
        return int(num_part)

    # Get label files for the given image
    label_files = [f for f in os.listdir(label_dir) if f.startswith(os.path.splitext(filename)[0])]
    label_files = sorted(label_files, key=extract_tile_number)

    label_tiles_tensors = []

    for f in label_files:
        file_path = os.path.join(label_dir, f)
        with open(file_path, 'r') as file:
            lines = file.read().splitlines()
            if len(lines) > 0:
                tile_labels = []
                for line in lines:
                    parts = list(map(float, line.split()))
                    cls, xc, yc, w, h = parts[:5]
                    score = parts[5] if len(parts) > 5 else 1.0  # default score 1.0 if missing
                    tile_labels.append([cls, xc, yc, w, h, score])
                tile_tensor = torch.tensor(tile_labels, dtype=torch.float32, device=device)
            else:
                tile_tensor = torch.empty((0,6), dtype=torch.float32, device=device)
            label_tiles_tensors.append(tile_tensor)

    return label_tiles_tensors



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

    min_area = torch.min(pred_areas[:, None], gt_areas[None, :])
    containment = inter / (min_area + 1e-6)

    # Class matching
    class_match = pred_classes[:, None] == gt_classes[None, :]
    match_matrix = class_match & ((iou >= iou_threshold) | (containment >= containment_threshold))

    matched_pred = torch.zeros(pred_boxes.size(0), dtype=torch.bool, device=device)
    matched_gt   = torch.zeros(gt_boxes.size(0), dtype=torch.bool, device=device)

    tp_boxes, tp_classes, tp_scores = [], [], []
    fp_boxes, fp_classes, fp_scores = [], [], []

    # Greedy matching
    for i in range(pred_boxes.size(0)):
        possible = torch.where(match_matrix[i] & ~matched_gt)[0]
        if possible.numel() > 0:
            j = possible[0].item()
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

def xyxy_to_yolo(x1, y1, x2, y2, tile_size=640):
    """
    Convert a box from [xmin, ymin, xmax, ymax] pixel coords to YOLO normalized format.
    """
    x_center = (x1 + x2) / 2 / tile_size
    y_center = (y1 + y2) / 2 / tile_size
    w = (x2 - x1) / tile_size
    h = (y2 - y1) / tile_size
    return [x_center, y_center, w, h]    