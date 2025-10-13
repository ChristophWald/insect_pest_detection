import torch
################### create_rectangles

def compute_intersection_area(box1, box2):
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])
    inter_width = max(0, xB - xA)
    inter_height = max(0, yB - yA)
    return inter_width * inter_height


def compute_iop(box1, box2):
    """
    Intersection over Prediction (IoP).
    Measures how much of the predicted box lies inside the ground truth.
    """
    inter_area = compute_intersection_area(box1, box2)
    pred_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    if pred_area == 0:
        return 0.0
    return inter_area / pred_area

def evaluate_detections(pred_rectangles, gt_rectangles, iou_threshold=0.5):
    """
    Compare predictions with YOLO ground-truth rectangles.

    
    """
    matched_gt = set()
    matched_pred = set()
    TP = 0

    # Match predictions to ground truth
    for pi, pred in enumerate(pred_rectangles):
        for gi, gt in enumerate(gt_rectangles):
            if gi in matched_gt:
                continue
            if compute_iop(pred, gt) >= iou_threshold:
                matched_gt.add(gi)
                matched_pred.add(pi)
                TP += 1
                break  # stop after first match

    # False Positives = predictions not matched
    FP_indices = [pi for pi in range(len(pred_rectangles)) if pi not in matched_pred]
    FP_boxes = [pred_rectangles[pi] for pi in FP_indices]

    FP = len(pred_rectangles) - TP
    FN = len(gt_rectangles) - TP

    return {"TP": TP, "FP": FP, "FN": FN}, FP_boxes

'''
def compute_intersection_area(box1, box2):
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])
    inter_width = max(0, xB - xA)
    inter_height = max(0, yB - yA)
    return inter_width * inter_height


def compute_iop(box1, box2):
    """
    Intersection over Prediction (IoP).
    Measures how much of the predicted box lies inside the ground truth.
    """
    inter_area = compute_intersection_area(box1, box2)
    pred_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    if pred_area == 0:
        return 0.0
    return inter_area / pred_area


def evaluate_detections(pred_boxes, pred_classes, pred_scores, gt_boxes, gt_classes, iou_threshold=0.5):
    """
    Simplified evaluation using Intersection-over-Prediction (IoP) threshold.
    Returns (TP), (FP), (FN) in the same structure as compare_labels.
    """
    matched_gt = set()
    tp_boxes, tp_classes = [], []
    fp_boxes, fp_classes, fp_scores = [], [], []
    fn_boxes, fn_classes = [], []

    for pred_box, pred_cls, pred_score in zip(pred_boxes, pred_classes, pred_scores):
        matched = False
        for i, (gt_box, gt_cls) in enumerate(zip(gt_boxes, gt_classes)):
            if i in matched_gt or pred_cls != gt_cls:
                continue
            if compute_iop(pred_box, gt_box) >= iou_threshold:
                tp_boxes.append(pred_box)
                tp_classes.append(pred_cls)
                matched_gt.add(i)
                matched = True
                break
        if not matched:
            fp_boxes.append(pred_box)
            fp_classes.append(pred_cls)
            fp_scores.append(pred_score)

    for i, (gt_box, gt_cls) in enumerate(zip(gt_boxes, gt_classes)):
        if i not in matched_gt:
            fn_boxes.append(gt_box)
            fn_classes.append(gt_cls)

    return (tp_boxes, tp_classes), (fp_boxes, fp_classes, fp_scores), (fn_boxes, fn_classes)
'''


#### test_full_images

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
    pred_scores,  # list of confidence scores
    gt_boxes,     # list of ground truth boxes: [[xmin, ymin, xmax, ymax], ...]
    gt_classes,   # list of ground truth classes: [cls_id, ...]
    iou_threshold=0.5,
    containment_threshold=0.9
):
    """
    Compare predicted boxes with ground truth and return TP, FP, FN.
    Only FP confidences are stored.
    """
    matched_gt = set()

    tp_boxes, tp_classes = [], []
    fp_boxes, fp_classes, fp_scores = [], [], []
    fn_boxes, fn_classes = [], []

    for pred_box, pred_cls, pred_score in zip(pred_boxes, pred_classes, pred_scores):
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
                    # True Positive
                    tp_boxes.append(pred_box)
                    tp_classes.append(pred_cls)
                    matched_gt.add(i)
                    matched = True
                    break
        if not matched:
            # False Positive
            fp_boxes.append(pred_box)
            fp_classes.append(pred_cls)
            fp_scores.append(pred_score)

    # False Negatives
    for i, (gt_box, gt_cls) in enumerate(zip(gt_boxes, gt_classes)):
        if i not in matched_gt:
            fn_boxes.append(gt_box)
            fn_classes.append(gt_cls)

    return (tp_boxes, tp_classes), (fp_boxes, fp_classes, fp_scores), (fn_boxes, fn_classes)

####evaluate

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

