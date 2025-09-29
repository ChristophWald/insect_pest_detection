from ultralytics import YOLO
import os
import cv2
import torch 
import math

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


