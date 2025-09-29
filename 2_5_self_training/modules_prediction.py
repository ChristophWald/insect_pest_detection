import cv2
import torch
from ultralytics import YOLO
import math
#import numpy as np

def sliding_window_prediction(image, model, conf_threshold=0.5, tile_size=640, stride=420):
    height, width, _ = image.shape
    all_boxes, all_scores, all_classes = [], [], []

    num_windows_y = (height - tile_size) // stride + 1
    num_windows_x = (width - tile_size) // stride + 1

    for i in range(num_windows_y):
        for j in range(num_windows_x):
            y_start, x_start = i * stride, j * stride
            window = image[y_start:y_start+tile_size, x_start:x_start+tile_size]
            pad_bottom = max(0, y_start+tile_size - height)
            pad_right  = max(0, x_start+tile_size - width)
            window_padded = cv2.copyMakeBorder(window, 0, pad_bottom, 0, pad_right,
                                               cv2.BORDER_CONSTANT, value=(0,0,0))
            window_rgb = cv2.cvtColor(window_padded, cv2.COLOR_BGR2RGB)
            window_tensor = torch.tensor(window_rgb/255.0).permute(2,0,1).unsqueeze(0).float().to(model.device)
            results = model(window_tensor, verbose=False)
            predictions = results[0].boxes
            valid = predictions.conf > conf_threshold
            predictions = predictions[valid]

            for box, cls, conf in zip(predictions.xyxy.cpu().numpy(), predictions.cls.cpu().numpy(), predictions.conf.cpu().numpy()):
                xmin, ymin, xmax, ymax = box
                xmin += x_start; xmax += x_start
                ymin += y_start; ymax += y_start
                all_boxes.append([xmin, ymin, xmax, ymax])
                all_scores.append(conf)
                all_classes.append(int(cls))

    return all_boxes, all_scores, all_classes


def nms(boxes, scores, classes, iou_threshold=0.5):
    """
    Applies non-maximum suppression (NMS) and returns lists instead of tensors.
    """
    boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
    scores_tensor = torch.tensor(scores, dtype=torch.float32)
    classes_tensor = torch.tensor(classes, dtype=torch.int)

    keep_nms = torch.ops.torchvision.nms(boxes_tensor, scores_tensor, iou_threshold)

    # Convert tensors back to lists
    boxes_nms = boxes_tensor[keep_nms].tolist()
    scores_nms = scores_tensor[keep_nms].tolist()
    classes_nms = classes_tensor[keep_nms].tolist()

    return boxes_nms, scores_nms, classes_nms


def filter_mostly_contained_boxes(boxes, scores, classes, threshold=0.5):
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
            xi1 = max(xa1, box_b[0]); yi1 = max(ya1, box_b[1])
            xi2 = min(xa2, box_b[2]); yi2 = min(ya2, box_b[3])
            inter_area = max(0, xi2-xi1) * max(0, yi2-yi1)
            if (inter_area / area_a) >= threshold:
                mostly_contained = True
                break
        if not mostly_contained:
            keep.append(i)
    return [boxes[i] for i in keep], [scores[i] for i in keep], [classes[i] for i in keep]


def compute_intersection_area(box_a, box_b):
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b
    xi1 = max(xa1, xb1)
    yi1 = max(ya1, yb1)
    xi2 = min(xa2, xb2)
    yi2 = min(ya2, yb2)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    return inter_area

def save_tile_labels_to_list(boxes, classes, scores, tile_box, tile_size, min_inside_ratio=0.8):
    """
    Returns a list of YOLO labels for a given tile, including class, normalized coordinates, and confidence score.
    """
    x_tile, y_tile, _, _ = tile_box
    tile_labels = []

    for (xmin, ymin, xmax, ymax), cls, conf in zip(boxes, classes, scores):
        # Skip zero-area boxes
        box_area = (xmax - xmin) * (ymax - ymin)
        if box_area == 0:
            continue
        # Intersection with tile
        inter_area = compute_intersection_area(tile_box, (xmin, ymin, xmax, ymax))
        inside_ratio = inter_area / box_area
        if inside_ratio < min_inside_ratio:
            continue
        # Clip box to tile
        cx1 = max(xmin, x_tile)
        cy1 = max(ymin, y_tile)
        cx2 = min(xmax, x_tile + tile_size)
        cy2 = min(ymax, y_tile + tile_size)
        # Convert to YOLO coordinates relative to tile
        box_w = cx2 - cx1
        box_h = cy2 - cy1
        box_xc = cx1 + box_w / 2
        box_yc = cy1 + box_h / 2
        nx = (box_xc - x_tile) / tile_size
        ny = (box_yc - y_tile) / tile_size
        nw = box_w / tile_size
        nh = box_h / tile_size
        tile_labels.append([cls, nx, ny, nw, nh, conf])

    return tile_labels


def get_labels_per_tile(image, boxes, classes, scores, tile_size=640, stride=440, min_inside_ratio=0.8):
    """
    Splits boxes into tiles and returns a list of label data per tile including confidences.
    Pads image to next multiple of tile_size to match your original tiling approach.
    """
    padded_img, orig_w, orig_h = pad_to_multiple(image, tile_size=tile_size)
    h, w = padded_img.shape[:2]

    tiles_data = []
    tile_id = 0
    for y in range(0, h - tile_size + 1, stride):
        for x in range(0, w - tile_size + 1, stride):
            tile_box = (x, y, x + tile_size, y + tile_size)
            labels = save_tile_labels_to_list(boxes, classes, scores, tile_box, tile_size, min_inside_ratio)
            tiles_data.append(labels)
            tile_id += 1

    return tiles_data

def extract_tile_number(f):
    # f is like "example_image_tile_3.txt" or "_12.txt"
    # Split by "_tile_" and remove ".txt"
    num_part = f.split("_tile_")[-1].replace('.txt','')
    return int(num_part)

def compute_iop(box1, box2):
    box1 = yolo_to_xyxy(box1, 640,640)
    box2 = yolo_to_xyxy(box2, 640,640)
    inter_area = compute_intersection_area(box1, box2)
    area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
    return inter_area / min(area1, area2) if min(area1, area2) > 0 else 0

def yolo_to_xyxy(box, img_w, img_h):
    x, y, w, h = box
    cx = x * img_w
    cy = y * img_h
    bw = w * img_w
    bh = h * img_h
    xmin = cx - bw / 2
    ymin = cy - bh / 2
    xmax = cx + bw / 2
    ymax = cy + bh / 2
    return [xmin, ymin, xmax, ymax]

def pad_to_multiple(image, tile_size=640, pad_value=(114,114,114)):
    h, w = image.shape[:2]
    pad_w = math.ceil(w / tile_size) * tile_size - w
    pad_h = math.ceil(h / tile_size) * tile_size - h
    padded = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=pad_value)
    return padded, w, h  # return original width/height for label conversion