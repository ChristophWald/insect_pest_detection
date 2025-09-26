import os
import cv2
import torch
from ultralytics import YOLO
import math
import numpy as np
import json

#comparision of labels and predictions does not take class_ids into account
#only the position of the box is of interest

# ------------------------
# Helper functions
# ------------------------
#taken from test_full_images, create_tiles and modules

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

def pad_image_to_multiple(image, tile_size=640, pad_value=(114,114,114)):
    """
    Pads image to the next multiple of tile_size.
    
    Returns:
        padded_image (np.ndarray)
        orig_w (int): original width
        orig_h (int): original height
    """
    h, w = image.shape[:2]
    pad_w = math.ceil(w / tile_size) * tile_size - w
    pad_h = math.ceil(h / tile_size) * tile_size - h
    padded = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=pad_value)
    return padded, w, h

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
    padded_img, orig_w, orig_h = pad_image_to_multiple(image, tile_size=tile_size)
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




model = YOLO("/user/christoph.wald/u15287/insect_pest_detection/2_5_self_training/runs/detect/train1/weights/best.pt")

#predict on cropped image contained in train/val
image_dirs = ["/user/christoph.wald/u15287/big-scratch/02_splitted_data/train_labeled/SSL/split/images/train",
              "/user/christoph.wald/u15287/big-scratch/02_splitted_data/train_labeled/SSL/split/images/val" ]
#compare to per tile labels
label_dirs = ["/user/christoph.wald/u15287/big-scratch/02_splitted_data/train_labeled/SSL/tiles/train/labels", 
              "/user/christoph.wald/u15287/big-scratch/02_splitted_data/train_labeled/SSL/tiles/val/labels"]
### also possible: use the labels by segmentation info and recreate the tile labels afterwards

output_dir = "/user/christoph.wald/u15287/insect_pest_detection/2_5_self_training/predictions"
os.makedirs(output_dir, exist_ok=True)

#conf_threshold = 0.25
conf_threshold = 0

#counts for boxes
sum_predictions = 0
sum_labels = 0
sum_missing = 0
sum_extra = 0

#to collect infos on confidences
conf_true_positives = []
conf_new_labels = []

#output
new_labels = {}

#cycles through train and val directories
for image_dir, label_dir in zip(image_dirs, label_dirs):
    print(image_dir)
    #cycles through all images in a directory
    for filename in os.listdir(image_dir):
        image_path = os.path.join(image_dir, filename)
        image = cv2.imread(image_path)
        print(filename)
        
        #predicts on an image
        boxes, scores, classes = sliding_window_prediction(image, model, conf_threshold)
        if boxes:
            boxes, scores, classes = nms(boxes, scores, classes, iou_threshold=0.4)
            boxes, scores, classes = filter_mostly_contained_boxes(boxes, scores, classes, threshold=0.5)
        pred_tiles_data = get_labels_per_tile(image, boxes, classes, scores)
        #predictions are save as lists of tile
        #each tile is a list of labels, given as [class_id, x,y,w,h,conf]
        
        #loads given labels
        label_files = [f for f in os.listdir(label_dir) if f.startswith(os.path.splitext(filename)[0])]
        label_files = sorted(label_files, key=extract_tile_number)
        label_tiles_data = []
        for f in label_files:
            file_path = os.path.join(label_dir, f)
            with open(file_path, 'r') as file:
                lines = file.read().splitlines()
                tile_labels = [list(map(float, line.split())) for line in lines]
                label_tiles_data.append(tile_labels)
        #same structure as above, but without conf

        #give number of labels/predictions
        total_pred_boxes = sum(len(tile_labels) for tile_labels in pred_tiles_data)
        total_gt_boxes = sum(len(tile_labels) for tile_labels in label_tiles_data)
        sum_predictions += total_pred_boxes
        sum_labels += total_gt_boxes
        print(f"Total predicted boxes: {total_pred_boxes}")
        print(f"Total ground-truth boxes: {total_gt_boxes}")


        missing_labels = []  # ground-truth not matched by predictions
        extra_preds = []     # predictions not matched by ground-truth

        #find missing labels and potential new labels
        for tile_id, (tile_pred, tile_lab) in enumerate(zip(pred_tiles_data, label_tiles_data)):

            # Check for missing labels
            for label in tile_lab:
                _, xl, yl, wl, hl = label
                matched = False
                for pred in tile_pred:
                    _, x, y, w, h, *rest = pred  # allow extra info if exists
                    if compute_iop([x, y, w, h], [xl, yl, wl, hl]) > 0.8:
                        matched = True
                        break
                if not matched:
                    missing_labels.append({'tile_id': tile_id, 'label': label})

            # Check for extra predictions
            for pred in tile_pred:
                _, x, y, w, h, conf = pred
                matched = False
                for label in tile_lab:
                    _, xl, yl, wl, hl = label
                    if compute_iop([x, y, w, h], [xl, yl, wl, hl]) > 0.8:
                        matched = True
                        conf_true_positives.append(conf)
                        break
                if not matched:
                    conf_new_labels.append(conf)
                    extra_preds.append({'tile_id': tile_id, 'prediction': pred})


        print(f"Total missing ground-truth labels: {len(missing_labels)}")
        print(f"Total extra predictions: {len(extra_preds)}")
        sum_missing += len(missing_labels)
        sum_extra += len(extra_preds)
        new_labels[filename] = extra_preds

print("### Total statistics ###")
print(f"{sum_labels} boxes were given.")
print(f"{sum_predictions} boxes were predicted.")
print(f"{sum_missing} labels were missed.")
print(f"{sum_extra} new labels were found.")




conf_true_positives = np.array(conf_true_positives)
print(f"Mean conf of correct predictions {np.round(np.mean(conf_true_positives),2)} with standard deviation {np.round(np.std(conf_true_positives),2)}")
conf_new_labels = np.array(conf_new_labels)
print(f"Mean conf of new predictions {np.round(np.mean(conf_new_labels),2)} with standard deviation {np.round(np.std(conf_new_labels),2)}")

with open(os.path.join(output_dir,'predictions_wo_threshold.json'), 'w') as f:
    json.dump(new_labels, f, indent=4)
