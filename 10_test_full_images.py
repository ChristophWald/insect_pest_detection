import cv2
import torch
import numpy as np
import os
import json
from ultralytics import YOLO

def filter_mostly_contained_boxes(boxes, scores, threshold=0.8):
    """
    Removes boxes that are mostly (>= threshold) contained in other boxes with equal or higher confidence.
    """
    keep = []
    for i, box_a in enumerate(boxes):
        xa1, ya1, xa2, ya2 = box_a
        area_a = (xa2 - xa1) * (ya2 - ya1)
        mostly_contained = False

        for j, box_b in enumerate(boxes):
            if i == j:
                continue
            xb1, yb1, xb2, yb2 = box_b
            if scores[j] < scores[i]:
                continue

            # Compute intersection
            xi1 = max(xa1, xb1)
            yi1 = max(ya1, yb1)
            xi2 = min(xa2, xb2)
            yi2 = min(ya2, yb2)

            inter_w = max(0, xi2 - xi1)
            inter_h = max(0, yi2 - yi1)
            inter_area = inter_w * inter_h

            if inter_area / area_a >= threshold:
                mostly_contained = True
                break

        if not mostly_contained:
            keep.append(i)
    return keep


def sliding_window_yolo_labels_with_nms(image_path, model, window_size=(640, 640), stride=440, conf_threshold=0.5, iou_threshold=0.5):
    
    img = cv2.imread(image_path)
    height, width, _ = img.shape

    window_height, window_width = window_size
    num_windows_y = (height - window_height) // stride + 1
    num_windows_x = (width - window_width) // stride + 1

    all_boxes = []
    all_scores = []
    all_classes = []

    for i in range(num_windows_y):
        for j in range(num_windows_x):
            y_start = i * stride
            x_start = j * stride
            y_end = y_start + window_height
            x_end = x_start + window_width

            window = img[y_start:y_end, x_start:x_end]
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

    if len(all_boxes) == 0:
        return []  # No detections

    # Convert to torch tensors
    boxes_tensor = torch.tensor(all_boxes, dtype=torch.float32)
    scores_tensor = torch.tensor(all_scores, dtype=torch.float32)
    classes_tensor = torch.tensor(all_classes, dtype=torch.int)

    # Perform NMS
    keep_nms = torch.ops.torchvision.nms(boxes_tensor, scores_tensor, iou_threshold)
    boxes_nms = boxes_tensor[keep_nms]
    scores_nms = scores_tensor[keep_nms]
    classes_nms = classes_tensor[keep_nms]

    # Remove nested boxes
    keep_final = filter_mostly_contained_boxes(boxes_nms.tolist(), scores_nms.tolist(), threshold=0.9)
    boxes_final = boxes_nms[keep_final]
    scores_final = scores_nms[keep_final]
    classes_final = classes_nms[keep_final]

    # Prepare YOLO-format output
    yolo_labels = []
    for i in range(len(boxes_final)):
        xmin, ymin, xmax, ymax = boxes_final[i]
        cls = classes_final[i].item()

        # Convert to YOLO format (normalized center x, y, width, height)
        x_center = ((xmin + xmax) / 2) / width
        y_center = ((ymin + ymax) / 2) / height
        box_width = (xmax - xmin) / width
        box_height = (ymax - ymin) / height

        yolo_labels.append(f"{cls} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}")

    return yolo_labels



def load_yolo_labels(file_path, img_width, img_height):
    """Load YOLO label file and convert to absolute pixel coordinates."""
    labels = []
    with open(file_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            cls = int(parts[0])
            x_center, y_center, w, h = map(float, parts[1:])
            xmin = (x_center - w / 2) * img_width
            xmax = (x_center + w / 2) * img_width
            ymin = (y_center - h / 2) * img_height
            ymax = (y_center + h / 2) * img_height
            labels.append([cls, xmin, ymin, xmax, ymax])
    return np.array(labels)


def compute_iou(box1, box2):
    """Compute IoU between two bounding boxes."""
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    inter_area = max(0, xB - xA) * max(0, yB - yA)
    if inter_area == 0:
        return 0.0

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    iou = inter_area / (box1_area + box2_area - inter_area)
    return iou


def draw_box(img, box, color, label):
    """Draw a bounding box with label on the image."""
    xmin, ymin, xmax, ymax = map(int, box)
    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 2)
    cv2.putText(img, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, color, 2)
    

def compare_labels(
    pred_file,
    gt_file,
    image_path,
    iou_threshold=0.5,
    output_image_path=None
):
    img = cv2.imread(image_path)
    height, width, _ = img.shape

    preds = load_yolo_labels(pred_file, width, height)
    gts = load_yolo_labels(gt_file, width, height)

    matched_gt = set()
    tp = 0
    fp = 0
    fn = 0

    drawn = img.copy()

    tp_boxes = []
    fp_boxes = []
    fn_boxes = []

    for pred in preds:
        pred_cls, *pred_box = pred
        matched = False
        for i, gt in enumerate(gts):
            gt_cls, *gt_box = gt
            if i in matched_gt:
                continue
            if pred_cls == gt_cls and compute_iou(pred_box, gt_box) >= iou_threshold:
                tp += 1
                matched_gt.add(i)
                tp_boxes.append((pred_cls, pred_box))
                matched = True
                break
        if not matched:
            fp += 1
            fp_boxes.append((pred_cls, pred_box))

    for i, gt in enumerate(gts):
        if i not in matched_gt:
            fn += 1
            fn_boxes.append((int(gt[0]), gt[1:]))

    # Draw results
    for cls, box in tp_boxes:
        draw_box(drawn, box, (0, 255, 0), f"TP: {cls}")

    for cls, box in fp_boxes:
        draw_box(drawn, box, (255, 0, 0), f"FP: {cls}")

    for cls, box in fn_boxes:
        draw_box(drawn, box, (0, 0, 255), f"FN: {cls}")

    if output_image_path:
        cv2.imwrite(output_image_path, drawn)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    print(f"TP: {tp}, FP: {fp}, FN: {fn}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    if output_image_path:
        print(f"Saved comparison image to {output_image_path}")

    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "precision": precision,
        "recall": recall
    }




model = YOLO('/user/christoph.wald/u15287/big-scratch/supervised_large/runs/detect/train/weights/best.pt')

base_image_path = "/user/christoph.wald/u15287/big-scratch/splitted_data/test_set/images"
base_label_path = "/user/christoph.wald/u15287/big-scratch/splitted_data/test_set/labels"
output_labels_path = "/user/christoph.wald/u15287/big-scratch/supervised_large/runs/detect/val/predicted_labels"
image_output_path = "/user/christoph.wald/u15287/big-scratch/supervised_large/runs/detect/val/predicted_boxes"
os.makedirs(output_labels_path, exist_ok=True)
os.makedirs(image_output_path, exist_ok=True)

results = []

filenames = os.listdir(base_image_path)

for filename in filenames:
    print(filename)
    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        continue  # skip non-image files

    image = os.path.join(base_image_path, filename)
    label = os.path.join(base_label_path, os.path.splitext(filename)[0] + ".txt")
    output_image = os.path.join(image_output_path, "pred_" + os.path.splitext(filename)[0] + ".jpg")
    output_label = os.path.join(output_labels_path, "pred_" + os.path.splitext(filename)[0] + ".txt")

    # Run sliding window prediction
    labels = sliding_window_yolo_labels_with_nms(
        image_path=image,
        model=model,
        stride=420,
        conf_threshold=0.356,
        iou_threshold=0.5
    )

    # Save predicted labels
    with open(output_label, "w") as f:
        f.write("\n".join(labels))

    # Compare with ground truth and save image with boxes
    result = compare_labels(
        pred_file=output_label,
        gt_file=label,
        image_path=image,
        output_image_path=output_image
    )
    results.append(result)


output_txt_path = "/user/christoph.wald/u15287/insect_pest_detection/results_testing.txt"


with open(output_txt_path, "w") as f:
    for res in results:
        line = json.dumps(res)
        f.write(line + "\n")