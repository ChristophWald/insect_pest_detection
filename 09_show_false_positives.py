import json
import os
from collections import defaultdict
import cv2


def load_ground_truth(labels_path):
    ground_truth = defaultdict(list)
    for filename in os.listdir(labels_path):
        if not filename.endswith(".txt"):
            continue

        print("Load " + filename)
        image_id = os.path.splitext(filename)[0]
        label_path = os.path.join(labels_path, filename)

        with open(label_path) as f:
            for line in f:
                cls_id, x, y, w, h = map(float, line.strip().split())
                x = (x - w / 2) * 640
                y = (y - h / 2) * 640
                w = w * 640
                h = h * 640

                ground_truth[image_id].append((int(cls_id), [x, y, w, h]))

    return ground_truth

def coco_to_xyxy(bbox):
    x, y, w, h = bbox
    return [x, y, x + w, y + h]

def iou(box1, box2):
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    box1Area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2Area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    unionArea = box1Area + box2Area - interArea

    return interArea / unionArea if unionArea > 0 else 0

def draw_bbox(img, bbox, label=None, color=(0, 0, 255), thickness=2):
    # bbox = [x_min, y_min, x_max, y_max]
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    if label is not None:
        cv2.putText(img, str(label), (x1 + 5, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


predictions_path = "/user/christoph.wald/u15287/big-scratch/supervised_large/runs/detect/val/predictions.json"
labels_path = "/user/christoph.wald/u15287/big-scratch/supervised_large/labels/test"
images_dir = "/user/christoph.wald/u15287/big-scratch/supervised_large/images/test"  # folder containing your images (e.g. image_id + ".png" or ".jpg")
output_dir = "/user/christoph.wald/u15287/big-scratch/supervised_large/runs/detect/val/false_positives"
os.makedirs(output_dir, exist_ok=True)

with open(predictions_path) as f:
    predictions = json.load(f)

ground_truth = load_ground_truth(labels_path)

threshold = 0.356
predictions = [p for p in predictions if p["score"] > threshold]

# Check for false positives
false_positives = []
iou_threshold = 0.5

for pred in predictions:
    img_id = pred["image_id"]
    pred_cls = pred["category_id"]-1
    pred_box = coco_to_xyxy(pred["bbox"])

    matched = False
    for gt_cls, gt_box in ground_truth[img_id]:
        if gt_cls != pred_cls:
            continue
        if iou(pred_box, coco_to_xyxy(gt_box)) >= iou_threshold:
            matched = True
            break

    if not matched:
        false_positives.append(pred)

print(f"Found {len(false_positives)} false positives.")

for fp in false_positives:
    img_id = fp["image_id"]
    bbox_xywh = fp["bbox"]
    
    # Convert bbox to xyxy format
    x, y, w, h = bbox_xywh
    bbox_xyxy = [x, y, x + w, y + h]

    # Load image: try PNG and JPG extensions
    img_path_png = os.path.join(images_dir, img_id + ".png")
    img_path_jpg = os.path.join(images_dir, img_id + ".jpg")

    if os.path.exists(img_path_png):
        img_path = img_path_png
    elif os.path.exists(img_path_jpg):
        img_path = img_path_jpg
    else:
        print(f"Image {img_id} not found, skipping.")
        continue

    img = cv2.imread(img_path)
    if img is None:
        print(f"Failed to load image {img_path}, skipping.")
        continue

    draw_bbox(img, bbox_xyxy, label=fp["category_id"] - 1, color=(0, 0, 255), thickness=2)

    # Save image with the same name to output_dir
    save_path = os.path.join(output_dir, img_id + ".png")
    cv2.imwrite(save_path, img)
