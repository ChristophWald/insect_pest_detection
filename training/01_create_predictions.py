import sys
sys.path.append("/user/christoph.wald/u15287/insect_pest_detection/modules")
from ultralytics import YOLO
from modules_prediction import * 
import time
import json
from modules import load_yolo_labels

'''
creates a json with per image (and tile) lists of true positives, false positives and false negatives
each entry contains bounding box in absolute xyxy and confidence if given
slightly different workflows for tiles and full images
'''


def predict_on_tiles(model, image_dir, label_dir, output_dir):
    start = time.time()

    print("Predicting on the tiles.")
   

    # output dict structured by FN/FP/TP -> species -> image
    json_results = {"FN": {}, "FP": {}, "TP": {}}

    # cycles through all images in a directory
    for filename in os.listdir(image_dir):
        if filename not in os.listdir("/user/christoph.wald/u15287/big-scratch/02_splitted_data/train_labeled/split/images/train"): #skip val images
            continue
        if filename.startswith("FRANOC"): #skip thrips images
            continue
        print(f"Predicting on {filename}.")
        image_path = os.path.join(image_dir, filename)
        image = cv2.imread(image_path)

        # predicts on the full images (with stride 420)
        boxes, confs, class_ids = sliding_window_prediction(image, model)
        
        
        
        if boxes.numel() > 0:
            boxes, confs, class_ids = nms(boxes, confs, class_ids, iou_threshold=0.4, device=model.device)
            
            boxes, confs, class_ids = filter_mostly_contained_boxes(boxes, confs, class_ids, threshold=0.5)
            #print(f"Predicted: {boxes.size(0)}")
        
        pred_tiles_data = get_labels_per_tile_tensor(image, boxes, class_ids, confs)


        label_tiles_data = load_label_tiles(label_dir, filename)
        results = []

        

        # This is the per-tile loop:
        for tile_id, (pred, label) in enumerate(zip(pred_tiles_data, label_tiles_data)):
            # pred = predictions for tile i
            # label = labels for tile i

            
            if pred.numel() == 0:
                pred_boxes = torch.empty((0, 4), device='cuda')
                pred_classes = torch.empty((0,), dtype=torch.long, device='cuda')
                pred_scores = torch.empty((0,), device='cuda')
            else:
                pred_boxes = pred[:, 1:5]
                pred_classes = pred[:, 0].long()
                pred_scores = pred[:, 5]

            if label.numel() == 0:
                gt_boxes = torch.empty((0, 4), device='cuda')
                gt_classes = torch.empty((0,), dtype=torch.long, device='cuda')
            else:
                gt_boxes = label[:, 1:5]
                gt_classes = label[:, 0].long()

            tp, fp, fn = compare_labels_vectorized(
                pred_boxes, pred_classes, pred_scores, gt_boxes, gt_classes
            )
            species = filename.split("_")[0]  # extract species from filename

            

            # --- Add entries to JSON with tile_id ---
            for category, items in zip(["TP", "FP", "FN"], [tp, fp, fn]):
                boxes, classes, scores = items if category != "FN" else (*items, [None]*len(items[0]))
                if len(classes) > 0:
                    json_results.setdefault(category, {}).setdefault(species, {}).setdefault(filename, [])
                    for cls, box, score in zip(classes, boxes, scores):
                        entry = {"tile_id": tile_id}
                        if category != "FN":
                            entry["prediction"] = [cls, *box, score]
                        else:
                            entry["prediction"] = [cls, *box]
                        #print(category,entry)
                        json_results[category][species][filename].append(entry)

    with open(os.path.join(output_dir, f'predictions.json'), 'w') as f:
        json.dump(json_results, f, indent=4)

    end = time.time()
    print(f"Predicting took {end-start:.2f} seconds.")
    start = end

def predict_on_images(model, image_dir, label_dir, output_dir):
    start = time.time()
    print("Predicting on full images.")

    # Structured output: FN / FP / TP -> species -> image
    json_results = {"FN": {}, "FP": {}, "TP": {}}

    # Loop through images
    for filename in sorted(os.listdir(image_dir)):
        if filename.startswith("FRANOC"): #skip thrips images
            continue
        print(f"Predicting on {filename}...")
        image_path = os.path.join(image_dir, filename)
        image = cv2.imread(image_path)

        # loading labels
        label_path = os.path.join(label_dir, os.path.splitext(filename)[0] + ".txt")
        if not os.path.exists(label_path):
            print(f"No label file for {filename}, treating as empty GT.")
            gt_boxes = torch.empty((0, 4), device='cuda')
            gt_classes = torch.empty((0,), dtype=torch.long, device='cuda')
        else:
            # --- Load ground-truth labels ---
            gt_boxes_list, gt_classes_list = load_yolo_labels(label_path, image.shape[1], image.shape[0])
            if len(gt_boxes_list) == 0:
                gt_boxes = torch.empty((0, 4), device='cuda')
                gt_classes = torch.empty((0,), dtype=torch.long, device='cuda')
            else:
                gt_boxes = torch.tensor(gt_boxes_list, device='cuda', dtype=torch.float32)
                gt_classes = torch.tensor(gt_classes_list, device='cuda', dtype=torch.long)

        #predict on image
        result = model(image, conf=0.0, iou=0.0, verbose=False, augment=True)
        predictions = result[0].boxes

        if predictions is None or len(predictions) == 0:
            boxes, confs, class_ids = torch.empty((0, 4)), torch.empty((0,)), torch.empty((0,))
        else:
            boxes = predictions.xyxy
            confs = predictions.conf
            class_ids = predictions.cls

        # filter predictions
        if len(boxes) > 0:
            boxes, confs, class_ids = nms(boxes, confs, class_ids, iou_threshold=0.4)
            boxes, confs, class_ids = filter_mostly_contained_boxes(boxes, confs, class_ids, threshold=0.5)

        # compare predicitions and ground truth
        tp, fp, fn = compare_labels_vectorized(
            boxes, class_ids, confs, gt_boxes, gt_classes,
            tile_size=640, iou_threshold=0.5, containment_threshold=0.5,
            convert_to_xyxy=False
        )

        species = filename.split("_")[0]

        # store results
        for category, items in zip(["TP", "FP", "FN"], [tp, fp, fn]):
            if category != "FN":
                det_boxes, det_classes, det_scores = items
              
            else:
                det_boxes, det_classes = items
                det_scores = [None] * len(det_boxes)

            if len(det_classes) > 0:
                json_results.setdefault(category, {}).setdefault(species, {}).setdefault(filename, [])
                for cls, box, score in zip(det_classes, det_boxes, det_scores):
                    entry = {"tile_id": None}  # full image, so no tile
                    if category != "FN":
                        entry["prediction"] = [int(cls)] + [float(x) for x in box] + [float(score)]
                    else:
                        entry["prediction"] = [int(cls)] + [float(x) for x in box]
                    json_results[category][species][filename].append(entry)

    # save json
    output_path = os.path.join(output_dir, f"predictions_fullimage.json")
    with open(output_path, "w") as f:
        json.dump(json_results, f, indent=4)

    end = time.time()
    print(f"Predicting took {end - start:.2f} seconds.")
    print(f"Results saved to: {output_path}")

'''
model = YOLO(f"/user/christoph.wald/u15287/insect_pest_detection/2_3_supervised_training/runs/detect/train1/weights/best.pt")
    
# full images folder (predicting on cropped image, because these are all rotated
image_dir = "/user/christoph.wald/u15287/big-scratch/02_splitted_data/train_labeled/SSL/04_images_cropped"
# tile labels folder
label_dir = "/user/christoph.wald/u15287/big-scratch/02_splitted_data/train_labeled/SSL/tiles_mininside08/labels"
#output folder for predictions
output_dir = f"/user/christoph.wald/u15287/insect_pest_detection/2_5_self_training/predictions"
os.makedirs(output_dir, exist_ok=True)

#predict_on_tiles(model, image_dir, label_dir, output_dir)
'''

 # Load YOLO model
model = YOLO(f"/user/christoph.wald/u15287/insect_pest_detection/results/self_training_evaluation/runs/detect/train1/weights/best.pt")

# Full images folder
#image_dir = "/user/christoph.wald/u15287/big-scratch/04_SSL_training_data/training_data_reduced/images/train"
#image_dir = "/user/christoph.wald/u15287/big-scratch/02_splitted_data/train_labeled/split/images/val"
image_dir = "/user/christoph.wald/u15287/big-scratch/02_splitted_data/test_set/test_set_w_new_labels/images"

# Ground-truth label folder (for full images)
#label_dir = "/user/christoph.wald/u15287/big-scratch/04_SSL_training_data/training_data_reduced/labels/train"
#label_dir = "/user/christoph.wald/u15287/big-scratch/02_splitted_data/train_labeled/split/labels/val"
label_dir = "/user/christoph.wald/u15287/big-scratch/02_splitted_data/test_set/test_set_w_new_labels/labels"

# Output folder for predictions
output_dir = f"/user/christoph.wald/u15287/insect_pest_detection/training/predictions"
os.makedirs(output_dir, exist_ok=True)

predict_on_images(model, image_dir, label_dir, output_dir)
