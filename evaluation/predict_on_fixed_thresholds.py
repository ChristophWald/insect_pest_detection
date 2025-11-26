import sys
sys.path.append("/user/christoph.wald/u15287/insect_pest_detection/modules")
from ultralytics import YOLO
import time
import os
import json
import cv2
import shutil
from modules_prediction import *
from modules import load_yolo_labels
from modules_evaluation import *
import random
import os
import yaml

'''
predicts on an image set and compare to labels with general confidence threshold and additional per class confidences 
saves predictions and status (TP/FP/FN) in results.json
saves precision and recall in metrics.json
'''

def evaluate(conf_threshold, #general confidence threshold for all classes
             model, #model number
             base_output_path,
             save_images = False, #if True, saves images with bounding boxes predicted 
             save_results = True,  #if True, saves predicted boxes (results.json) and calculated metrics (metrics.json)
             skip_FRANOC = True, #skips thrips if True
             per_class_confs = None, #dictionary with per class confidence thresholds
             predict_on_tiles = False, #if False, predicts on images
             set = "test" #test: test set with revised labels, old_test: test set with unrevised labels, val: validation set
             ):
    

    start = time.time()

    results = []

    
    if set == "test":
        base_input_path = "/user/christoph.wald/u15287/big-scratch/02_splitted_data/test_set/test_set_w_new_labels"
        base_image_path = os.path.join(base_input_path, "images") 
        base_label_path = os.path.join(base_input_path, "labels")
    if set == "old_test" :
        base_input_path = "/user/christoph.wald/u15287/big-scratch/02_splitted_data/test_set/test_set_w_old_labels"
        base_image_path = os.path.join(base_input_path, "images") 
        base_label_path = os.path.join(base_input_path, "labels")
    if set == "val":
        base_input_path = "/user/christoph.wald/u15287/big-scratch/02_splitted_data/train_labeled/split"
        base_image_path = os.path.join(base_input_path, "images/val") 
        base_label_path = os.path.join(base_input_path, "labels/val") 

   
    #collecting test files
    filenames = os.listdir(base_image_path)
    filenames.sort()

    if save_images:
        image_output_path = os.path.join(base_output_path, "images_w_bboxes")
        os.makedirs(image_output_path, exist_ok=True)

    for filename in filenames:
        if skip_FRANOC and filename.startswith("FRANOC"):
            #print("skipping " + filename)
            continue

        label_path = os.path.join(base_label_path, os.path.splitext(filename)[0] + ".txt")
        if not os.path.exists(label_path):
            continue

        #print(f"Processing {filename}...")
        image = cv2.imread(os.path.join(base_image_path, filename))
        #prediction on tiles
        if predict_on_tiles:
            boxes, confs, class_ids = sliding_window_prediction(image, model, conf_threshold = conf_threshold)
            
        #prediction on full images     
        else:
            result = model(image, conf=conf_threshold, iou=0.0, verbose=False, augment=True)
            predictions = result[0].boxes

            if predictions is None or len(predictions) == 0:
                boxes = torch.empty((0, 4), dtype=torch.float32, device = "cuda")
                confs = torch.empty((0,), dtype=torch.float32, device = "cuda")
                class_ids = torch.empty((0,), dtype=torch.int64, device = "cuda")
            else:
                boxes = predictions.xyxy.to("cuda")
                confs = predictions.conf.to("cuda")
                class_ids = predictions.cls.to("cuda")

        
        #optional per class confidence filtering
        if per_class_confs is not None:
            boxes, confs, class_ids = filter_by_class_confidence(boxes, confs, class_ids, per_class_confs)

        #filtering the predictions for overlapping or contained boxes
        if len(boxes) > 0:
            boxes, confs, class_ids = nms(boxes, confs, class_ids, iou_threshold=0.4) 
            boxes, confs, class_ids = filter_mostly_contained_boxes(boxes, confs, class_ids, threshold=0.5)    

        #load labels
        label_path = os.path.join(base_label_path, os.path.splitext(filename)[0] + ".txt")
        label_boxes, label_classes_ids = load_yolo_labels(label_path, image.shape[1], image.shape[0])
        label_boxes = torch.tensor(label_boxes).to("cuda")
        label_classes_ids = torch.tensor(label_classes_ids).to("cuda")

        #compare to ground truth
        tp, fp, fn = compare_labels_vectorized(boxes, class_ids, confs, label_boxes, label_classes_ids,
                                               tile_size = 640, iou_threshold=0.5, containment_threshold=0.5, 
                                               convert_to_xyxy=False)

        


        #save results
        results.append([filename, tp, fp, fn])
        if save_images: make_image_with_boxes(image, tp, fp, fn, image_output_path, filename)    
        metrics = compute_metrics(results)
        if save_results: 
            with open(os.path.join(base_output_path, "metrics.json"), "w") as f:
                json.dump(metrics, f, indent=4)
            save_results_to_json(base_output_path, results)
        
      
        end = time.time()
        print(f"Predicting took {end-start:.2f} seconds.")
        start = end


#S1 supervised on tiles
class_conf_thresholds = {0: 0.6420546174049377, 
                            1: 0.4253721833229065, 
                            2: 0.5088263750076294, 
                            3: 0.5793536305427551}

#6.2 self-trained on tiles
class_conf_thresholds = {0: 0.5729679465293884, 
                            1: 0.6294105052947998, 
                            2: 0.0, 
                            3: 0.3088245987892151}



#S4 supervised on full images
class_conf_thresholds = {0: 0.5641838312149048, 
                            1: 0.3325055241584778, 
                            2: 0.380521297454834, 
                            3: 0.5533483624458313}


#7.2 self-trained on full images
class_conf_thresholds = {0:  0.612306535243988, 
                            1:0.49190375208854675, 
                            2: 0.0, 
                            3: 0.49190375208854675}

model_number = "7_2"
model = YOLO(f"/user/christoph.wald/u15287/insect_pest_detection/results/self_training_evaluation/runs/detect/train{model_number}/weights/best.pt")
base_output_path = f"/user/christoph.wald/u15287/insect_pest_detection/results/self_training_evaluation/runs/detect/train{model_number}_test_set"


os.makedirs(base_output_path, exist_ok=True)

evaluate(conf_threshold=0.2, 
         model=model, 
         base_output_path=base_output_path,  
         save_images = False,
         skip_FRANOC = True,  
         per_class_confs = class_conf_thresholds, 
         predict_on_tiles=False,
         set = "test")


