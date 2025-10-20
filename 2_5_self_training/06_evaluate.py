import sys
sys.path.append("/user/christoph.wald/u15287/insect_pest_detection/modules")
print("Importing.")
from ultralytics import YOLO
import os
import cv2
from modules_prediction import *
from modules import load_yolo_labels
from modules_evaluation import *
import torch

#modules for the final evaluation
import glob
import re
import json
import pandas as pd
import matplotlib.pyplot as plt

import time

print("Initializing.")

save_images = False
save_results = True
skip_FRANOC = True
conf_threshold=[0.373, 0.535, 0.415, 0.461]
test_runs = len(conf_threshold)


#set in- & output path
base_input_path = "/user/christoph.wald/u15287/big-scratch/02_splitted_data/test_set"
base_image_path = os.path.join(base_input_path, "images")
base_label_path = os.path.join(base_input_path, "labels")
output_path  = "/user/christoph.wald/u15287/insect_pest_detection/2_5_self_training/metrics"
os.makedirs(output_path, exist_ok=True)

#collecting test files
filenames = os.listdir(base_image_path)
filenames.sort()
plot_histograms("/user/christoph.wald/u15287/insect_pest_detection/2_5_self_training/predictions", output_path)

#print("Plotting training curves.")
plot_prec_recall(output_path)
#print("Plot histograms of predictions on tiles.")
'''

#test runs
for i in range(test_runs):
  
    start = time.time()

    results = []

    print(f"Testing model {i+1}")
    if i == 0:
        model = YOLO(f"/user/christoph.wald/u15287/insect_pest_detection/2_5_self_training/runs/detect/train/weights/best.pt")
        base_output_path = f"/user/christoph.wald/u15287/insect_pest_detection/2_5_self_training/metrics/train"
    else:
        model = YOLO(f"/user/christoph.wald/u15287/insect_pest_detection/2_5_self_training/runs/detect/train{i+1}/weights/best.pt")
        base_output_path = f"/user/christoph.wald/u15287/insect_pest_detection/2_5_self_training/metrics/train{i+1}"
    os.makedirs(base_output_path, exist_ok=True)

    if save_images:
        image_output_path = os.path.join(base_output_path, "images_w_bboxes")
        os.makedirs(image_output_path, exist_ok=True)

    for filename in filenames:
        if skip_FRANOC and filename.startswith("FRANOC"):
            #print("skipping " + filename)
            continue
        #print(f"Processing {filename}...")
        image = cv2.imread(os.path.join(base_image_path, filename))
        boxes, confs, class_ids = sliding_window_prediction(image, model, conf_threshold[i])
        
        if len(boxes) > 0:
            boxes, confs, class_ids = nms(boxes, confs, class_ids, iou_threshold=0.4) 
            boxes, confs, class_ids = filter_mostly_contained_boxes(boxes, confs, class_ids, threshold=0.5)    
        #print(f"Predicted: {boxes.size(0)}")
    
        label_path = os.path.join(base_label_path, os.path.splitext(filename)[0] + ".txt")
        label_boxes, label_classes_ids = load_yolo_labels(label_path, image.shape[1], image.shape[0])
 
        label_boxes = torch.tensor(label_boxes).to("cuda")
        label_classes_ids = torch.tensor(label_classes_ids).to("cuda")
 
        tp, fp, fn = compare_labels_vectorized(boxes, class_ids, confs, label_boxes, label_classes_ids,
                                               tile_size = 640, iou_threshold=0.5, containment_threshold=0.8, 
                                               convert_to_xyxy=False)
        
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
'''
print("Doing the final evaluation.")

# Class name mapping
class_name_map = {
    '0': 'BRAIIM',
    '1': 'LIRIBO',
    '3': 'TRIVA'
}

import os, re

def natural_sort_key(path):
    folder = os.path.basename(os.path.dirname(path))
    # Extract the number after 'train', default to 0 if missing
    m = re.search(r'train(\d+)', folder)
    num = int(m.group(1)) if m else 0
    return num


# Collect metrics.json files
metric_files = sorted(
    glob.glob(os.path.join(output_path, 'train*/metrics.json')),
    key=natural_sort_key
)

if not metric_files:
    raise FileNotFoundError(f"No metrics files found under {output_path}")

# Read all JSON metrics
results = {}
folders = [os.path.basename(os.path.dirname(f)) for f in metric_files]
for folder, file in zip(folders, metric_files):
    with open(file, 'r') as f:
        results[folder] = json.load(f)

# Build DataFrames for summary + per_class
dfs = {}
all_keys = ['summary'] + [f"per_class.{k}" for k in results[folders[0]]['per_class'].keys()]

for key in all_keys:
    rows = []
    for folder in folders:
        if key == 'summary':
            stats = results[folder]['summary']
        else:
            class_id = key.split('.')[-1]
            stats = results[folder]['per_class'].get(class_id, {})
        row = {'folder': folder, **stats}
        rows.append(row)
    dfs[key] = pd.DataFrame(rows)

# Combine all DataFrames into one for saving
combined_df = pd.concat(dfs.values(), keys=dfs.keys(), names=['metric_type', 'index'])


# Plotting
save_dir = output_path
os.makedirs(save_dir, exist_ok=True)

x_labels = []
x_values = []
for folder in folders:
    m = re.match(r'train(\d*)', folder)
    num = int(m.group(1)) if m and m.group(1) else 0
    x_labels.append(str(num))
    x_values.append(num)

for key, df in dfs.items():
    fig, ax1 = plt.subplots(figsize=(10,6))

    # Left y-axis (TP, FP, FN)
    ax1.set_xlabel('Folder')
    ax1.set_ylabel('Counts', color='tab:blue')
    for col in ['TP','FP','FN']:
        if col in df.columns:
            ax1.plot(range(len(x_values)), df[col], marker='o', label=col)
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.grid(True, which='both', axis='both', linestyle='--', alpha=0.5)

    # Right y-axis (precision, recall)
    ax2 = ax1.twinx()
    ax2.set_ylabel('Score', color='tab:orange')
    for col in ['precision','recall']:
        if col in df.columns:
            ax2.plot(range(len(x_values)), df[col], marker='x', linestyle='--', label=col)
    ax2.set_ylim(0,1.0)
    ax2.tick_params(axis='y', labelcolor='tab:orange')
    ax2.grid(False)

    ax1.set_xticks(range(len(x_values)))
    ax1.set_xticklabels(x_labels)

    # Title & filename
    if key.startswith('per_class'):
        class_id = key.split('.')[-1]
        class_name = class_name_map.get(class_id, f'Class {class_id}')
        title = f'Metrics for {class_name}'
        filename_key = class_name
    else:
        title = 'Metrics for Summary'
        filename_key = 'summary'

    plt.title(title)

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    plt.tight_layout()
    save_path = os.path.join(save_dir, f"{filename_key}.jpg")
    plt.savefig(save_path)
    plt.close()

print(f"Plots saved in {save_dir}")
combined_df.to_csv(os.path.join(save_dir, 'all_metrics_combined.csv'))
print(f"Combined metrics saved as all_metrics_combined.csv")
