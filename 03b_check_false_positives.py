import json
import os
import re
import cv2
import numpy as np

# Paths
json_path = "/home/wald/Schreibtisch/10_BA_Arbeit/01_data_preparation/runs/test_train1/test_conf465/results_conf465.json"
boxes_path = "/home/wald/Schreibtisch/10_BA_Arbeit/01_data_preparation/runs/test_train1/test_conf465/boxes"
output_folder = "/home/wald/Schreibtisch/10_BA_Arbeit/01_data_preparation/runs/test_train1/test_conf465/annotated_boxes"
os.makedirs(output_folder, exist_ok=True)

# Load JSON
with open(json_path) as f:
    data = json.load(f)
json_dict = {entry['filename'].rsplit('.', 1)[0]: entry for entry in data}

# Class names
class_names = {
    0: "FungusGnats",
    1: "LeafMinerFlies",
    2: "Thrips",
    3: "WhiteFlies"
}

# Filename regex pattern
pattern = re.compile(r'^(.*)_(\d+)\.jpg$')

# Process images
for filename in sorted(os.listdir(boxes_path)):
    match = pattern.match(filename)
    if not match:
        continue

    file_base, box_index_str = match.groups()
    box_index = int(box_index_str)
    entry = json_dict[file_base]['false_positives']
    box = [int(c) for c in entry['boxes'][box_index]]
    cls = entry['classes'][box_index]
    class_name = class_names.get(cls, f"Unknown({cls})")

    # Load image and pad height to canvas height
    image = cv2.imread(os.path.join(boxes_path, filename))
    img_h = image.shape[0]
    canvas_w, canvas_h = 300, 300
    image = cv2.copyMakeBorder(image, 0, canvas_h - img_h, 0, 0, cv2.BORDER_CONSTANT, value=[255,255,255])

    # Create canvas with class name
    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255
    cv2.putText(canvas, class_name, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    # Combine and save
    combined = np.hstack((image, canvas))
    output_path = os.path.join(output_folder, filename)
    cv2.imwrite(output_path, combined)
    print(f"Saved: {output_path}")