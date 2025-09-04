import os
import re
import json
import cv2

'''
for improving the labels of the testset to get less false positives

transforms predicted boxes, which were manual selected as true positive to yolo labels
filename of box gives as [Species]_[IMGnumber]_[boxnumber].jpg
predicitions with boxes coordinates given by a json saved by 03a_test_full_images.py
'''

def convert_to_yolo(box, img_width, img_height):
    '''
    converts boxes given in absolute values to normalized yolo-format
    '''
    x1, y1, x2, y2 = box
    x_center = (x1 + x2) / 2.0 / img_width
    y_center = (y1 + y2) / 2.0 / img_height
    width = (x2 - x1) / img_width
    height = (y2 - y1) / img_height
    return x_center, y_center, width, height

def merge_labels(original_label_folder, new_label_folder, merged_label_folder):
    '''
    merges the new labels with the already existing ones
    '''
    for new_label_file in os.listdir(new_label_folder):
        
        base_name = os.path.splitext(new_label_file)[0]

        original_label_path = os.path.join(original_label_folder, new_label_file)
        new_label_path = os.path.join(new_label_folder, new_label_file)
        merged_label_path = os.path.join(merged_label_folder, new_label_file)

        original_lines = []
        with open(original_label_path, 'r') as f:
            original_lines = f.read().strip().split('\n')

        with open(new_label_path, 'r') as f:
            new_lines = f.read().strip().split('\n')

        combined_lines = [line for line in original_lines + new_lines if line.strip() != ""]

        with open(merged_label_path, 'w') as f:
            f.write('\n'.join(combined_lines))

json_path = "/user/christoph.wald/u15287/big-scratch/supervised_large/tests_train1/test_conf465/results_conf465.json" #predictions
boxes_folder = "/user/christoph.wald/u15287/big-scratch/improve_test_set" #boxes selected as true positive
original_images_folder = "/user/christoph.wald/u15287/big-scratch/splitted_data/test_set/images" #images predicted on initially
original_label_folder = "/user/christoph.wald/u15287/big-scratch/splitted_data/test_set/labels" #labels initially used
output_label_folder = "/user/christoph.wald/u15287/big-scratch/improve_test_set/tile_labels" #output for yolo_format label files
merged_label_folder = "/user/christoph.wald/u15287/big-scratch/improve_test_set/merged_labels" #output for merged yolo format files
os.makedirs(merged_label_folder, exist_ok=True)
os.makedirs(output_label_folder, exist_ok=True)

#load json and built a dict
with open(json_path) as f:
    data = json.load(f)

box_class_dict = {}
for entry in data:
    filename = entry['filename'].rsplit('.', 1)[0]
    false_positives = entry.get('false_positives', {})
    boxes = false_positives.get('boxes', [])
    classes = false_positives.get('classes', [])
    box_class_pairs = [[list(map(int, box)), cls] for box, cls in zip(boxes, classes)]
    box_class_dict[filename] = box_class_pairs

#get true positive filenames and box indices from saved boxes
pattern = re.compile(r'^(.*)_(\d+)\.jpg$')
available_indices = {}

for fname in os.listdir(boxes_folder):
    match = pattern.match(fname)
    if match:
        base_name, idx_str = match.groups()
        idx = int(idx_str)
        available_indices.setdefault(base_name, set()).add(idx)

#filter the dict to keep only the relevant boxes
filtered_box_class_dict = {}

for base_name, index_set in available_indices.items():
    if base_name in box_class_dict:
        box_class_pairs = box_class_dict[base_name]
        filtered_pairs = [
            pair for i, pair in enumerate(box_class_pairs)
            if i in index_set
        ]
        if filtered_pairs:
            filtered_box_class_dict[base_name] = filtered_pairs

#write YOLO-format .txt files
for base_name, box_class_list in filtered_box_class_dict.items():
    original_image_path = os.path.join(original_images_folder, f"{base_name}.jpg")
    img = cv2.imread(original_image_path)
    
    img_height, img_width = img.shape[:2]

    # Write YOLO labels
    label_lines = []
    for box, class_id in box_class_list:
        x_center, y_center, width, height = convert_to_yolo(box, img_width, img_height)
        label_line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
        label_lines.append(label_line)

    # Save label file
    label_filename = f"{base_name}.txt"
    label_path = os.path.join(output_label_folder, label_filename)
    with open(label_path, "w") as f:
        f.write("\n".join(label_lines))


#create new labels
merge_labels(original_label_folder, output_label_folder, merged_label_folder)