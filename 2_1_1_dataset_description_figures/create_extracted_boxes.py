from modules.modules import load_yolo_labels, save_cropped_boxes
import os
import cv2

'''
for visual inspection of labels

takes all labels from a directory and puts out all bounding boxes as individual jpgs
'''

species = "Thrips"
base_path = "/user/christoph.wald/u15287/big-scratch/dataset"
base_image_path = os.path.join(base_path, "images", species)
base_label_path = os.path.join(base_path, "labels", species)
output_path = os.path.join(base_path, species + "_boxes")
os.makedirs(output_path, exist_ok = True)

label_files = os.listdir(base_label_path)

for file in label_files:
    base_name = os.path.splitext(file)[0]
    image_path = os.path.join(base_image_path, base_name + ".jpg")
    
    print(f"Processing {image_path}")
    label_path = os.path.join(base_label_path, file)
    
    image = cv2.imread(image_path)
    boxes, _= load_yolo_labels(label_path, image.shape[1], image.shape[0])
    save_cropped_boxes(image, boxes, file, output_path)
