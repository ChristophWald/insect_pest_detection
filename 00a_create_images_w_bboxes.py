import os
import cv2
from modules import draw_box, load_yolo_labels

'''
walks through all subdirectories, checks for .jpg-files and if there is a corresponding label file
if yes, a .jpg file with the visible bounding boxes is saved in a folder images_w_bboxes
'''

data_root = "/user/christoph.wald/u15287/big-scratch/dataset"
image_root = os.path.join(data_root, "images")
label_root = os.path.join(data_root, "labels")
output_root = os.path.join(data_root, "images_w_bboxes")
for root, dirs, files in os.walk(image_root):
    for file in files:

        image_path = os.path.join(root, file)
        rel_path = os.path.relpath(image_path, image_root)
        rel_base, _ = os.path.splitext(rel_path)

        annotation_path = os.path.join(label_root, rel_base + '.txt')
        output_path = os.path.join(output_root, rel_base + '_with_boxes.jpg')
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        if not os.path.exists(annotation_path):
            #print(f"Label not found for: {rel_path}, skipping.")
            continue

        image = cv2.imread(image_path)
        boxes, labels = load_yolo_labels(annotation_path, image.shape[1], image.shape[0])
        for idx, box in enumerate(boxes):
            draw_box(image, box, (0,255,0), str(labels[idx]))

        cv2.imwrite(output_path, image)
