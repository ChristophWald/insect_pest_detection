from modules import draw_boxes
import os

#maybe the crawling could be replace by the get_files_by_subfolder func in modules?

'''
walks through all subdirectories, checks for .jpg-files and if there is a corresponding label file
if yes, a .jpg file with the visible bounding boxes is saved in a folder images_w_bboxes

Test:

image_path = "/user/christoph.wald/u15287/big-scratch/dataset/images/FungusGnats/BRAIIM_0001.jpg"
label_path = "/user/christoph.wald/u15287/big-scratch/dataset/labels/FungusGnats/BRAIIM_0001.txt"
output_path = "/user/christoph.wald/u15287/insect_pest_detection/test_box_drawing.jpg"

draw_yolo_boxes(image_path, label_path, output_path)
'''

data_root = "/user/christoph.wald/u15287/big-scratch/dataset"
#data_root = "/user/christoph.wald/u15287/insect_pest_detection/test_tiles"
image_root = os.path.join(data_root, "images")
label_root = os.path.join(data_root, "labels")
output_root = os.path.join(data_root, "images_w_bboxes")
for root, dirs, files in os.walk(image_root):
    for file in files:
        if not file.lower().endswith(('.jpg')):
            continue

        image_path = os.path.join(root, file)
        rel_path = os.path.relpath(image_path, image_root)
        rel_base, _ = os.path.splitext(rel_path)

        annotation_path = os.path.join(label_root, rel_base + '.txt')
        output_path = os.path.join(output_root, rel_base + '_with_boxes.jpg')
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        if not os.path.exists(annotation_path):
            #print(f"Label not found for: {rel_path}, skipping.")
            continue

        draw_boxes(image_path, annotation_path, output_path)
