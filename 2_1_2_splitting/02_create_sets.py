import json
import os
import shutil

'''
copies image/label files according to a split given by json files created with 01a_train_test_templates
realizes planned split
'''


json_dir = 'split_info'  
dataset_images_root = '/user/christoph.wald/u15287/big-scratch/01_dataset/images'
dataset_labels_root = '/user/christoph.wald/u15287/big-scratch/01_dataset/labels'
base_output_dir = "/user/christoph.wald/u15287/big-scratch/02_splitted_data"
image_ext = '.jpg'
label_ext = '.txt'


json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]

for json_file in json_files:
    json_path = os.path.join(json_dir, json_file)

    # setup folders
    base_name = os.path.splitext(json_file)[0]
    output_dir = os.path.join(base_output_dir, base_name)
    images_out_dir = os.path.join(output_dir, 'images')
    labels_out_dir = os.path.join(output_dir, 'labels')
    os.makedirs(images_out_dir, exist_ok=True)
    os.makedirs(labels_out_dir, exist_ok=True)

    with open(json_path, 'r') as f:
        data = json.load(f)

    for class_name, items in data.items():
        for key in items.keys():
            src_img = os.path.join(dataset_images_root, class_name, key + image_ext)
            src_label = os.path.join(dataset_labels_root, class_name, key + label_ext)

            dst_img = os.path.join(images_out_dir, key + image_ext)
            dst_label = os.path.join(labels_out_dir, key + label_ext)

            if os.path.exists(src_img):
                shutil.copy2(src_img, dst_img)
            else:
                print(f"[WARNING] Image not found: {src_img}")

            if os.path.exists(src_label):
                shutil.copy2(src_label, dst_label)
            else:
                print(f"[WARNING] Label not found: {src_label}")
