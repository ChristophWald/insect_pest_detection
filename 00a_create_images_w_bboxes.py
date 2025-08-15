import os
import cv2
from modules import draw_box, load_yolo_labels, visualize_yolo_boxes

'''
for visual inspection of the labels

plotting the images with the boxes given by yolo formatted labels
visualize_labels: uses only labels in a specific folder; corresponding image folder as specified
visualize_all_images_with_labels: uses all label folders in a given directory, presumes corresponding image folders
'''

def visualize_labels(label_root, image_root, output_path):
    '''
    iterates over the label files in label_root,
    and for each one, draws boxes on the corresponding image from image_root
    '''
    for file in os.listdir(label_root):
        label_path = os.path.join(label_root, file)
        image_name = os.path.splitext(file)[0] + ".jpg"
        image_path = os.path.join(image_root, image_name)
       
        visualize_yolo_boxes(image_path, label_path, output_path)

def visualize_all_images_with_labels(image_root, label_root, output_root):
    '''
    walk through all label files in label_root (including subdirs),
    and for each one, draws boxes on the corresponding image from image_root
    '''
    for root, dirs, files in os.walk(label_root):
        for file in files:

            label_path = os.path.join(root, file)
            rel_path = os.path.relpath(label_path, label_root)
            rel_base, _ = os.path.splitext(rel_path)

            image_path = os.path.join(image_root, rel_base + ".jpg")
            output_path = os.path.join(output_root, os.path.dirname(rel_path))

            visualize_yolo_boxes(image_path, label_path, output_path)



#this is for only one specific folder
label_root = "/user/christoph.wald/u15287/big-scratch/improve_test_set/merged_labels"
image_root = "/user/christoph.wald/u15287/big-scratch/splitted_data/test_set/images"
output_path = "/user/christoph.wald/u15287/big-scratch/improve_test_set/predicted_images"
os.makedirs(output_path, exist_ok=True)

visualize_labels(label_root, image_root, output_path)