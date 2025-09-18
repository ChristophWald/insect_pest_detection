import os
import cv2
from modules_segmentation import find_contour, is_upside_orientated

#flags for use with automated/specified rotation and for rotating labels (or not)
automated_rotating = False #if False, set individual files below
use_labels = False 

image_folder = "/user/christoph.wald/u15287/big-scratch/02_splitted_data/train_unlabeled/images_uncropped"
image_files = os.listdir(image_folder)

labels_folder = "/user/christoph.wald/u15287/big-scratch/00_uncropped_dataset/labels"
output_labels_folder = "/user/christoph.wald/u15287/big-scratch/02_splitted_data/train_labeled/labels_uncropped"
output_images_folder = "/user/christoph.wald/u15287/big-scratch/02_splitted_data/train_unlabeled/images_uncropped"

if use_labels:
    os.makedirs(output_labels_folder, exist_ok=True)


def rotate_yolo_labels(txt_path, save_path):
    rotated_lines = []
    with open(txt_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls, x, y, w, h = parts
            x, y, w, h = map(float, (x, y, w, h))
            # rotate 180°: flip both x and y
            x = 1.0 - x
            y = 1.0 - y
            rotated_lines.append(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

    with open(save_path, "w") as f:
        f.writelines(rotated_lines)


def copy_labels(txt_path, save_path):
    with open(txt_path, "r") as src, open(save_path, "w") as dst:
        dst.write(src.read())


#for automatic rotating
if automated_rotating:
    for file in image_files:
        print(f"Loading {file}.")
        path = os.path.join(image_folder, file)
        image = cv2.imread(path)

        output_image_path = os.path.join(output_images_folder, file)

        # Label file paths
        if use_labels:
            label_file = os.path.splitext(file)[0] + ".txt"
            label_path = os.path.join(labels_folder, label_file)
            output_label_path = os.path.join(output_labels_folder, label_file)

        #find YST contour and check for orientation
        imageYST = find_contour(image)
        if not is_upside_orientated(image, imageYST):
            print("Rotating!")
            image = cv2.rotate(image, cv2.ROTATE_180)
            cv2.imwrite(output_image_path, image)

            if use_labels:
                rotate_yolo_labels(label_path, output_label_path)
        else:
            cv2.imwrite(output_image_path, image)

            if use_labels:
                copy_labels(label_path, output_label_path)


#for rotating specified images
else:
    #falsly_detected_images = ["BRAIIM_0042.jpg", "BRAIIM_0635.jpg", "BRAIIM_0666.jpg", "BRAIIM_0668.jpg"] #labeled training set
    falsly_detected_images = ["BRAIIM_0112.jpg", "BRAIIM_0142.jpg","LIRIBO_1263.jpg", "LIRIBO_1294.jpg"] #unlabeled training set
    for file in falsly_detected_images:
        print(f"Loading {file}.")
        path = os.path.join(image_folder, file)
        image = cv2.imread(path)

        output_image_path = os.path.join(output_images_folder, file)

        if use_labels:
            label_file = os.path.splitext(file)[0] + ".txt"
            label_path = os.path.join(labels_folder, label_file)
            output_label_path = os.path.join(output_labels_folder, label_file)

        #rotate image
        image = cv2.rotate(image, cv2.ROTATE_180)
        cv2.imwrite(output_image_path, image)

        if use_labels:
            rotate_yolo_labels(label_path, output_label_path)
