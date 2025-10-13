import os
import cv2
from modules_segmentation import find_contour, is_upside_orientated

'''
turns images by 180° either by automatic detection or by a list of images to be rotated (use flag automated rotating)
can also turn the according labels (use flag use_labels)
'''

#flags for use with automated/specified rotation and for rotating labels (or not)
automated_rotating = True #if False, set individual files below
use_labels = True 

#image_folder = "/user/christoph.wald/u15287/big-scratch/02_splitted_data/train_labeled/SSL/01_images_uncropped"
#image_folder = "/user/christoph.wald/u15287/big-scratch/02_splitted_data/train_unlabeled/01_images_uncropped"
image_folder = "/user/christoph.wald/u15287/big-scratch/02_splitted_data/test_set/SSL/01_images_uncropped"

image_files = os.listdir(image_folder)

#labels_folder = "/user/christoph.wald/u15287/big-scratch/02_splitted_data/train_labeled/SSL/01_labels_uncropped"
labels_folder = "/user/christoph.wald/u15287/big-scratch/02_splitted_data/test_set/SSL/01_labels_uncropped"

#output_images_folder = "/user/christoph.wald/u15287/big-scratch/02_splitted_data/train_labeled/SSL/02_images_rotated"
#output_labels_folder = "/user/christoph.wald/u15287/big-scratch/02_splitted_data/train_labeled/SSL/02_labels_rotated"
output_images_folder = "/user/christoph.wald/u15287/big-scratch/02_splitted_data/train_unlabeled/02_images_rotated"
output_labels_folder = "/user/christoph.wald/u15287/big-scratch/02_splitted_data/train_unlabeled/02_labels_rotated"
output_images_folder = "/user/christoph.wald/u15287/big-scratch/02_splitted_data/test_set/SSL/02_images_rotated"
output_labels_folder = "/user/christoph.wald/u15287/big-scratch/02_splitted_data/test_set/SSL/02_labels_rotated"

os.makedirs(output_labels_folder, exist_ok=True)
os.makedirs(output_images_folder, exist_ok=True)


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
    rotated_files = []
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
            rotated_files.append(file)
            image = cv2.rotate(image, cv2.ROTATE_180)
            cv2.imwrite(output_image_path, image)

            if use_labels:
                rotate_yolo_labels(label_path, output_label_path)
        else:
            cv2.imwrite(output_image_path, image)

            if use_labels:
                copy_labels(label_path, output_label_path)
    with open("rotated_files.txt", "w") as f:
        f.write("\n".join(rotated_files))


#for rotating specified images
else:
    #falsly_detected_images = ["BRAIIM_0042.jpg", "BRAIIM_0635.jpg", "BRAIIM_0666.jpg", "BRAIIM_0668.jpg"] #labeled training set
    falsly_detected_images = ["BRAIIM_0112.jpg", "BRAIIM_0142.jpg","LIRIBO_1263.jpg", "LIRIBO_1294.jpg"] #unlabeled training set
    test_set = [] #all correct
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
    with open("rotated_files.txt", "a") as f:
        f.write("\n".join(falsly_detected_images) + "\n")
