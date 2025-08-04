import os
import cv2
import numpy as np


#add titles to the plots
#histplot is odd because of the changing scales

def draw_box(img, box, color, label):
    """Draw a bounding box with label on the image."""
    xmin, ymin, xmax, ymax = map(int, box)
    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 2)
    cv2.putText(img, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, color, 2)
    

def load_yolo_labels(file_path, img_width, img_height):
    """Load YOLO label file and convert to absolute pixel coordinates."""
    boxes = []
    classes = []
    with open(file_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            cls = int(parts[0])
            x_center, y_center, w, h = map(float, parts[1:])
            xmin = (x_center - w / 2) * img_width
            xmax = (x_center + w / 2) * img_width
            ymin = (y_center - h / 2) * img_height
            ymax = (y_center + h / 2) * img_height
            boxes.append([xmin,ymin,xmax,ymax])
            classes.append(cls)
    return [boxes, classes]

def save_cropped_boxes(image, boxes, filename, output_dir):
    """
    Save cropped regions from the original image as separate JPG files.

    Args:
        image (np.ndarray): Original image (BGR format).
        boxes (List[List[int]]): List of bounding boxes [xmin, ymin, xmax, ymax].
        filename (str): Original image filename (used to name crops).
        output_dir (str): Directory to save cropped images.
    """
    base_name = os.path.splitext(filename)[0]
    h, w = image.shape[:2]
    
    for idx, box in enumerate(boxes):
        xmin = max(0, int(np.floor(box[0])))
        ymin = max(0, int(np.floor(box[1])))
        xmax = min(w, int(np.ceil(box[2])))
        ymax = min(h, int(np.ceil(box[3])))

        cropped = image[ymin:ymax, xmin:xmax]
        if len(cropped) == 0:
            print(xmin, ymin, xmax, ymax)
        crop_filename = f"{base_name}_{idx}.jpg"
        crop_path = os.path.join(output_dir, crop_filename)
        cv2.imwrite(crop_path, cropped)

def get_files_by_subfolder(root_dir, count_lines=False):
    '''
    Function to get filenames (and optional line counts for labels) from subfolders
    number of lines in label files is the number of labels
    '''
    file_dict = {}
    for subdir, _, files in os.walk(root_dir):
        if subdir == root_dir:
            continue  # Skip root directory itself
        folder_name = os.path.basename(subdir)
        file_entries = {}
        for f in files:
            file_path = os.path.join(subdir, f)
            if os.path.isfile(file_path):
                name = os.path.splitext(f)[0]
                if count_lines:
                    with open(file_path, 'r', encoding='utf-8') as file:
                        lines = file.readlines()
                    file_entries[name] = len(lines)
                else:
                    file_entries[name] = None
        file_dict[folder_name] = file_entries
    return file_dict

