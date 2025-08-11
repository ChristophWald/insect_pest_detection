import os
import cv2
import numpy as np

'''
MODULES:
draw a single box into an image
load a yolo label and transform to absolute coordinates
visualize all yolo boxes in one image
compute intersection of two boxes
save boxes as individual jpgs
get all files in all subfolders incl. optional linecount for label files
'''


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

def visualize_yolo_boxes(image_path, label_path, output_folder):
    """
    Draw YOLO bounding boxes on a single image and save the result.

    Args:
        image_path (str): Path to the image file.
        label_path (str): Path to the corresponding YOLO label file.
        output_folder (str): Folder to save the annotated image.
    """
    if not os.path.exists(label_path):
        print(f"Label file not found for: {image_path}")
        return

    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not read image: {image_path}")
        return

    height, width = image.shape[:2]
    boxes, labels = load_yolo_labels(label_path, width, height)

    for idx, box in enumerate(boxes):
        draw_box(image, box, (0, 255, 0), str(labels[idx]))

    base_name = os.path.basename(image_path)
    name_wo_ext, _ = os.path.splitext(base_name)
    output_filename = f"{name_wo_ext}_with_boxes.jpg"
    output_path = os.path.join(output_folder, output_filename)

    os.makedirs(output_folder, exist_ok=True)
    cv2.imwrite(output_path, image)
    print(f"Saved: {output_path}")

def compute_intersection_area(box1, box2):
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])
    inter_width = max(0, xB - xA)
    inter_height = max(0, yB - yA)
    return inter_width * inter_height


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

