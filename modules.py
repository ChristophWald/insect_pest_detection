import os
import cv2

def yolo_to_bbox(yolo_label, img_width, img_height):
    '''
    converts yolo labels to values in relation to the current image
    '''
    class_id = yolo_label[0]
    cx, cy, w, h = map(float, yolo_label[1:])
    #for some unknown reason, the width and heigth values are sometimes negative
    #simply taking the absolute value solves this
    w = abs(w)
    h = abs(h)
    x1 = int((cx - w / 2) * img_width)
    y1 = int((cy - h / 2) * img_height)
    x2 = int((cx + w / 2) * img_width)
    y2 = int((cy + h / 2) * img_height)
    
    return class_id, x1, y1, x2, y2
    
def draw_boxes(image_path, labels_path, output_path):
    '''
    saves an image with the corresponding bboxes (in yolo format) drawn
    '''
    image = cv2.imread(image_path)
    height, width, _ = image.shape
    with open(labels_path, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        parts = line.strip().split()
        class_id, x1, y1, x2, y2 = yolo_to_bbox(parts, width, height)
        color = (0, 255, 0)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(output_path, image)
    print(f"Saved annotated image to: {output_path}")

def extract_boxes(image_path, labels_path, output_path):
    '''
    saves bounding boxes (in yolo format) of one image as single jpgs
    '''
    img = cv2.imread(image_path)
    img_height, img_width = img.shape[:2]

    with open(labels_path, 'r') as f:
        lines = f.readlines()

    base_name = os.path.splitext(os.path.basename(image_path))[0]

    for idx, line in enumerate(lines):
        parts = line.strip().split()
        class_id, x1, y1, x2, y2 = yolo_to_bbox(parts, img_width, img_height)
        cropped = img[y1:y2, x1:x2]

        # Create filename: baseName_classId_idx.jpg
        out_name = f"{base_name}_class{class_id}_{idx}.jpg"
        out_path = os.path.join(output_path, out_name)

        cv2.imwrite(out_path, cropped)


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

