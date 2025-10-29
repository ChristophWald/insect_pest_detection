import os
import cv2
import numpy as np

'''
crops all images in a folder to the YST-rectangle and adjusts the corresponding yolo labels
'''

def create_binary_mask(image):
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    return mask

def crop(image):
    mask = create_binary_mask(image)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        cropped_image = image[y:y+h, x:x+w]
        return cropped_image, (x, y, w, h)
    else:
        return None, None

def adjust_yolo_labels(labels_path, crop_x, crop_y, crop_w, crop_h, orig_w, orig_h, output_labels_path):
    adjusted_labels = []
    with open(labels_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue

            class_id, cx, cy, w, h = parts
            cx = float(cx) * orig_w
            cy = float(cy) * orig_h
            w = float(w) * orig_w
            h = float(h) * orig_h

            x_min = cx - w / 2
            y_min = cy - h / 2
            x_max = cx + w / 2
            y_max = cy + h / 2

            x_min_new = x_min - crop_x
            y_min_new = y_min - crop_y
            x_max_new = x_max - crop_x
            y_max_new = y_max - crop_y

            if x_max_new <= 0 or y_max_new <= 0 or x_min_new >= crop_w or y_min_new >= crop_h:
                continue

            x_min_clipped = max(0, x_min_new)
            y_min_clipped = max(0, y_min_new)
            x_max_clipped = min(crop_w, x_max_new)
            y_max_clipped = min(crop_h, y_max_new)

            new_w = x_max_clipped - x_min_clipped
            new_h = y_max_clipped - y_min_clipped
            new_cx = x_min_clipped + new_w / 2
            new_cy = y_min_clipped + new_h / 2

            new_cx_norm = new_cx / crop_w
            new_cy_norm = new_cy / crop_h
            new_w_norm = new_w / crop_w
            new_h_norm = new_h / crop_h

            adjusted_labels.append(f"{class_id} {new_cx_norm:.6f} {new_cy_norm:.6f} {new_w_norm:.6f} {new_h_norm:.6f}")

    os.makedirs(os.path.dirname(output_labels_path), exist_ok=True)
    with open(output_labels_path, 'w') as f_out:
        for lbl in adjusted_labels:
            f_out.write(lbl + '\n')

def process_image(image_path, annotation_path, output_image_path, output_label_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Warning: Could not read image {image_path}")
        return

    orig_h, orig_w = image.shape[:2]

    cropped_image, crop_coords = crop(image)
    if cropped_image is None:
        print(f"No yellow region found in {image_path}, skipping.")
        return

    x, y, w, h = crop_coords

    # Save cropped image
    os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
    cv2.imwrite(output_image_path, cropped_image)

    # Adjust and save labels if annotation file exists
    if os.path.exists(annotation_path):
        adjust_yolo_labels(annotation_path, x, y, w, h, orig_w, orig_h, output_label_path)
    else:
        print(f"Annotation file {annotation_path} not found, skipping label adjustment.")

def process_all_images(image_root='images', label_root='labels', output_root='output'):
    for root, dirs, files in os.walk(image_root):
        for file in files:
            if not file.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue

            image_path = os.path.join(root, file)
            rel_path = os.path.relpath(image_path, image_root)
            rel_base, ext = os.path.splitext(rel_path)

            annotation_path = os.path.join(label_root, rel_base + '.txt')

            output_image_path = os.path.join(output_root, 'images', rel_base + ext)
            output_label_path = os.path.join(output_root, 'labels', rel_base + '.txt')

            process_image(image_path, annotation_path, output_image_path, output_label_path)

if __name__ == "__main__":
    process_all_images()
