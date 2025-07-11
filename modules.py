import os
import cv2

def draw_yolo_boxes(image_path, annotation_path, output_path):
    image = cv2.imread(image_path)
    height, width, _ = image.shape
    with open(annotation_path, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        parts = line.strip().split()
        x_center, y_center, box_width, box_height = map(float, parts[1:])
        x_center *= width
        y_center *= height
        box_width *= width
        box_height *= height

        x1 = int(x_center - box_width / 2)
        y1 = int(y_center - box_height / 2)
        x2 = int(x_center + box_width / 2)
        y2 = int(y_center + box_height / 2)

        color = (0, 255, 0)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)


    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(output_path, image)
    print(f"Saved annotated image to: {output_path}")