import json
import os
import re
import cv2
import numpy as np

# Load JSON data
with open('/user/christoph.wald/u15287/insect_pest_detection/test_prediction/results.json') as f:
    data = json.load(f)

# Create dict keyed by filename base (no extension)
json_dict = {entry['filename'].rsplit('.', 1)[0]: entry for entry in data}

input_folder = "/user/christoph.wald/u15287/insect_pest_detection/test_prediction/boxes"
pattern = re.compile(r'^(.*)_(\d+)\.jpg$')

output_folder = "/user/christoph.wald/u15287/insect_pest_detection/test_prediction/annotated_boxes"
os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    match = pattern.match(filename)
    
    file_base, box_index_str = match.groups()
    box_index = int(box_index_str)

    false_positives = json_dict[file_base]['false_positives']
    boxes = false_positives['boxes']
    classes = false_positives['classes']

    box = boxes[box_index]
    cls = classes[box_index]

    image = cv2.imread(os.path.join(input_folder, filename))
    
    box_int = [int(coord) for coord in box]
    class_names = {
    0: "FungusGnats",
    1: "LeafMinerFlies",
    2: "Thrips",
    3: "WhiteFlies"
    }

    class_name = class_names.get(cls, f"Unknown({cls})")

    text_lines = [f"{class_name}"]

    # Image dimensions
    img_h, img_w = image.shape[:2]

    # Create white canvas for text, same height as image, width ~300px
    canvas_w = 300
    canvas_h = 300
    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255  # white background


    pad_height = canvas_h - img_h
    image = cv2.copyMakeBorder(image, 0, pad_height, 0, 0, cv2.BORDER_CONSTANT, value=[255,255,255])

    # Put text lines on canvas
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    color = (0, 0, 0)  # black text
    thickness = 2
    line_height = 30
    y0 = 40

    for i, line in enumerate(text_lines):
        y = y0 + i * line_height
        cv2.putText(canvas, line, (10, y), font, font_scale, color, thickness)

    # Concatenate image and canvas horizontally
    combined = np.hstack((image, canvas))

    # Show the combined image (if you can)
    # cv2.imshow("Image with Info", combined)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Or save combined image
    output_path = os.path.join(output_folder, filename)
    cv2.imwrite(output_path, combined)
    print(f"Saved combined image to {output_path}")