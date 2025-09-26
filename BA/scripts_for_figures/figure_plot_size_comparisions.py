#Load 3 images
# Load labels
#Cut out the boxes into an array

import matplotlib.pyplot as plt
from modules_segmentation import scale_rect
from modules import load_yolo_labels, save_cropped_boxes
import numpy as np
import cv2

fungus = cv2.imread("/user/christoph.wald/u15287/big-scratch/02_splitted_data/train_labeled/supervised/images/BRAIIM_0016.jpg")
leafminer = cv2.imread("/user/christoph.wald/u15287/big-scratch/02_splitted_data/train_labeled/supervised/images/LIRIBO_0641.jpg")
whitefly = cv2.imread("/user/christoph.wald/u15287/big-scratch/02_splitted_data/train_labeled/supervised/images/TRIAVA_0360.jpg")
images = [fungus, leafminer, whitefly]
label_paths = ["/user/christoph.wald/u15287/big-scratch/02_splitted_data/train_labeled/supervised/labels/BRAIIM_0016.txt",
               "/user/christoph.wald/u15287/big-scratch/02_splitted_data/train_labeled/supervised/labels/LIRIBO_0641.txt",
               "/user/christoph.wald/u15287/big-scratch/02_splitted_data/train_labeled/supervised/labels/TRIAVA_0360.txt"]

boxes = []
for label_path, image in zip(label_paths, images):
    boxes_data, _= load_yolo_labels(label_path, image.shape[1], image.shape[0])
    boxes.append(save_cropped_boxes(image, boxes_data))

print(len(boxes))



#9x9 grid with examples
areas = [2000, 10000, 1000, 10000, 100, 1000] #contour areas min max
blue = [41704, 22340,3168 ] #95th threshold box size found
red = [97980,28525, 7830] #max box size found


# Compute cell size slightly larger than red[0] side length
cell_size = np.sqrt(red[0]) * 1.1  # 10% larger

fig, ax = plt.subplots(figsize=(6, 6))

# Draw 3x3 grid
for i in range(4):
    ax.axvline(i * cell_size, color='black', linewidth=1)
    ax.axhline(i * cell_size, color='black', linewidth=1)

# Draw circles and rectangles
for row in range(3):
    # First cell in row
    area_idx = [0, 2, 4][row]
    area = areas[area_idx]
    radius = np.sqrt(area / np.pi)
    center_x = cell_size / 2
    center_y = cell_size / 2 + row * cell_size
    circle = plt.Circle((center_x, center_y), radius, color='green', alpha=0.5)
    ax.add_patch(circle)
    w, h = 2*radius, 2*radius
    x, y, w_s, h_s = scale_rect(center_x - radius, center_y - radius, w, h, 1.5)
    rect = plt.Rectangle((x, y), w_s, h_s, edgecolor='green', facecolor='none', linewidth=1)
    ax.add_patch(rect)

    # Last cell in row
    area_idx = [1, 3, 5][row]
    area = areas[area_idx]
    radius = np.sqrt(area / np.pi)
    center_x = 2.5 * cell_size
    center_y = cell_size / 2 + row * cell_size
    circle = plt.Circle((center_x, center_y), radius, color='green', alpha=0.5)
    ax.add_patch(circle)
    w, h = 2*radius, 2*radius
    x, y, w_s, h_s = scale_rect(center_x - radius, center_y - radius, w, h, 1.5)
    rect = plt.Rectangle((x, y), w_s, h_s, edgecolor='green', facecolor='none', linewidth=1)
    ax.add_patch(rect)

    # Last cell center
    center_x = 2.5 * cell_size
    center_y = cell_size / 2 + row * cell_size
    '''
    # Red rectangle
    area = red[row]  # one red per row
    side = np.sqrt(area)
    x = center_x - side / 2
    y = center_y - side / 2
    rect_red = plt.Rectangle((x, y), side, side, edgecolor='red', facecolor='none', linewidth=1)
    ax.add_patch(rect_red)

    # Blue rectangle
    area = blue[row]  # one blue per row
    side = np.sqrt(area)
    x = center_x - side / 2
    y = center_y - side / 2
    rect_blue = plt.Rectangle((x, y), side, side, edgecolor='blue', facecolor='none', linewidth=1)
    ax.add_patch(rect_blue)
    '''
    
# Set equal aspect
ax.set_aspect('equal')
ax.set_xlim(0, 3 * cell_size)
ax.set_ylim(0, 3 * cell_size)
ax.axis('off')
plt.gca().invert_yaxis()
plt.tight_layout()

# Place one cropped image (original size, centered) into the middle cell of each row
for row in range(3):
    if len(boxes[row]) > 0:
        img = boxes[row][2]  # first cropped box
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        h, w, _ = img_rgb.shape

        # Middle cell center
        center_x = 1.5 * cell_size
        center_y = cell_size / 2 + row * cell_size

        # Compute bottom-left corner so image is centered
        x0 = center_x - w / 2
        y0 = center_y - h / 2

        # Show image at original size
        ax.imshow(img_rgb, extent=[x0, x0+w, y0 , y0+h])



plt.savefig("size_comparisions.jpg")