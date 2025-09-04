import os
import cv2
import numpy as np
from modules.modules import load_yolo_labels

'''

creates figure 2 - one row with nine boxes for every pest class

'''


def crop_boxes_from_image(image, boxes):
    """Return a list of cropped boxes (as np.ndarrays) from the image."""
    h, w = image.shape[:2]
    crops = []
    for box in boxes:
        xmin = max(0, int(np.floor(box[0])))
        ymin = max(0, int(np.floor(box[1])))
        xmax = min(w, int(np.ceil(box[2])))
        ymax = min(h, int(np.ceil(box[3])))
        cropped = image[ymin:ymax, xmin:xmax]
        if cropped.size > 0:
            crops.append(cropped)
    return crops


# create line grid
def pad_to_size(img, target_size, pad_color=(255,255,255)):
    """Pad an image to target_size"""
    target_w, target_h = target_size
    h, w = img.shape[:2]
    canvas = np.full((target_h, target_w, 3), pad_color, dtype=np.uint8)
    x_offset = (target_w - w) // 2
    y_offset = (target_h - h) // 2
    canvas[y_offset:y_offset+h, x_offset:x_offset+w] = img
    return canvas

def create_grid_with_lines(images_per_category, line_color=(0,0,0), line_thickness=2, pad_color=(255,255,255)):
    """
    Create a table-like grid with lines around cells.
    """
    num_cells = 9  # 9 crops per row

    # Compute max cell width across all crops
    max_cell_width = max(img.shape[1] for crops in images_per_category for img in crops)

    # Compute row heights dynamically
    row_heights = [max(img.shape[0] for img in crops) for crops in images_per_category]
    grid_h = sum(row_heights) + (len(images_per_category)+1)*line_thickness
    grid_w = max_cell_width*num_cells + (num_cells+1)*line_thickness

    grid_image = np.full((grid_h, grid_w, 3), pad_color, dtype=np.uint8)

    y_offset = line_thickness
    for row_idx, crops in enumerate(images_per_category):
        row_h = row_heights[row_idx]

        # Ensure exactly 9 crops per row
        if len(crops) < num_cells:
            crops = (crops * (num_cells // len(crops) + 1))[:num_cells]
        else:
            crops = crops[:num_cells]

        x_offset = line_thickness
        for col_idx, crop in enumerate(crops):
            cell = pad_to_size(crop, (max_cell_width, row_h), pad_color)
            grid_image[y_offset:y_offset+row_h, x_offset:x_offset+max_cell_width] = cell

            # Draw vertical line on the right of the cell
            x_line = x_offset + max_cell_width
            cv2.line(grid_image, (x_line, y_offset-line_thickness), (x_line, y_offset+row_h), line_color, line_thickness)

            x_offset += max_cell_width + line_thickness

        # Draw horizontal line below the row
        y_line = y_offset + row_h
        cv2.line(grid_image, (0, y_line), (grid_w, y_line), line_color, line_thickness)

        y_offset += row_h + line_thickness

    # Draw outer border lines
    cv2.rectangle(grid_image, (0,0), (grid_w-1, grid_h-1), line_color, line_thickness)

    return grid_image

image_paths = [
    "/user/christoph.wald/u15287/big-scratch/01_dataset/images/FungusGnats/BRAIIM_0008.jpg", 
    "/user/christoph.wald/u15287/big-scratch/01_dataset/images/LeafMinerFlies/LIRIBO_0007.jpg", 
    "/user/christoph.wald/u15287/big-scratch/01_dataset/images/Thrips/FRANOC_0021.jpg",  
    "/user/christoph.wald/u15287/big-scratch/01_dataset/images/WhiteFlies/TRIAVA_0008.jpg"]

label_paths = [
    "/user/christoph.wald/u15287/big-scratch/01_dataset/labels/FungusGnats/BRAIIM_0008.txt", 
    "/user/christoph.wald/u15287/big-scratch/01_dataset/labels/LeafMinerFlies/LIRIBO_0007.txt", 
    "/user/christoph.wald/u15287/big-scratch/01_dataset/labels/Thrips/FRANOC_0021.txt",  
    "/user/christoph.wald/u15287/big-scratch/01_dataset/labels/WhiteFlies/TRIAVA_0008.txt"]

# collect all crops
all_crops = []  # list of lists: one sublist per image
for image_path, label_path in zip(image_paths, label_paths):
    print(f"Processing {os.path.basename(image_path)}")
    image = cv2.imread(image_path)
    boxes, _ = load_yolo_labels(label_path, image.shape[1], image.shape[0])
    crops = crop_boxes_from_image(image, boxes)
    all_crops.append(crops)



# create figre
grid_with_lines = create_grid_with_lines(all_crops)
cv2.imwrite("/user/christoph.wald/u15287/insect_pest_detection/figure2.jpg", grid_with_lines)
