import sys
sys.path.append("/user/christoph.wald/u15287/insect_pest_detection/modules")
from modules_segmentation import *
from modules_prediction import *
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import cv2
import numpy as np

def enlarge_box(box, image_shape, scale=0.2):
    """
    Enlarge a bounding box by a given percentage.

    Parameters:
        box (tuple): (x1, y1, x2, y2)
        image_shape (tuple): (height, width, channels) of the image
        scale (float): Fraction to enlarge the box (e.g., 0.2 = 20%)

    Returns:
        tuple: New enlarged box (x1_new, y1_new, x2_new, y2_new)
    """
    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1

    # Amount to expand
    dx = int(width * scale / 2)
    dy = int(height * scale / 2)

    # New coordinates
    x1_new = max(x1 - dx, 0)
    y1_new = max(y1 - dy, 0)
    x2_new = min(x2 + dx, image_shape[1] - 1)
    y2_new = min(y2 + dy, image_shape[0] - 1)

    return (x1_new, y1_new, x2_new, y2_new)


def plot_value_otsu_grid_with_segment(image, boxes, output_dir=".", filename="value_otsu_segment_grid.png"):
    """
    Create a single stacked PNG with one row per box:
    [ original | Value histogram + Otsu threshold | binary mask | segmented foreground ]

    Parameters:
        image (np.ndarray): Input BGR image.
        boxes (list): List of (x1, y1, x2, y2) tuples.
        output_dir (str): Folder to save the final stacked image.
        filename (str): Output filename.
    """
    if image is None or len(boxes) == 0:
        print("No image or boxes provided. Skipping plot.")
        return

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    os.makedirs(output_dir, exist_ok=True)

    num_boxes = len(boxes)
    fig, axes = plt.subplots(num_boxes, 4, figsize=(20, 3 * num_boxes))
    if num_boxes == 1:
        axes = np.array([axes])  # handle single box case

    for idx, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        region_bgr = image[y1:y2, x1:x2]
        region_hsv = hsv_image[y1:y2, x1:x2]

        if region_bgr.size == 0:
            continue

        value = region_hsv[:, :, 2]

        # --- Otsu threshold ---
        otsu_thresh, binary_mask = cv2.threshold(value, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # --- 1) Original region ---
        axes[idx, 0].imshow(cv2.cvtColor(region_bgr, cv2.COLOR_BGR2RGB))
        axes[idx, 0].set_title(f"Box {idx+1}: Original")
        axes[idx, 0].axis("off")

        # --- 2) Value histogram ---
        axes[idx, 1].hist(value.ravel(), bins=50, color='gray', alpha=0.7)
        axes[idx, 1].axvline(otsu_thresh, color='r', linestyle='--', label=f"Otsu = {otsu_thresh:.1f}")
        axes[idx, 1].set_title("Value Histogram + Otsu Threshold")
        axes[idx, 1].set_xlabel("Value")
        axes[idx, 1].set_ylabel("Pixel Count")
        axes[idx, 1].legend()

        # --- 3) Binary mask ---
        axes[idx, 2].imshow(binary_mask, cmap='gray')
        axes[idx, 2].set_title("Binary Mask (Otsu)")
        axes[idx, 2].axis("off")

        # --- 4) Segmented region ---


        binary_mask_inv = cv2.bitwise_not(binary_mask)  # foreground=255, background=0

        # Apply mask to the region_bgr
        foreground = cv2.bitwise_and(region_bgr, region_bgr, mask=binary_mask_inv)

        # Optional: set background to white
        background = np.full_like(region_bgr, 255)  # white background
        result = np.where(binary_mask_inv[:, :, np.newaxis] == 0, background, foreground)


        # Show
        axes[idx, 3].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        axes[idx, 3].set_title("Segmented Foreground")
        axes[idx, 3].axis("off")


    plt.tight_layout()
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Otsu segmentation grid saved to: {save_path}")



def plot_hist(image, boxes, output_dir=".", base_filename="histogram"):
    """
    Plot and save one histogram figure for all boxes.
    Each row corresponds to a box with 3 histograms (H, S, V).

    Parameters:
        image (np.ndarray): Input BGR image.
        boxes (list): List of (x1, y1, x2, y2) tuples.
        output_dir (str): Folder to save the output image.
        base_filename (str): Base output filename.
    """
    if image is None or len(boxes) == 0:
        print("No image or boxes provided. Skipping histogram plot.")
        return

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    channel_names = ['Hue', 'Saturation', 'Value']
    colors = ['r', 'g', 'b']

    os.makedirs(output_dir, exist_ok=True)

    n_boxes = len(boxes)
    fig, axes = plt.subplots(n_boxes, 3, figsize=(15, 4 * n_boxes))

    # Ensure axes is always 2D array for consistency
    if n_boxes == 1:
        axes = np.expand_dims(axes, axis=0)

    for idx, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        region = hsv_image[y1:y2, x1:x2]
        if region.size == 0:
            continue

        for i, (ch_name, color) in enumerate(zip(channel_names, colors)):
            ax = axes[idx, i]
            channel_data = region[:, :, i].flatten()
            ax.hist(channel_data, bins=50, color=color, alpha=0.7)
            ax.set_title(f'{ch_name} Histogram (Box {idx+1})')
            ax.set_xlabel(ch_name)
            ax.set_ylabel('Pixel Count')

    plt.tight_layout()
    save_path = os.path.join(output_dir, f"{base_filename}_all_boxes.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"All histograms saved to: {save_path}")



#load image file
images_masked = "/user/christoph.wald/u15287/big-scratch/02_splitted_data/train_labeled/SSL/03_images_masked"
file_list = os.listdir(images_masked)
images_folder = "/user/christoph.wald/u15287/big-scratch/02_splitted_data/train_labeled/SSL/04_images_cropped" 
labels_folder ="/user/christoph.wald/u15287/big-scratch/02_splitted_data/train_labeled/SSL/04_labels_cropped"

#image_file = random.choice(file_list)#
#image_file = "BRAIIM_0268.jpg"

for image_file in file_list[:5]:
    print(image_file)
    image = cv2.imread(os.path.join(images_folder, image_file))
    image_masked = cv2.imread(os.path.join(images_masked, image_file))

        
    #recangles
    if "TRIAVA" in image_file:
        min_area_contour = 100 
        max_area_contour = 2000
        scale = 1.5
        max_ratio = 2 
        upper_limit_rectangles = None
        value_threshold =97 # 5th percentile
        binary_default = False
    elif "LIRIBO" in image_file: 
        min_area_contour = 1000 
        max_area_contour = 10000 
        scale = 1.5
        max_ratio = 1.76 #95th percentile
        upper_limit_rectangles = 28530 #22340 #95th percentile
        value_threshold = None
        binary_default = True
    elif "BRAIIM" in image_file:
        min_area_contour = 2000 
        max_area_contour = 10000
        scale = 1.5
        max_ratio = 1.73 #1.75 #95the percentile
        upper_limit_rectangles = 42970 #41703 #95th percentil
        value_threshold = None
        binary_default = True

    rectangles, v = get_list_of_rectangles(image_masked, min_area_contour, max_area_contour, scale, max_ratio, upper_limit_rectangles, value_threshold, binary_default)

    if "TRIAVA" in image_file:
        rectangles = remove_smaller_overlaps(rectangles)

    #convert xywh -> xyxy
    rectangles = [(x, y, x + w, y + h) for (x, y, w, h) in rectangles]

    scale = 1
    rectangles = [enlarge_box(box, image.shape, scale) for box in rectangles]

    plot_value_otsu_grid_with_segment(image, rectangles, "box_histograms", filename = f"{image_file}")
#plot_hist(image, rectangles, "box_histograms")