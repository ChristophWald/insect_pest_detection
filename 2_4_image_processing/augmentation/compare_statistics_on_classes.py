import sys
sys.path.append("/user/christoph.wald/u15287/insect_pest_detection/modules")
import matplotlib.pyplot as plt
from modules_segmentation import *
from modules_prediction import *
import numpy as np
import os
import cv2
import torch

#segmentation or prediction
processing = "prediction"

device = "cuda"
# Folders
images_masked = "/user/christoph.wald/u15287/big-scratch/02_splitted_data/train_labeled/SSL/03_images_masked"
images_folder = "/user/christoph.wald/u15287/big-scratch/02_splitted_data/train_labeled/SSL/04_images_cropped"
labels_folder = "/user/christoph.wald/u15287/big-scratch/02_splitted_data/train_labeled/SSL/04_labels_cropped"

species = ["BRAIIM", "LIRIBO", "FRANOC", "TRIAVA"]

for pest in species:
    file_list = [f for f in os.listdir(images_masked) if f.startswith(pest)]



    tp_pixels_all = []
    fn_pixels_all = []
    contrast_tp_list = []
    contrast_fn_list = []

    for image_file in file_list:
        print(f"Processing {image_file}")
        image = cv2.imread(os.path.join(images_folder, image_file))
        image_masked = cv2.imread(os.path.join(images_masked, image_file))

        # Load ground truth
        label_file = os.path.splitext(image_file)[0] + ".txt"
        label_path = os.path.join(labels_folder, label_file)
        with open(label_path, "r") as f:
            yolo_rectangles = [
                tuple(map(int, line.strip("()\n ").split(",")))
                for line in f
            ]

        # Parameters by species
        if "TRIAVA" in image_file:
            min_area_contour = 100 
            max_area_contour = 2000
            scale = 1.5
            max_ratio = 2 
            upper_limit_rectangles = None
            value_threshold = 97
            binary_default = False
        elif "LIRIBO" in image_file: 
            min_area_contour = 1000 
            max_area_contour = 10000 
            scale = 1.5
            max_ratio = 1.76
            upper_limit_rectangles = 28530
            value_threshold = None
            binary_default = True
        elif "BRAIIM" in image_file:
            min_area_contour = 2000 
            max_area_contour = 10000
            scale = 1.5
            max_ratio = 1.73
            upper_limit_rectangles = 42970
            value_threshold = None
            binary_default = True

        # Get rectangles from segmentation
        if processing == "segmentation":
            rectangles, v = get_list_of_rectangles(
                image_masked, min_area_contour, max_area_contour, scale, max_ratio,
                upper_limit_rectangles, value_threshold, binary_default
            )
            rectangles = [(x, y, x + w, y + h) for (x, y, w, h) in rectangles]
            pred_boxes = torch.tensor(rectangles, dtype=torch.float32).to(device) if rectangles else torch.empty((0,4), dtype=torch.float32).to(device)
            pred_classes = torch.tensor([row_index]*len(pred_boxes), dtype=torch.long).to(device)
            confs = torch.ones(len(pred_boxes)).to(device)


        if processing == "prediction":
            #predict boxes
            model = YOLO(f"/user/christoph.wald/u15287/insect_pest_detection/2_5_self_training/runs/detect/train4/weights/best.pt")
            boxes, confs, class_ids = sliding_window_prediction(image, model)
                        
            if boxes.numel() > 0:
                boxes, confs, class_ids = nms(boxes, confs, class_ids, iou_threshold=0.4, device=model.device)
                boxes, confs, class_ids = filter_mostly_contained_boxes(boxes, confs, class_ids, threshold=0.5)
                pred_boxes = boxes
                pred_classes = class_ids
                confs = confs
            else:
                pred_boxes = torch.empty((0,4), dtype=torch.float32).to(device)
                pred_classes = torch.empty((0,), dtype=torch.long).to(device)
                confs = torch.empty((0,), dtype=torch.float32).to(device)

        # Convert to tensors
        yolo_rectangles = [(x, y, x + w, y + h) for (x, y, w, h) in yolo_rectangles]
        row_index = next((i for i, sp in enumerate(species) if image_file.startswith(sp)), len(species))
        label_boxes = torch.tensor(yolo_rectangles, dtype=torch.float32).to(device) if yolo_rectangles else torch.empty((0,4), dtype=torch.float32).to(device)
        label_classes = torch.tensor([row_index]*len(label_boxes), dtype=torch.long).to(device)

        

        # Compare to ground truth
        tp, fp, fn = compare_labels_vectorized(
            pred_boxes, pred_classes, confs, label_boxes, label_classes,
            tile_size=640, iou_threshold=0.5, containment_threshold=0.8, 
            convert_to_xyxy=False
        )

        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        full_v_mean = hsv_image[:, :, 2].mean()

        # Helper function to collect pixels
        def collect_pixels(boxes):
            pixels_list = []
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                patch = hsv_image[y1:y2, x1:x2]
                if patch.size > 0:
                    patch = patch.reshape(-1, 3)
                    if patch.shape[1] == 3:
                        pixels_list.append(patch)
            return pixels_list

        # TP pixels
        tp_patches = collect_pixels(tp[0]) if tp[0] else []
        if tp_patches:
            tp_pixels_all.extend(tp_patches)
            tp_mean_v = np.vstack(tp_patches).mean(axis=0)[2]
        else:
            tp_mean_v = 0
        contrast_tp_list.append(full_v_mean - tp_mean_v)

        # FN pixels
        fn_patches = collect_pixels(fn[0]) if fn[0] else []
        if fn_patches:
            fn_pixels_all.extend(fn_patches)
            fn_mean_v = np.vstack(fn_patches).mean(axis=0)[2]
        else:
            fn_mean_v = 0
        contrast_fn_list.append(full_v_mean - fn_mean_v)

    # Stack all pixels
    tp_pixels_all = np.vstack(tp_pixels_all) if tp_pixels_all else np.empty((0,3))
    fn_pixels_all = np.vstack(fn_pixels_all) if fn_pixels_all else np.empty((0,3))

    # Flatten channels for plotting
    tp_flat = [tp_pixels_all[:, i] if tp_pixels_all.size else np.array([]) for i in range(3)]
    fn_flat = [fn_pixels_all[:, i] if fn_pixels_all.size else np.array([]) for i in range(3)]

    # Add contrast
    tp_flat.append(np.array(contrast_tp_list) if contrast_tp_list else np.array([]))
    fn_flat.append(np.array(contrast_fn_list) if contrast_fn_list else np.array([]))

    # Plotting
    labels = ['H', 'S', 'V', 'Contrast']
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10,6))

    # TP boxes
    tp_positions = x - width/2
    bp1 = ax.boxplot(tp_flat, positions=tp_positions, widths=width, patch_artist=True,
                     boxprops=dict(facecolor='skyblue', color='blue'),
                     medianprops=dict(color='navy'))

    # FN boxes
    fn_positions = x + width/2
    bp2 = ax.boxplot(fn_flat, positions=fn_positions, widths=width, patch_artist=True,
                     boxprops=dict(facecolor='lightcoral', color='red'),
                     medianprops=dict(color='darkred'))

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Value')
    ax.set_title(f'HSV + Contrast Distributions for {pest}')
    ax.legend([bp1["boxes"][0], bp2["boxes"][0]], ['TP','FN'])

    plt.tight_layout()
    save_path = f"{pest}_hsv_contrast_boxplot.png"
    plt.savefig(save_path)
    plt.close()
    print(f"Box plot saved to: {save_path}")
