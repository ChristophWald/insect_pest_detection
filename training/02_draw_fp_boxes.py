import os
import cv2
import numpy as np
import json

'''
plots grids with cut_out bounding boxes in a given confidence range
for visual inspection
'''

def visualize_fp_boxes_global_grid(json_file, image_dir, output_dir, conf_range=(0.0, 1.0), grid_dim=(10, 10)):
    """
    Extract FP boxes from JSON across all images and save as 10x10 grids of crops.
    Saves each grid immediately after filling it to avoid high memory usage.
    """
    os.makedirs(output_dir, exist_ok=True)

    with open(json_file, "r") as f:
        data = json.load(f)

    fp_data = data.get("FP", {})
    rows, cols = grid_dim
    grid_size = rows * cols
    batch_crops = []
    max_h, max_w = 0, 0
    grid_count = 0

    for species, images in fp_data.items():
        for img_name, entries in images.items():
            boxes_to_draw = [
                entry for entry in entries
                if len(entry.get("prediction", [])) >= 6 and
                   conf_range[0] <= entry["prediction"][-1] <= conf_range[1]
            ]
            if not boxes_to_draw:
                continue

            img_path = os.path.join(image_dir, img_name)
            if not os.path.exists(img_path):
                print(f"[WARN] Image not found: {img_name}")
                continue

            img = cv2.imread(img_path)
            if img is None:
                print(f"[ERROR] Failed to read {img_path}")
                continue

            for entry in boxes_to_draw:
                _, x1, y1, x2, y2, conf = entry["prediction"]
                crop = img[int(y1):int(y2), int(x1):int(x2)]
                if crop.size == 0:
                    continue
                batch_crops.append(crop)
                max_h = max(max_h, crop.shape[0])
                max_w = max(max_w, crop.shape[1])

                # When batch reaches grid size, save and reset
                if len(batch_crops) == grid_size:
                    grid_img = np.ones((rows * max_h, cols * max_w, 3), dtype=np.uint8) * 255
                    for idx, c in enumerate(batch_crops):
                        r = idx // cols
                        c_idx = idx % cols
                        h, w = c.shape[:2]
                        grid_img[r*max_h:r*max_h+h, c_idx*max_w:c_idx*max_w+w] = c

                    grid_count += 1
                    out_path = os.path.join(output_dir, f"fp_grid_{grid_count}.jpg")
                    cv2.imwrite(out_path, grid_img)
                    print(f"Saved grid: {out_path}")

                    # Reset batch
                    batch_crops = []
                    max_h, max_w = 0, 0

    # Save remaining crops if any
    if batch_crops:
        grid_img = np.ones((rows * max_h, cols * max_w, 3), dtype=np.uint8) * 255
        for idx, c in enumerate(batch_crops):
            r = idx // cols
            c_idx = idx % cols
            h, w = c.shape[:2]
            grid_img[r*max_h:r*max_h+h, c_idx*max_w:c_idx*max_w+w] = c

        grid_count += 1
        out_path = os.path.join(output_dir, f"fp_grid_{grid_count}.jpg")
        cv2.imwrite(out_path, grid_img)
        print(f"Saved final grid with {len(batch_crops)} crops: {out_path}")


json_file = "/user/christoph.wald/u15287/insect_pest_detection/training/predictions/predictions_fullimage_.json"
#image_dir = "/user/christoph.wald/u15287/big-scratch/02_splitted_data/train_labeled/SSL/04_images_cropped"
image_dir = "/user/christoph.wald/u15287/big-scratch/02_splitted_data/train_unlabeled/04_images_cropped"


#generate confidence ranges in 0.1 steps
step = 0.1
conf_ranges = [(round(i, 1), round(i + step, 1)) for i in np.arange(0.1, 1.0, step)]

for i, conf in enumerate(reversed(conf_ranges), start=1):
    output_dir = f"/user/christoph.wald/u15287/big-scratch/test_crops_grids_{str(i).zfill(2)}"
    visualize_fp_boxes_global_grid(json_file, image_dir, output_dir, conf_range=conf, grid_dim=(10, 10))
