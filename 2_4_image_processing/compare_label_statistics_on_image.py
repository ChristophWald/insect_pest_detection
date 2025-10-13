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

def compute_hsv_stats_with_contrast(image, boxes):
    """
    Compute mean, std HSV for a list of boxes and contrast against full image V channel.
    
    Parameters:
        image (np.array): BGR image
        boxes (list): list of boxes in [x1, y1, x2, y2] format
    
    Returns:
        dict: {'mean': [H,S,V], 'std': [H,S,V], 'contrast': float}
    """
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    all_pixels = []

    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        patch = hsv_image[y1:y2, x1:x2]
        if patch.size > 0:
            all_pixels.append(patch.reshape(-1, 3))  # Flatten pixels
    
    if len(all_pixels) == 0:
        mean_hsv = np.array([0, 0, 0])
        std_hsv = np.array([0, 0, 0])
    else:
        pixels = np.vstack(all_pixels)
        mean_hsv = pixels.mean(axis=0)
        std_hsv = pixels.std(axis=0)

    # Contrast: mean V of full image minus mean V of boxes
    full_v_mean = hsv_image[:, :, 2].mean()
    contrast = full_v_mean - mean_hsv[2]

    return {'mean': mean_hsv, 'std': std_hsv, 'contrast': contrast}


#load image file
images_masked = "/user/christoph.wald/u15287/big-scratch/02_splitted_data/train_labeled/SSL/03_images_masked"
file_list = os.listdir(images_masked)


#image_file = random.choice(file_list)#
image_file = "BRAIIM_0268.jpg"
print(image_file)
images_folder = "/user/christoph.wald/u15287/big-scratch/02_splitted_data/train_labeled/SSL/04_images_cropped" 
labels_folder ="/user/christoph.wald/u15287/big-scratch/02_splitted_data/train_labeled/SSL/04_labels_cropped"
image = cv2.imread(os.path.join(images_folder, image_file))
image_masked = cv2.imread(os.path.join(images_masked, image_file))

#load ground truth
label_file = os.path.splitext(image_file)[0] + ".txt"
label_path = os.path.join(labels_folder, label_file)
with open(label_path, "r") as f:
    yolo_rectangles = [
        tuple(map(int, line.strip("()\n ").split(",")))
        for line in f
    ]
    
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
yolo_rectangles = [(x, y, x + w, y + h) for (x, y, w, h) in yolo_rectangles]



#convert ground truth
device = "cpu"
#device = "cuda"
species = ["BRAIIM", "LIRIBO","FRANOC", "TRIAVA"]  # adjust as needed
row_index = next((i for i, sp in enumerate(species) if image_file.startswith(sp)), len(species))
label_boxes = torch.tensor(yolo_rectangles, dtype= torch.float32).to(device)
label_classes = torch.tensor([row_index]*len(label_boxes), dtype=torch.long).to(device)

#convert image process results
pred_boxes = torch.tensor(rectangles, dtype=torch.float32).to(device)
pred_classes = torch.tensor([row_index]*len(pred_boxes), dtype=torch.long).to(device)
confs = torch.ones(len(pred_boxes)).to(device)


#compare to ground truth
tp, fp, fn = compare_labels_vectorized(pred_boxes, pred_classes, confs, label_boxes, label_classes,
                                            tile_size = 640, iou_threshold=0.5, containment_threshold=0.8, 
                                            convert_to_xyxy=False)

tp_stats_processing = compute_hsv_stats_with_contrast(image, tp[0])
fn_stats_processing = compute_hsv_stats_with_contrast(image, fn[0])

'''
#predict boxes
model = YOLO(f"/user/christoph.wald/u15287/insect_pest_detection/2_5_self_training/runs/detect/train4/weights/best.pt")
boxes, confs, class_ids = sliding_window_prediction(image, model)
            
if boxes.numel() > 0:
    boxes, confs, class_ids = nms(boxes, confs, class_ids, iou_threshold=0.4, device=model.device)
    boxes, confs, class_ids = filter_mostly_contained_boxes(boxes, confs, class_ids, threshold=0.5)
    #print(f"Predicted: {boxes.size(0)}")

#compare to ground truth
tp, fp, fn = compare_labels_vectorized(boxes, class_ids, confs, label_boxes, label_classes,
                                            tile_size = 640, iou_threshold=0.5, containment_threshold=0.8, 
                                            convert_to_xyxy=False)

#tp[0], fp[0], fn[0] are lists with boxes in x1,y1,x2, y2 format

tp_stats_predicting = compute_hsv_stats_with_contrast(image, tp[0])
fn_stats_predicting = compute_hsv_stats_with_contrast(image, fn[0])
'''
import matplotlib.pyplot as plt
import numpy as np
import os

# Group stats
stats_groups = {
    'TP Processing': tp_stats_processing,
    'FN Processing': fn_stats_processing,
    #'TP Prediction': tp_stats_predicting,
    #'FN Prediction': fn_stats_predicting
}

# Prepare data for plotting
labels = ['H', 'S', 'V', 'Contrast']
means = []
stds = []

for key in stats_groups:
    stat = stats_groups[key]
    mean_vals = list(stat['mean']) + [stat['contrast']]  # append contrast
    std_vals = list(stat['std']) + [0]  # no std for contrast
    means.append(mean_vals)
    stds.append(std_vals)

means = np.array(means)
stds = np.array(stds)

# Plot grouped bar chart
x = np.arange(len(labels))
width = 0.2

fig, ax = plt.subplots(figsize=(10,6))

for i in range(len(means)):
    ax.bar(x + i*width, means[i], width, yerr=stds[i], capsize=5, label=list(stats_groups.keys())[i])

ax.set_xticks(x + 1.5*width)
ax.set_xticklabels(labels)
ax.set_ylabel('Value')
ax.set_title(f'HSV Stats + Contrast for {image_file}')
ax.legend()
plt.tight_layout()

# Save figure
save_path = f"{os.path.splitext(image_file)[0]}_hsv_contrast.png"
plt.savefig(save_path)
plt.close()
print(f"Plot saved to: {save_path}")