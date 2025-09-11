import cv2
import os
from modules_segmentation import *
import pandas as pd

inspection = False
predicted_rectangles = []
results = []

overlaps = 0
value_problems = 0

#set paths
image_folder = "/user/christoph.wald/u15287/big-scratch/02_splitted_data/train_labeled/images_masked"
labels_folder = "/user/christoph.wald/u15287/big-scratch/02_splitted_data/train_labeled/labels_for_masked_images"
image_files = sorted(os.listdir(image_folder))
cropped_images_folder = "/user/christoph.wald/u15287/big-scratch/02_splitted_data/train_labeled/images"
test_folder = "/user/christoph.wald/u15287/insect_pest_detection/image_processing_in_progress/test_generated_fat"
os.makedirs(test_folder, exist_ok= True)

#changed to run only on one insect pest class
#image_files = [f for f in image_files if "TRIAVA" in f]

for i, image_file in enumerate(image_files):

    filename= image_file.split(".")[0]
    entry = [filename]
    #Load image file
    print(f"Loading {image_file}, {i}/{len(image_files)}")
    image = cv2.imread(os.path.join(image_folder, image_file))
    
    
    #if inspection: cv2.imwrite(os.path.join(test_folder, filename + "_04_binary_mask.jpg"), create_binary_mask(image)) 
    
    
    #handcrafted features for filtering bounding boxes
    if "TRIAVA" in image_file:
        min_area_contour = 100  
        max_area_contour = 1000
        scale = 1.5
        max_ratio = 2
        upper_limit_rectangles = None
        lower_limit_rectangles = None
        value_threshold = 133  #133: 5th percentile
    elif "LIRIBO" in image_file: 
        min_area_contour = 1000 
        max_area_contour = 10000 
        scale = 1.5
        max_ratio = 1.62 #90th percentile
        upper_limit_rectangles = 28626 #95th percentile
        lower_limit_rectangles = None
        value_threshold = None
    elif "BRAIIM" in image_file:
        min_area_contour = 2000 
        max_area_contour = 10000
        scale = 1.5
        max_ratio = 1.59 #90the percentile
        upper_limit_rectangles = 43210 #95th percentil
        lower_limit_rectangles = 7788 #5th percentile
        value_threshold = None

    #find bounding boxes, filtered by handcrafted features and ratio of w/h, scale them    
    rectangles, v = get_list_of_rectangles(image, min_area_contour, max_area_contour, scale, max_ratio, upper_limit_rectangles, lower_limit_rectangles, value_threshold)
    value_problems += v
    predicted_rectangles.append(rectangles)
    
    if "TRIAVA" in image_file:
        count = len(rectangles)
        rectangles = remove_smaller_overlaps(rectangles)
        overlaps += count - len(rectangles)

    #loading the rectangles given by the yolo labels
    label_file = os.path.splitext(image_file)[0] + ".txt"
    label_path = os.path.join(labels_folder, label_file)
    with open(label_path, "r") as f:
        yolo_rectangles = [
            tuple(map(int, line.strip("()\n ").split(",")))
            for line in f
        ]
       
    if inspection:
        image_labels = draw_bounding_boxes(image, rectangles)
        image_labels = draw_bounding_boxes(image_labels, yolo_rectangles, color = (0,255,0))
        cv2.imwrite(os.path.join(test_folder, filename + "_w_labels.jpg"), image_labels)
   
   
    rectangles = [(x, y, x + w, y + h) for (x, y, w, h) in rectangles]
    yolo_rectangles = [(x, y, x + w, y + h) for (x, y, w, h) in yolo_rectangles]
    stats, fp_boxes = evaluate_detections(rectangles, yolo_rectangles)
    fp_boxes = [(x1, y1, x2-x1, y2-y1) for (x1,y1, x2, y2) in fp_boxes]
    if len(fp_boxes) > 0:
        image_labels = draw_bounding_boxes(image, fp_boxes, color = (0,0,255) )
        cv2.imwrite(os.path.join(test_folder, filename + "_w_labels.jpg"), image_labels)
    entry.append(stats)
    #print(entry)
    results.append(entry)
    

with open(os.path.join(test_folder,"rectangles.txt"), "w") as f:
    for item in predicted_rectangles:
        f.write(str(item) + "\n")

print(f"Deleted {overlaps} overlaps.")
print(f"Detected {value_problems} value problems.")

#evaluate
rows = []
for name, metrics in results:
    row = {"image": name, **metrics}  # merge the dict with image name
    rows.append(row)

df = pd.DataFrame(rows)
df['prefix'] = df['image'].str[:6]  # first 6 chars like LIRIBO, BRAIIM, TRIAVA
#df.to_csv(os.path.join(test_folder,"results.csv"))
# Sum TP, FP, FN over each prefix
grouped = df.groupby('prefix')[['TP', 'FP', 'FN']].sum().reset_index()
grouped['precision'] = grouped['TP'] / (grouped['TP'] + grouped['FP'])
grouped['recall'] = grouped['TP'] / (grouped['TP'] + grouped['FN'])

TP_total = df['TP'].sum()
FP_total = df['FP'].sum()
FN_total = df['FN'].sum()

precision_overall = TP_total / (TP_total + FP_total)
recall_overall = TP_total / (TP_total + FN_total)

print(grouped)
print("Overall precision:", precision_overall)
print("Overall recall:", recall_overall)
grouped.to_csv("metrics.csv")
#changed to append
#grouped.to_csv("/user/christoph.wald/u15287/insect_pest_detection/image_processing_in_progress/test_generated_fat/metrics.csv", mode='a', header=False, index=False)
