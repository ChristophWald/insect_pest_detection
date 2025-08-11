from ultralytics import YOLO
from ultralytics.data.utils import verify_image_label
import pandas as pd

'''
building bricks for training (supervised)
metrics are only done for tiles/not full images
'''

def save_class_metrics(model, metrics, filename):
    """
    Extracts per-class metrics from a YOLOv8 model evaluation and saves them to a CSV file.

    Args:
        model: YOLO model (used for class names).
        metrics: Result of model.val(), containing class_result(i) method.
        filename: Path to save the CSV file.
    """
    num_classes = len(model.names)
    rows = []

    for i in range(num_classes):
        class_metrics = metrics.class_result(i)  # [recall, precision, map50, map50-95]

        row = {
            'class_idx': i,
            'class_name': model.names[i],
            'precision': class_metrics[0],
            'recall': class_metrics[1],
            'map50': class_metrics[2],
            'map50-95': class_metrics[3]
        }

        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(filename, index=False)
    print(f"Saved class metrics to {filename}")





#training
#medium augmentation - train1 with the large dataset
model = YOLO('yolov8s.pt')
model.train(data='/user/christoph.wald/u15287/big-scratch/supervised_large/data.yaml', 
            epochs=100, 
            patience = 10, 
            imgsz=640,
            
            scale=0.3, #instead of 0.5
            mosaic= 0.25, #instead of 1.0
            mixup=0.05, #instead of 0.0
            erasing=0.4, #default (increase when oberving false positives)
            auto_augment="randaugment", #default, maybe try augmix
            
            
            
            )



#for resuming training
model = YOLO('runs/detect/train2/weights/last.pt')
model.train(resume=True)  

#for evaluation with test set - replace by checking full images
model = YOLO('/user/christoph.wald/u15287/insect_pest_detection/runs/detect/train2/weights/best.pt')
metrics = model.val(data='/user/christoph.wald/u15287/big-scratch/supervised_large/data.yaml', split='test', plots= True, save_json= True)
save_class_metrics(model, metrics, "train2.csv")

