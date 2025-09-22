from ultralytics import YOLO

model = YOLO("/user/christoph.wald/u15287/insect_pest_detection/2_5_self_training/runs/detect/train/weights/best.pt")

train_images = "/user/christoph.wald/u15287/big-scratch/02_splitted_data/train_labeled/tiles/train/images"
val_images = "/user/christoph.wald/u15287/big-scratch/02_splitted_data/train_labeled/tiles/val/images"

results = model.predict(source = train_images, save= True, imgsz=640, conf= 0.25)

print[results[0]]