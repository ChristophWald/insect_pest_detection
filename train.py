from ultralytics import YOLO

model = YOLO('yolov8s.pt')
#model = YOLO('yolov8s.yaml') 
model.train(data='/user/christoph.wald/u15287/big-scratch/supervised_test/data.yaml', epochs=100, patience = 10, imgsz=640)



metrics = model.val(data='/user/christoph.wald/u15287/big-scratch/supervised_test/data.yaml', split='test')