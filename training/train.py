from ultralytics import YOLO

'''
just the model.train() with parameters
'''


#
#model = YOLO('yolov8s.pt')
model = YOLO("/user/christoph.wald/u15287/insect_pest_detection/training/runs/detect/train2/weights/best.pt")
model.train(data = "/user/christoph.wald/u15287/big-scratch/04_SSL_training_data/training_data_04/data.yaml", 
            epochs=20, 
            #patience = 10, 
            imgsz=1280,
            #save_period=1,
            scale=0.3, #instead of 0.5
            mosaic= 0.25, #instead of 1.0
            mixup=0.05, #instead of 0.0
            erasing=0.4, #default (increase when oberving false positives)
            auto_augment="randaugment", #default, maybe try augmix
            flipud = 0.5,

            
            
        
            )