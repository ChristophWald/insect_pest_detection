from ultralytics import YOLO

#training
#medium augmentation 
model = YOLO('yolov8s.pt')
model.train(data="/user/christoph.wald/u15287/big-scratch/04_SSL_training_data/training_data/data.yaml" ,
            epochs=30,
            #patience = 20, 
            imgsz=640,
            #close_mosaic=18,
            scale=0.3, #instead of 0.5
            mosaic= 0.25, #instead of 1.0
            mixup=0.05, #instead of 0.0
            erasing=0.4, #default (increase when oberving false positives)
            auto_augment="randaugment", #default, maybe try augmix
            )



'''

#for resuming training
model = YOLO('/user/christoph.wald/u15287/insect_pest_detection/2_5_self_training/runs/detect/train8/weights/last.pt')
model.train(resume=True)  
'''