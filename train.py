from ultralytics import YOLO
from ultralytics.data.utils import verify_image_label

'''
#training
#more augmentation - train3
model = YOLO('yolov8s.pt')
model.train(data='/user/christoph.wald/u15287/big-scratch/03_train_background_30/data.yaml', 
            epochs=100, 
            patience = 10, 
            imgsz=640,
            
            scale=0.3, #instead of 0.5
            mosaic= 0.25, #instead of 1.0
            mixup=0.05, #instead of 0.0
            erasing=0.4, #default (increase when oberving false positives)
            auto_augment="randaugment", #default, maybe try augmix
            
            crop_fraction= 0.1, #(heavy cropping!) instead of 1.0
            multi_scale= True,
            fliplr = 0.3 #instead fo 0.5
            
            
            )


'''


#training
#medium augmentation - train1 
model = YOLO('yolov8s.pt')
model.train(data='/user/christoph.wald/u15287/big-scratch/03_train_background_30/data.yaml', 
            epochs=100, 
            patience = 10, 
            imgsz=640,
            
            scale=0.3, #instead of 0.5
            mosaic= 0.25, #instead of 1.0
            mixup=0.05, #instead of 0.0
            erasing=0.4, #default (increase when oberving false positives)
            auto_augment="randaugment", #default, maybe try augmix
            
            
            
            )

'''

#for resuming training
model = YOLO('runs/detect/train6/weights/last.pt')
model.train(resume=True)  
'''