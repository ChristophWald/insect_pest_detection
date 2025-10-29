from ultralytics import YOLO

'''
just the model.train with parameters
'''
'''



#training
#p2
model = YOLO('yolov8s.pt')
model.train(data='/user/christoph.wald/u15287/big-scratch/04_SSL_training_data/03_train_reduced_background/data.yaml', 
            epochs=200, 
            #patience = 10, 
            imgsz=640,
            
            scale=0.3, #instead of 0.5
            mosaic= 0.25, #instead of 1.0
            mixup=0.05, #instead of 0.0
            erasing=0.4, #default (increase when oberving false positives)
            auto_augment="randaugment", #default, maybe try augmix
            
            crop_fraction= 0.1, #(heavy cropping!) instead of 1.0
            multi_scale= True,
            fliplr = 0.3 #instead of 0.5
            
            
            )
'''
'''

#training
#p1
#model = YOLO('yolov8s.pt')
model = YOLO("/user/christoph.wald/u15287/insect_pest_detection/3_1_supervised_training_evaluation/runs/detect/train9/weights/last.pt")
model.train(data='/user/christoph.wald/u15287/big-scratch/02_splitted_data/train_labeled/tiles_improved/data.yaml', 
            epochs=200, 
            #patience = 20, 
            imgsz=640,
            save_period=5,
            scale=0.3, #instead of 0.5
            mosaic= 0.25, #instead of 1.0
            mixup=0.05, #instead of 0.0
            erasing=0.4, #default (increase when oberving false positives)
            auto_augment="randaugment", #default, maybe try augmix
            flipud = 0.5,
            degrees = 0.1

            
            
            
            )
'''

'''
#for resuming training
model = YOLO('/user/christoph.wald/u15287/insect_pest_detection/2_2_supervised _training/runs/detect/train3/weights/last.pt')
model.train(resume=True)  
'''


model = YOLO("/user/christoph.wald/u15287/insect_pest_detection/3_1_supervised_training_evaluation/runs/detect/train6/weights/best.pt")
#metrics = model.val(data="/user/christoph.wald/u15287/big-scratch/02_splitted_data/train_labeled/split/data.yaml", imgsz = 1024, conf = 0.5608052586391568)
metrics = model.val(data = "/user/christoph.wald/u15287/big-scratch/02_splitted_data/test_set/for_yolo_test/data.yaml", imgsz = 1024, conf = 0.5608052586391568, save_json = True)
