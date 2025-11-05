from ultralytics import YOLO

'''
just the model.train with parameters
'''



#training
#p1
#model = YOLO('yolov8s.pt')


#
#model = YOLO('yolov8s.pt')
model = YOLO("/user/christoph.wald/u15287/insect_pest_detection/2_3_supervised_training/runs/detect/train15/weights/best.pt")
model.train(data = "/user/christoph.wald/u15287/big-scratch/04_SSL_training_data/training_data_mininside08_added025_train15/data.yaml", 
            epochs=20, 
            #patience = 10, 
            imgsz=640,
            #save_period=1,
            scale=0.3, #instead of 0.5
            mosaic= 0.25, #instead of 1.0
            mixup=0.05, #instead of 0.0
            erasing=0.4, #default (increase when oberving false positives)
            auto_augment="randaugment", #default, maybe try augmix
            flipud = 0.5,

            
            
        
            )
'''



#for resuming training
model = YOLO('/user/christoph.wald/u15287/insect_pest_detection/2_3_supervised_training/runs/detect/train17/weights/last.pt')
model.train(resume=True)  

'''
'''

model = YOLO("/user/christoph.wald/u15287/insect_pest_detection/3_1_supervised_training_evaluation/runs/detect/train6/weights/best.pt")
#metrics = model.val(data="/user/christoph.wald/u15287/big-scratch/02_splitted_data/train_labeled/split/data.yaml", imgsz = 1024, conf = 0.5608052586391568)
metrics = model.val(data = "/user/christoph.wald/u15287/big-scratch/02_splitted_data/test_set/test_set_w_new_labels/for_yolo_test/data.yaml", imgsz = 1024, conf = 0.5608052586391568, save_json = True)
'''