# Self-Training with Structured Data for Efficient Insect Pest Detection
The thesis explores an alternative to manually labeling images for object detection training. In the given data each image contains only one type of object and images are sorted into categories corresponding to object types. This allows the use of image processing for automatic labeling. Trained models are used to create pseudo-labels for further training. 

The dataset consists of images of insect pests collected by the Julius-Kühn-Institut Braunschweig.

## BA
Final thesis.

## data_preprocessing
Misc. scripts for preprocessing.

## data_splitting
Splits data into labeled and unlabeled training sets, as well as test and validation sets.

## image_processing
Scripts to create images with background structures masked out, to generate labels for foreground objects, and to create two different types of augmented training data.

## prepare_training_data
Scripts for creating training data as expected by YOLO for full images or tiles.

## training
Scripts to generate pseudo-labels and a wrapper to train a YOLOv8 model.

## evaluation
Per class evaluation and comparison of trained models.

## results
Weights and metrics for models.

## data visualizations
Misc. scripts for additional visualizations.


Required packages: numpy, torch, torchvision, matplotlib, opencv-python, pandas, ultralytics

