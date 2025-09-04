# Project Structure

## 2_1_1 Dataset Description
- **create_extracted_boxes**: get the bounding boxes as single JPGs (as given by the labels)  
- **create_image_w_bboxes**: get JPGs with the bounding boxes drawn in (as given by the labels)  
- **figure1_four_YSTs**: creates a JPG with four YSTs  
- **figure2_some_bboxes**: creates a JPG with some bounding boxes for each pest class  

## 2_1_2 Preprocessing
- **check_duplicates**: checking for identical images with different filenames  
- **check_labels**: checking for labels that do not correspond with the pest class of the subset  
- **clean_thrips_after_manual_sorting**: copies files not excluded by manual sorting into new folder  
- **delete_labels**: delete specific lines in label files that were identified as false by inspecting single bounding boxes  
- **plot_label_statistics**: some basic statistics and statistic plots for the dataset  
- **sort_by_creation_time**: sort image and corresponding labels by creation time (JPG EXIF data)  

## 2_1_3 Splitting
- **01_train_test_templates**: creates JSON filelists for the splitting  
- **02_create_sets**: copies the files according to the split filelists  
- **03_train_val_splits**: splits the training set  

## 2_2 Supervised Training
- **01_crop_all_images**: cuts out the relevant part of the images and recalculates labels  
- **02_create_tiles**: create tiles from the images and recalculate labels  
- **03_check_training_data**: checking image/label pairs  
- **04_remove_background**: removes empty images to get a specified foreground/background ratio  
- **train.py**: simple training script  

## 2_3 Testing
- **01_test_full_images**: predicts on full images of the test set and calculates metrics, returns metrics (JSON), per-image results (JSON), images with color-coded bounding boxes, cut-out predicted boxes  
- **02_check_false_positives_single_files**: takes the boxes of `01_test_full_images` and adds the predicted class (maybe not needed anymore)  

## 2_4 Image Processing
- **01_copy_uncropped**: gets the uncropped image files of a set specified by `train_test_templates` in `2_2_2 splitting` (maybe not needed anymore)  

## 3_1 Supervised Training Evaluation
- **folder `false_positive_boxes`**: false positives of train7 and train10 with averaged confidence > 0.6  
- **folder `metrics_collected`**: metrics collected by `test_full_images` for all trainings/testings  
- **folder `runs`**: training results  
- **check_false_positives_filtered_by_scores**: creates 10x10 grids with false positives filtered by a confidence threshold, also creates distribution plots (change from notebook to script)  
- **eval_supervised**: creates overview tables from all training runs and some plots (reorder, change from notebook to script)  
- **plot_prec_recall**: plots losses and metrics for training runs  

## BA
- **latex data and pdf** of BA thesis  

## Modules
- converting YOLO to bbox coordinates  
- draw bboxes into an image  
- extract bboxes from an image  
- get all filenames from all subfolders  
- plot label distribution boxplots and histograms  
- plot split distribution and std/mean plot  
(incomplete list)