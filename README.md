functions in modules: converting yolo to bbox coordinates, draw bboxes into an image, extraxt bboxes from an image, get all filenames from all subfolders, plot_label distribution boxplots and histograms, plot split distribution and std/mean plot 

#scripts for visual inspection
- create_images_w_bboxes (for all subfolders),
- extract_bboxes (for a specific folder)
- plot_statistics (box plots and distribution of labels over time)

#scripts for checking the thrips labels:
- clustering_thrips (with ResNet50, PCA and tSNE), 
- sort_thrips_by_label_entropy

#dataset creation
- train_test_templates (splits the dataset into test set and 6 training sets)
- create_set (using the templates)
- create tiles
- train_val_split (splits a dataset into train/validation set ready to use for supervised training)