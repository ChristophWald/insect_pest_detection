from modules_testing import *


'''
####################
#make predictions on tiles
####################

#for unsegmented images
#create_labels(image_folder = "/user/christoph.wald/u15287/big-scratch/02_splitted_data/train_labeled/SSL/images_not_segmented/images", model_number = "3")


predict_on_tiles(model_number = "9", output_number = "on_all_images")

####################
#find labels
####################

#plot histograms
output_path  = "/user/christoph.wald/u15287/insect_pest_detection/2_5_self_training/metrics"
plot_histograms("/user/christoph.wald/u15287/insect_pest_detection/2_5_self_training/predictions", output_path)

#log labels / corrections

add_labels(
    pred_file = "predictions_on_all_images.json",
    thresholds = {"BRAIIM": 0.4, "LIRIBO": 0.4, "FRANOC": 0, "TRIAVA": 0.4},
    run_number = "04",
    correct_labels = True,
    threshold_steps = True,
    write = False,
    write_into_tiles = False
)

#create fp boxes (see draw_fp_boxes)
'''
####################
#add the labels to the training data / create new training data
####################

add_labels(
    pred_file = "predictions_on_all_images.json",
    thresholds = {"BRAIIM": 0.75, "LIRIBO": 0.75, "FRANOC": 0, "TRIAVA": 0.75},
    run_number = "075all",
    correct_labels = True,
    threshold_steps = False,
    write = True,
    write_into_tiles = False
)

'''
####################
#train
####################


training_folders = ["training_data07all", "training_data08all"]
e = 10


train(
    train_data_dir = "/user/christoph.wald/u15287/big-scratch/04_SSL_training_data/training_data_new_more_tiles",
    model_number= None,
    epochs = e
)


####################
#evaluate on test set
####################

conf_thresholds = [0.427]

#evaluate_on_test_set(conf_thresholds[0], "8",  save_images = False)

#tests the model on the image processed labels to compare to the other test
evaluate_on_test_set_image_proc(conf_thresholds[0], "9")



#collect metrics (see collect_metrics.ipynb)

#plot training curves (see plot_prec_recall_for_all)

#save images from selected runs

####################
#write labels into tiles before the next iteratoin
####################
'''