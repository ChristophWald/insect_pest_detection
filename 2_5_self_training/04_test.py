import sys
sys.path.append("/user/christoph.wald/u15287/insect_pest_detection/modules")
from modules_testing import *

predict = False
model_number = "5"
predict_on_new = False
update_labels = False
#add_empty_tiles()
train_model = False
eval_run = True


###############
#make predictions on tiles
####################
#these actually do not predict on the tiles, but use the full images and slide over it 


if predict:
    predict_on_tiles(model_number = model_number, output_number = "on_all_images")


#for unsegmented images
if predict_on_new:
    create_labels(image_folder = "/user/christoph.wald/u15287/big-scratch/02_splitted_data/train_labeled/manual_SSL/images", model_number = model_number)

if predict or predict_on_new:
    #plot histograms
    output_path  = "/user/christoph.wald/u15287/insect_pest_detection/2_5_self_training/predictions"
    plot_histograms("/user/christoph.wald/u15287/insect_pest_detection/2_5_self_training/predictions", output_path)


####################
#add the labels to the training data / create new training data
####################

if update_labels:
    add_labels(
        pred_file = "predictions_on_all_images.json", #change if needed!
        thresholds = {"BRAIIM": 0.75, "LIRIBO": 0.75, "FRANOC": 0, "TRIAVA": 0.75},
        run_number = "075added_labels",
        correct_labels = True,
        threshold_steps = False,
        write = True,
        write_into_tiles = True,
        add_weights=True
    )

####################
#train
####################
if train_model:
    # Path to your YAML file
    base_path = "/user/christoph.wald/u15287/big-scratch/04_SSL_training_data/training_data"
    yaml_path = os.path.join(base_path, "data.yaml")

    check_data_yaml(yaml_path)
    delete_cache_files(os.path.join(base_path, "labels"))


    train(
        train_data_dir = base_path,
        model_number= "5",
        epochs = 5
    )


####################
#evaluate on test set
####################

if eval_run:
    conf_thresholds = [0.603]

    evaluate_on_test_set(conf_thresholds[0], "6",  save_images = True)

#tests the model on the image processed labels to compare to the other test
#evaluate_on_test_set_image_proc(conf_thresholds[0], "9")

#collect metrics (see collect_metrics.ipynb)
#plot training curves (see plot_prec_recall_for_all)
#save images from selected runs
