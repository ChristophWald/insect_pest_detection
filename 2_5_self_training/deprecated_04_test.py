import sys
sys.path.append("/user/christoph.wald/u15287/insect_pest_detection/modules")
from modules_testing import *

predict = False
model_number = "15"
predict_on_new = False
update_labels = True
#add_empty_tiles()
train_model = False
eval_run = False


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
        thresholds = {"BRAIIM": 0.6, "LIRIBO": 0.6, "FRANOC": 0, "TRIAVA": 0.6},
        run_number = "15_06",
        correct_labels = False,
        threshold_steps = True,
        write = False,
        write_into_tiles = False,
        add_weights=False
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
        model_number= None,
        epochs = 10
    )


####################
#evaluate on test set
####################

if eval_run:
    
    #train4 supervised
    class_conf_thresholds = {0: 0.6434290409088135, 
                             1: 0.4253721833229065, 
                             2: 0.5093783140182495, 
                             3: 0.5793536305427551}
    class_conf_thresholds = {0: 0.5729679465293884, 
                             1: 0.6294105052947998, 
                             2: 0.0, 
                             3: 0.3088245987892151}
    
    evaluate(0.2, "15",  save_images = False, per_class_confs = class_conf_thresholds)


    
#tests the model on the image processed labels to compare to the other test
#evaluate_on_test_set_image_proc(conf_thresholds[0], "9")

#collect metrics (see collect_metrics.ipynb)
#plot training curves (see plot_prec_recall_for_all)
#save images from selected runs
