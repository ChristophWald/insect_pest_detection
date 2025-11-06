import sys
sys.path.append("/user/christoph.wald/u15287/insect_pest_detection/modules")
from modules_testing import *


model_number = "15"


#these actually do not predict on the tiles, but use the full images and slide over it 



#predict_on_tiles(model_number = model_number, output_number = "on_all_images")


#for unsegmented images
#create_labels(image_folder = "/user/christoph.wald/u15287/big-scratch/02_splitted_data/train_labeled/manual_SSL/images", model_number = model_number)

#plot histograms
output_path  = "/user/christoph.wald/u15287/insect_pest_detection/2_5_self_training/predictions"
#
plot_histograms_dynamic_fn("/user/christoph.wald/u15287/insect_pest_detection/2_5_self_training/predictions", output_path)
plot_histograms("/user/christoph.wald/u15287/insect_pest_detection/2_5_self_training/predictions", output_path)