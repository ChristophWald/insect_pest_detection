import os
import shutil

'''
copies only files which are not manually selected (selection by putting them into folders)
for thrips subset cleaning
'''

#copy the manual sorted thrips

path = "/home/wald/Schreibtisch/10_BA_Arbeit/01_data_preparation/Thrips_labels_manually_sorted/delete_batches_with_generally_bad_labels"
low_quality = os.listdir(path)

duplicate = ["FRANOC_0093_with_boxes.jpg"]

path = "/home/wald/Schreibtisch/10_BA_Arbeit/01_data_preparation/Thrips_labels_manually_sorted/delete_false_negatives"
false_negatives = os.listdir(path)

path = "/home/wald/Schreibtisch/10_BA_Arbeit/01_data_preparation/Thrips_labels_manually_sorted/delete_false_positives"
false_positive = os.listdir(path)

path = "/home/wald/Schreibtisch/10_BA_Arbeit/01_data_preparation/Thrips_labels_manually_sorted/possibly_also_false_negatives"
more_false_negatives = os.listdir(path)

delete = [low_quality, duplicate, false_negatives, false_positive, more_false_negatives]

skip = []
for l in delete:
    for f in l:
        match = re.match(r'^([^_]+_\d+)', f)
        skip.append(match.group(1))

def copy_except_list(input_folder, output_folder, exclude_list):
    """
    Copies all files from input_folder to output_folder,
    excluding any files whose name (without extension) is in exclude_list.
    
    Parameters:
        input_folder (str): Source folder with files to copy
        output_folder (str): Destination folder
        exclude_list (list): List of filenames to exclude (no extensions)
    """
    os.makedirs(output_folder, exist_ok=True)
    
    for filename in os.listdir(input_folder):
        name, ext = os.path.splitext(filename)
        if name not in exclude_list:
            src_path = os.path.join(input_folder, filename)
            dst_path = os.path.join(output_folder, filename)
            if os.path.isfile(src_path):
                shutil.copy(src_path, dst_path)


input_folder = "/home/wald/Schreibtisch/10_BA_Arbeit/01_data_preparation/dataset/images/Thrips"
output_folder = "/home/wald/Schreibtisch/10_BA_Arbeit/01_data_preparation/dataset_Thrips_cleaned/images"

copy_except_list(input_folder, output_folder, skip)

input_folder = "/home/wald/Schreibtisch/10_BA_Arbeit/01_data_preparation/dataset/labels/Thrips"
output_folder = "/home/wald/Schreibtisch/10_BA_Arbeit/01_data_preparation/dataset_Thrips_cleaned/labels"

copy_except_list(input_folder, output_folder, skip)
