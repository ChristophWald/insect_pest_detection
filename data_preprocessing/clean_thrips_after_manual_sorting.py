import os
import shutil
import re

'''
copies only files which are not manually selected
selection is done by putting them into folders
used for thrips subset cleaning
'''

def copy_except_list(input_folder, output_folder, exclude_list):
    """
    Copies all files from input_folder to output_folder,
    excluding any files whose name (without extension) is in exclude_list.
    
    """
    os.makedirs(output_folder, exist_ok=True)
    
    for filename in os.listdir(input_folder):
        name, ext = os.path.splitext(filename)
        if name not in exclude_list:
            src_path = os.path.join(input_folder, filename)
            dst_path = os.path.join(output_folder, filename)
            if os.path.isfile(src_path):
                shutil.copy(src_path, dst_path)


#set up list with files to delete (i.e. not to copy)

path = #path to images with low quality
low_quality = os.listdir(path)

duplicates = ["FRANOC_0093_with_boxes.jpg"]

path = #path to images with too much false negatives
false_negatives = os.listdir(path)

path = #path to images with too much false postives
false_positive = os.listdir(path)

path = #path to even more problematic images
more_false_negatives = os.listdir(path)

delete = [low_quality, duplicates, false_negatives, false_positive, more_false_negatives]

skip = []
for l in delete:
    for f in l:
        match = re.match(r'^([^_]+_\d+)', f)
        skip.append(match.group(1))

#copy images and labels
input_folder = #set parent folder
output_folder = #set output folder

copy_except_list(input_folder, output_folder, skip)
