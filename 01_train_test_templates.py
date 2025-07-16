from modules import *
import pandas as pd
import numpy as np
import random
import json
import os

random.seed(43)

'''
creates filelists for the datasplitting (not in place)
'''

#for saving plots of label distribution in folder "split_plots"
plot_statistics = True

#function to get statistics for a set of images
def count_files_per_subdict_df(data_dict, dict_name):
    counts = {key: len(subdict) for key, subdict in data_dict.items()}
    return pd.DataFrame([counts], index=[dict_name])

def count_files(data_dict, dict_name, statistics=False):
    counts = {key: len(subdict) for key, subdict in data_dict.items()}
    df = pd.DataFrame(counts, index=[f'{dict_name}_count']).T
    df[f'{dict_name}_count'] = df[f'{dict_name}_count'].astype(int)

    if statistics:
        means = {}
        stds = {}
        for key, subdict in data_dict.items():
            values = list(subdict.values())
            if values:
                means[key] = round(np.mean(values), 2)
                stds[key] = round(np.std(values, ddof=1), 2) if len(values) > 1 else 0.0
            else:
                means[key] = np.nan
                stds[key] = np.nan

        df[f'{dict_name}_mean'] = pd.Series(means)
        df[f'{dict_name}_std'] = pd.Series(stds)

    return df

'''
create a dataframe
collect data for all images and all labels
'''
print("Loading files.")
image_path = "/user/christoph.wald/u15287/big-scratch/dataset/images/"
label_path = "/user/christoph.wald/u15287/big-scratch/dataset/labels/"
files_labeled = get_files_by_subfolder(label_path, count_lines=True)  # Names + line counts
files_images = get_files_by_subfolder(image_path)

df = count_files(files_images, "all")
df_new = count_files(files_labeled, "all labeled", statistics = True)
df = pd.concat([df, df_new], axis = 1)


'''
keep only 200 random sampled labels of the FungusGnats
'''
print("Reducing number of FungusGnats labels.")
folder_name = "FungusGnats"
files_labeled["FungusGnats"] = {
    k: files_labeled["FungusGnats"][k]
    for k in random.sample(list(files_labeled["FungusGnats"].keys()),200)
}
#remove labeled files (for FungusGnats only those, that have been kept) from image files
for folder_name in files_images:
    if folder_name in files_labeled:
        labeled_keys = set(files_labeled[folder_name].keys())
        files_images[folder_name] = {
            k: v for k, v in files_images[folder_name].items()
            if k not in labeled_keys
        }
'''
keep only 140 random sampled labels of the WhiteFlies
'''
print("Reducing number of WhiteFlies labels.")
folder_name = "WhiteFlies"
files_labeled["WhiteFlies"] = {
    k: files_labeled["WhiteFlies"][k]
    for k in random.sample(list(files_labeled["WhiteFlies"].keys()),140)
}
#remove labeled files (for WhiteFlies only those, that have been kept) from image files
for folder_name in files_images:
    if folder_name in files_labeled:
        labeled_keys = set(files_labeled[folder_name].keys())
        files_images[folder_name] = {
            k: v for k, v in files_images[folder_name].items()
            if k not in labeled_keys
        }

if plot_statistics:
    plot_label_distribution_boxplots(files_labeled, output_dir="split_plots", filename = "reduced_labels_box.png")
    plot_label_histograms(files_labeled, output_dir="split_plots", filename="reduced_labels_hist.png")

files_unlabeled = files_images
df_new = count_files(files_unlabeled, "unlabeled")
df_new2 = count_files(files_labeled, "labeled reduced", statistics = True)
df = pd.concat([df, df_new, df_new2], axis= 1)

''''
drawing a 20% (per class) test set
'''
print("Drawing test set.")
test_set = {}

for folder_name, subdict in files_labeled.items():
    keys = list(subdict.keys())
    sample_size = max(1, int(0.2 * len(keys)))
    sampled_keys = random.sample(keys, sample_size)

    # Create test split
    test_set[folder_name] = {k: subdict[k] for k in sampled_keys}

    # Remove those from original files_labeled
    for k in sampled_keys:
        del files_labeled[folder_name][k]

train_labeled = files_labeled
df_new = count_files(test_set, "test set", statistics = True)
df_new2 = count_files(train_labeled, "train/val", statistics = True)
df = pd.concat([df, df_new, df_new2], axis= 1)

if plot_statistics:
    plot_label_distribution_boxplots(test_set, output_dir="split_plots", filename = "testset_box.png")
    plot_label_histograms(test_set, output_dir="split_plots", filename="testset_hist.png")
    plot_label_distribution_boxplots(train_labeled, output_dir="split_plots", filename = "train-val_box.png")
    plot_label_histograms(train_labeled, output_dir="split_plots", filename="train-val_hist.png")

'''
create 3 sets of unlabeled data with 200 images each
'''
print("Create unlabeled training sets.")
train2_unlabeled, train3_unlabeled, train4_unlabeled = {}, {}, {}

for folder_name, subdict in files_unlabeled.items():
    keys = list(subdict.keys())
    
    if len(keys) < 200 * 3:
        raise ValueError(f"Not enough items in {folder_name} to create 3 sets of 200")

    random.shuffle(keys)

    keys1 = keys[:200]
    keys2 = keys[200:400]
    keys3 = keys[400:600]

    train2_unlabeled[folder_name] = {k: subdict[k] for k in keys1}
    train3_unlabeled[folder_name] = {k: subdict[k] for k in keys2}
    train4_unlabeled[folder_name] = {k: subdict[k] for k in keys3}

    # Remove selected keys from the original subdict
    for k in keys1 + keys2 + keys3:
        del subdict[k]
df_new = count_files(train2_unlabeled, "unlabeled training set 1")
df_new2 = count_files(train3_unlabeled, "unlabeled training set 2")
df_new3 = count_files(train4_unlabeled, "unlabeled training set 3")
df_new4 = count_files(files_images, "unlabeled remaining")
df = pd.concat([df, df_new, df_new2, df_new3, df_new4], axis = 1)

df.to_csv("split.csv")

if plot_statistics:
    final_set = df.copy()
    columns_to_drop = [
        "all labeled", "all", "unlabeled", 
        "labeled reduced", "remaining labels", 
        "unlabeled remaining"
    ]
    columns_to_drop = [col + "_count" for col in columns_to_drop]
    final_set = final_set.drop(columns=columns_to_drop, errors="ignore")
    mean_std_df = df[[col for col in df.columns if col.endswith('_mean') or col.endswith('_std')]]
    final_set = final_set.drop(columns=mean_std_df.columns)
    plot_split(final_set)
    plot_mean_std_grid(mean_std_df)

'''
save splitting info to json
'''
print("Save splitting info.")
datasets = [test_set, train_labeled, train2_unlabeled, train3_unlabeled, train4_unlabeled]
filenames = ["test_set", "train_labeled", "train2_unlabeled", "train3_unlabeled", "train4_unlabeled" ]

output_folder = "split_info"  # change this to your desired folder name
os.makedirs(output_folder, exist_ok=True)  # create the folder if it doesn't exist

for dataset, filename in zip(datasets, filenames):
    output_path = os.path.join(output_folder, filename + ".json")
    with open(output_path, "w") as f:
        json.dump(dataset, f, indent=4)