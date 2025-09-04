from insect_pest_detection.modules.modules import *
from insect_pest_detection.plotting.modules_plotting import *
import pandas as pd
import numpy as np
import random
import json
import os

random.seed(43)

'''
creates json filelists for the datasplitting (not in place)
'''

#for saving plots of label distribution in folder "split_plots"
plot_statistics = True

def count_files(data_dict, dict_name, statistics=False):
    '''
    creates a dataframe with basic statistics for each pest class
    '''
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

def reduce_class_labels(class_name, all_labeled, keep_n=200):
    """
    subtracts a number of randomly selected labels
    keeps only keep_n label files
    """
    print(f"Reducing number of '{class_name}' labels to {keep_n}...")

    if class_name not in all_labeled:
        raise ValueError(f"Class '{class_name}' not found in all_labeled.")

    original_keys = list(all_labeled[class_name].keys())
    if keep_n >= len(original_keys):
        print(f"Requested keep_n={keep_n} is greater than available samples ({len(original_keys)}); no reduction performed.")
        return all_labeled

    kept_keys = set(random.sample(original_keys, keep_n))

    # Reduce labels
    all_labeled[class_name] = {
        k: all_labeled[class_name][k] for k in kept_keys
    }

    return all_labeled


def remove_labeled_from_images(all_labeled, all_images):
    """
    returns a dict with all images not labeled
    """
    for class_name in all_labeled:
        if class_name in all_images:
            labeled_keys = set(all_labeled[class_name].keys())
            all_images[class_name] = {
                k: v for k, v in all_images[class_name].items()
                if k not in labeled_keys
            }
    return all_images

def custom_split_unlabeled(unlabeled_set):
    """
    returns two dicts of unlabelled files split by given rules
    """
    train_unlabeled = {}
    
    for class_name, items in unlabeled_set.items():
        keys = list(items.keys())
        random.shuffle(keys)

        if class_name == "Thrips":
            train_unlabeled[class_name] = {}
            unlabeled_set[class_name] = {k: items[k] for k in keys}

        elif class_name == "FungusGnats":
            n = 160
            train_unlabeled[class_name] = {k: items[k] for k in keys[:n]}
            unlabeled_set[class_name] = {k: items[k] for k in keys[n:]}

        elif class_name == "WhiteFlies":
            n = 160
            train_unlabeled[class_name] = {k: items[k] for k in keys[:n]}
            unlabeled_set[class_name] = {k: items[k] for k in keys[n:]}

        elif class_name == "LeafMinerFlies":
            n = 550
            train_unlabeled[class_name] = {k: items[k] for k in keys[:n]} 
            unlabeled_set[class_name] = {k: items[k] for k in keys[n:]} 

    return train_unlabeled, unlabeled_set

'''
create a dataframe
collect data for all images and all labels
'''
print("Loading files.")
image_path = "/user/christoph.wald/u15287/big-scratch/01_dataset/images/"
label_path = "/user/christoph.wald/u15287/big-scratch/01_dataset/labels/"
all_labeled = get_files_by_subfolder(label_path, count_lines=True)  # Names + line counts
all_images = get_files_by_subfolder(image_path)

df = count_files(all_images, "all_images")
df_new = count_files(all_labeled, "all labeled", statistics = True)
df = pd.concat([df, df_new], axis = 1)

'''
keep only 200 random sampled labels of the FungusGnats
'''
print("Reducing number of FungusGnats labels.")
all_labeled = reduce_class_labels("FungusGnats", all_labeled, 200)
'''
keep only 200 random sampled labels of the WhiteFlies
'''
all_labeled = reduce_class_labels("WhiteFlies", all_labeled, 200)

labeled_set = all_labeled

if plot_statistics:
    plot_label_distribution_boxplots(labeled_set, output_dir="split_plots", filename = "reduced_labels_box.png")
    plot_label_histograms(labeled_set, output_dir="split_plots", filename="reduced_labels_hist.png")

unlabeled_set = remove_labeled_from_images(labeled_set, all_images)

df_new = count_files(unlabeled_set, "unlabeled")
df_new2 = count_files(labeled_set, "labeled set", statistics = True)
df = pd.concat([df, df_new, df_new2], axis= 1)

''''
drawing a 20% (per class) test set
'''
print("Drawing test set.")
test_set = {}

for folder_name, subdict in labeled_set.items():
    keys = list(subdict.keys())
    sample_size = max(1, int(0.2 * len(keys)))
    sampled_keys = random.sample(keys, sample_size)

    # Create test split
    test_set[folder_name] = {k: subdict[k] for k in sampled_keys}

    # Remove those from original labeled_set
    for k in sampled_keys:
        del labeled_set[folder_name][k]

train_labeled = labeled_set
df_new = count_files(test_set, "test set", statistics = True)
df_new2 = count_files(train_labeled, "train/val", statistics = True)
df = pd.concat([df, df_new, df_new2], axis= 1)

if plot_statistics:
    plot_label_distribution_boxplots(test_set, output_dir="split_plots", filename = "testset_box.png")
    plot_label_histograms(test_set, output_dir="split_plots", filename="testset_hist.png")
    plot_label_distribution_boxplots(train_labeled, output_dir="split_plots", filename = "train-val_box.png")
    plot_label_histograms(train_labeled, output_dir="split_plots", filename="train-val_hist.png")

'''
create unlabeled training set
'''

train_unlabeled, unlabeled_set = custom_split_unlabeled(unlabeled_set)

# Optional: get counts
df1 = count_files(train_unlabeled, "train_unlabeled")
df2 = count_files(unlabeled_set, "remaining_unlabeled")

df = pd.concat([df, df1, df2], axis=1)


df.to_csv("split.csv")

if plot_statistics:
    final_set = df.copy()
    columns_to_drop = [
        "all_images", "all labeled", "unlabeled", 
        "labeled set", "remaining_unlabeled"
    ]
    columns_to_drop = [col + "_count" for col in columns_to_drop]
    final_set = final_set.drop(columns=columns_to_drop, errors="ignore")
    mean_std_df = df[[col for col in df.columns if col.endswith('_mean') or col.endswith('_std')]]
    final_set = final_set.drop(columns=mean_std_df.columns)
    final_set.rename(columns={"test set_count": "test set", 
                              "train/val_count": "labeled test set", 
                              "train_unlabeled_count": "unlabeled test set"})
    plot_split(final_set)
    mean_std_df.rename(columns = {"train/val": "training set"})
    plot_mean_std_grid(mean_std_df)

'''
save splitting info to json
'''
print("Save splitting info.")
datasets = [test_set, train_labeled, train_unlabeled]
filenames = ["test_set", "train_labeled", "train_unlabeled"]

output_folder = "split_info"  # change this to your desired folder name
os.makedirs(output_folder, exist_ok=True)  # create the folder if it doesn't exist

for dataset, filename in zip(datasets, filenames):
    output_path = os.path.join(output_folder, filename + ".json")
    with open(output_path, "w") as f:
        json.dump(dataset, f, indent=4)