import os
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

#add titles to the plots
#histplot is odd because of the changing scales

def draw_box(img, box, color, label):
    """Draw a bounding box with label on the image."""
    xmin, ymin, xmax, ymax = map(int, box)
    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 2)
    cv2.putText(img, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, color, 2)
    

def load_yolo_labels(file_path, img_width, img_height):
    """Load YOLO label file and convert to absolute pixel coordinates."""
    boxes = []
    classes = []
    with open(file_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            cls = int(parts[0])
            x_center, y_center, w, h = map(float, parts[1:])
            xmin = (x_center - w / 2) * img_width
            xmax = (x_center + w / 2) * img_width
            ymin = (y_center - h / 2) * img_height
            ymax = (y_center + h / 2) * img_height
            boxes.append([xmin,ymin,xmax,ymax])
            classes.append(cls)
    return [boxes, classes]

def save_cropped_boxes(image, boxes, filename, output_dir):
    """
    Save cropped regions from the original image as separate JPG files.

    Args:
        image (np.ndarray): Original image (BGR format).
        boxes (List[List[int]]): List of bounding boxes [xmin, ymin, xmax, ymax].
        filename (str): Original image filename (used to name crops).
        output_dir (str): Directory to save cropped images.
    """
    base_name = os.path.splitext(filename)[0]
    h, w = image.shape[:2]
    
    for idx, box in enumerate(boxes):
        xmin = max(0, int(np.floor(box[0])))
        ymin = max(0, int(np.floor(box[1])))
        xmax = min(w, int(np.ceil(box[2])))
        ymax = min(h, int(np.ceil(box[3])))

        cropped = image[ymin:ymax, xmin:xmax]
        if len(cropped) == 0:
            print(xmin, ymin, xmax, ymax)
        crop_filename = f"{base_name}_{idx}.jpg"
        crop_path = os.path.join(output_dir, crop_filename)
        cv2.imwrite(crop_path, cropped)

####old functions

def get_files_by_subfolder(root_dir, count_lines=False):
    '''
    Function to get filenames (and optional line counts for labels) from subfolders
    number of lines in label files is the number of labels
    '''
    file_dict = {}
    for subdir, _, files in os.walk(root_dir):
        if subdir == root_dir:
            continue  # Skip root directory itself
        folder_name = os.path.basename(subdir)
        file_entries = {}
        for f in files:
            file_path = os.path.join(subdir, f)
            if os.path.isfile(file_path):
                name = os.path.splitext(f)[0]
                if count_lines:
                    with open(file_path, 'r', encoding='utf-8') as file:
                        lines = file.readlines()
                    file_entries[name] = len(lines)
                else:
                    file_entries[name] = None
        file_dict[folder_name] = file_entries
    return file_dict

def plot_label_distribution_boxplots(files_labeled, output_dir, filename="label_boxplots_grid.png", cols=4):
    """
    Creates a grid of boxplots showing the distribution of label counts per image for each subfolder.

    Parameters:
        files_labeled (dict): Nested dict with {folder_name: {filename: count}} structure.
        output_dir (str): Directory to save the output plot.
        filename (str): Output image filename.
        cols (int): Number of columns in the subplot grid.
    """
    num_plots = len(files_labeled)
    rows = math.ceil(num_plots / cols)
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 4, rows * 12))
    axs = axs.flatten()

    # Determine global y-axis max
    all_counts = [count for folder_data in files_labeled.values() for count in folder_data.values()]
    ymax = max(all_counts)

    for i, (folder_name, file_data) in enumerate(files_labeled.items()):
        line_counts = list(file_data.values())
        axs[i].boxplot(line_counts, patch_artist=True)
        axs[i].set_title(f"{folder_name} labels")
        axs[i].set_ylabel('Instances labeled')
        axs[i].set_ylim(0, ymax)
        axs[i].set_yticks(np.arange(0, ymax + 10, 10))
        axs[i].grid(True, axis='y')
        axs[i].set_xticks([])

    # Hide unused subplots
    for j in range(i + 1, len(axs)):
        fig.delaxes(axs[j])

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path)
    plt.close()

def plot_label_histograms(data_dict, output_dir, filename, cols=4, figsize_per_plot=(4, 4), bins='auto'):
    """
    Plot histograms of label distributions for each key in a dictionary.

    Parameters:
    - data_dict (dict): A dictionary where each key maps to another dict of labeled items.
    - cols (int): Number of columns in the subplot grid.
    - figsize_per_plot (tuple): Size of each subplot (width, height).
    - bins (int or str): Number of bins or binning strategy for histograms.

    Returns:
    - None (shows the plot).
    """
    folders = list(data_dict.keys())
    num_plots = len(folders)

    rows = math.ceil(num_plots / cols)
    fig, axs = plt.subplots(rows, cols, figsize=(cols * figsize_per_plot[0], rows * figsize_per_plot[1]))
    axs = axs.flatten()

    for i, folder_name in enumerate(folders):
        line_counts = list(data_dict[folder_name].values())
        axs[i].hist(line_counts, bins=bins, color='skyblue', edgecolor='black')
        axs[i].set_title(folder_name + " label distribution")
        axs[i].set_xlabel('Label count')
        axs[i].set_ylabel('Frequency')
        axs[i].grid(True, axis='y')

    # Hide unused axes
    for j in range(i + 1, len(axs)):
        fig.delaxes(axs[j])

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path)
    plt.close()

def plot_split(df, output_dir = "split_plots", filename = "splitting"):
    """
    Plots a horizontal stacked bar plot for each row of a transposed DataFrame.
    Each column (original data row) contributes a segment starting at multiples of 200.
    Includes dotted vertical lines every 200 units and a legend based on column names.
    """

    rows = df.index  # Now rows are the metrics (e.g. dataset1_count, etc.)
    col_labels = df.columns  # These represent your data keys (e.g., folder1, folder2)
    num_cols = len(col_labels)

    fig, ax = plt.subplots(figsize=(12, 6))

    left_offsets = np.arange(num_cols) * 600

    # Draw bars
    for i, row in enumerate(rows):
        values = df.loc[row].values
        for j, value in enumerate(values):
            start = left_offsets[j]
            color = f"C{j}"
            ax.barh(y=i, width=value, left=start, height=0.8, color=color, edgecolor="black")

            # Add value label
            if value > 10:
                ax.text(start + value / 2, i, str(value), va='center', ha='center', color='white', fontsize=9)
            else:
                ax.text(start + value + 3, i, str(value), va='center', ha='left', color='black', fontsize=9)

    # Set y-axis
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows)
    ax.set_xlabel('Value')
    ax.set_title('Dataset splitting')

    # Dotted vertical lines every 200 units
    max_x = left_offsets[-1] + 600
    for x in range(0, int(max_x + 200), 200):
        ax.axvline(x=x, color='gray', linestyle='dotted', linewidth=1)

    # Create custom legend for columns
    legend_patches = [
        mpatches.Patch(color=f"C{i}", label=str(col_labels[i]))
        for i in range(num_cols)
    ]
    ax.legend(handles=legend_patches, title="Data Columns", bbox_to_anchor=(1.01, 1), loc='upper left')

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path)
    plt.close()

def plot_mean_std_grid(mean_std_df, output_dir = "split_plots", filename = "labeled_sets", cols=2):
    """
    Creates a grid of subplots, one per subdict (row in mean_std_df),
    showing mean ± std across metrics.
    
    Parameters:
        mean_std_df: DataFrame with *_mean and *_std columns
        cols: number of columns in the subplot grid
    """

    # Extract base metric names
    metric_bases = sorted(set(col.replace('_mean', '') for col in mean_std_df.columns if col.endswith('_mean')))
    x = np.arange(len(metric_bases))

    n_subdicts = len(mean_std_df)
    rows = math.ceil(n_subdicts / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4), squeeze=False)
    axes = axes.flatten()

    for i, subdict in enumerate(mean_std_df.index):
        means = []
        stds = []
        for metric in metric_bases:
            means.append(mean_std_df.at[subdict, f"{metric}_mean"])
            stds.append(mean_std_df.at[subdict, f"{metric}_std"])

        ax = axes[i]
        ax.errorbar(x, means, yerr=stds, fmt='o-', capsize=4, color='C0')
        ax.set_title(f"{subdict}")
        ax.set_xticks(x)
        ax.set_xticklabels(metric_bases, rotation=45)
        ax.set_ylabel("Mean ± Std")
        ax.grid(True, linestyle='--', alpha=0.5)

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path)
    plt.close()

