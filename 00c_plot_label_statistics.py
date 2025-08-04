from modules import get_files_by_subfolder
from modules_visualization import plot_label_distribution_boxplots
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

#set paths
image_path = "/user/christoph.wald/u15287/big-scratch/dataset/images/"
label_path = "/user/christoph.wald/u15287/big-scratch/dataset/labels/"

#get file dicts
files_labeled = get_files_by_subfolder(label_path, count_lines=True)  # Names + line counts
files_images = get_files_by_subfolder(image_path)

#get statistics
data = []
keys = list(files_labeled.keys())
for key in keys:
    num_images = len(files_images.get(key, {}))
    num_labels = len(files_labeled.get(key, {}))
    num_instances = sum(files_labeled[key].values())
    mean_instances = num_instances/num_labels
    data.append({
        'class': key, 
        "image_files": num_images, 
        "label_files": num_labels, 
        "percentage labeled": round(num_labels/num_images*100, 2),
        "instances_labeled": num_instances,
        "mean_instances/image": round(mean_instances,2),
        "estimated_total_instances": round(mean_instances*num_images,2)
    })
df = pd.DataFrame(data)

print(df)
print(f"Total files: {df['image_files'].sum()}")
print(f"Labeled files: {df['label_files'].sum()}")
print(f"Instances labels: {df['instances_labeled'].sum()}")

#plot number of labels vs. time of creation
output_dir = "label_plots"
os.makedirs(output_dir, exist_ok=True)

# Plot each subfolder and save
for folder_name, file_data in files_labeled.items():
    filenames = sorted(file_data.keys())
    line_counts = [file_data[fname] for fname in filenames]

    plt.figure(figsize=(20, 4))
    plt.plot(line_counts, marker='o')
    plt.title(f'Instances labeled per image: {folder_name}')
    plt.xlabel('Files in order of creation')
    plt.ylabel('Number of instances')
    plt.tight_layout()
    plt.grid(True)

    # Save the figure
    plot_filename = f"{folder_name.replace(' ', '_')}_label_distribution.png"
    plot_path = os.path.join(output_dir, plot_filename)
    plt.savefig(plot_path)
    plt.close()  # Close the figure to free memory

    print(f"Saved plot to {plot_path}")

plot_label_distribution_boxplots(files_labeled, output_dir=output_dir)