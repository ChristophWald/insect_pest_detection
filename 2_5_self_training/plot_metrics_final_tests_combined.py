import os
import glob
import re
import json
import pandas as pd
import matplotlib.pyplot as plt

base_path = '/user/christoph.wald/u15287/insect_pest_detection/2_5_self_training/'

# Class name mapping
class_name_map = {
    '0': 'BRAIIM',
    '1': 'LIRIBO',
    '3': 'TRIVA'
}

# Natural sorting function
def natural_sort_key(path):
    folder = os.path.basename(os.path.dirname(path))
    m = re.match(r'train(\d*)_test_test_set', folder)
    if m:
        num = int(m.group(1)) if m.group(1) else 0
        return (num,)
    return (float('inf'),)

# Collect metrics.json files
metric_files = sorted(
    glob.glob(os.path.join(base_path, 'train*_test_test_set/metrics.json')),
    key=natural_sort_key
)

if not metric_files:
    raise FileNotFoundError(f"No metrics files found under {base_path}")

# Read all JSON metrics
results = {}
folders = [os.path.basename(os.path.dirname(f)) for f in metric_files]
for folder, file in zip(folders, metric_files):
    with open(file, 'r') as f:
        results[folder] = json.load(f)

# Build DataFrames for summary + per_class
dfs = {}
all_keys = ['summary'] + [f"per_class.{k}" for k in results[folders[0]]['per_class'].keys()]

for key in all_keys:
    rows = []
    for folder in folders:
        if key == 'summary':
            stats = results[folder]['summary']
        else:
            class_id = key.split('.')[-1]
            stats = results[folder]['per_class'].get(class_id, {})
        row = {'folder': folder, **stats}
        rows.append(row)
    dfs[key] = pd.DataFrame(rows)

# Combine all DataFrames into one for saving
combined_df = pd.concat(dfs.values(), keys=dfs.keys(), names=['metric_type', 'index'])


# Plotting
save_dir = os.path.join(base_path, 'metrics_plots')
os.makedirs(save_dir, exist_ok=True)

x_labels = []
x_values = []
for folder in folders:
    m = re.match(r'train(\d*)_test_test_set', folder)
    num = int(m.group(1)) if m and m.group(1) else 0
    x_labels.append(str(num))
    x_values.append(num)

for key, df in dfs.items():
    fig, ax1 = plt.subplots(figsize=(10,6))

    # Left y-axis (TP, FP, FN)
    ax1.set_xlabel('Folder')
    ax1.set_ylabel('Counts', color='tab:blue')
    for col in ['TP','FP','FN']:
        if col in df.columns:
            ax1.plot(range(len(x_values)), df[col], marker='o', label=col)
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.grid(True, which='both', axis='both', linestyle='--', alpha=0.5)

    # Right y-axis (precision, recall)
    ax2 = ax1.twinx()
    ax2.set_ylabel('Score', color='tab:orange')
    for col in ['precision','recall']:
        if col in df.columns:
            ax2.plot(range(len(x_values)), df[col], marker='x', linestyle='--', label=col)
    ax2.set_ylim(0,1.0)
    ax2.tick_params(axis='y', labelcolor='tab:orange')
    ax2.grid(False)

    ax1.set_xticks(range(len(x_values)))
    ax1.set_xticklabels(x_labels)

    # Title & filename
    if key.startswith('per_class'):
        class_id = key.split('.')[-1]
        class_name = class_name_map.get(class_id, f'Class {class_id}')
        title = f'Metrics for {class_name}'
        filename_key = class_name
    else:
        title = 'Metrics for Summary'
        filename_key = 'summary'

    plt.title(title)

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    plt.tight_layout()
    save_path = os.path.join(save_dir, f"{filename_key}.jpg")
    plt.savefig(save_path)
    plt.close()

print(f"Plots saved in {save_dir}")
combined_df.to_csv(os.path.join(save_dir, 'all_metrics_combined.csv'))
print(f"Combined metrics saved as all_metrics_combined.csv")
