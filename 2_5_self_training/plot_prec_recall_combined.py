import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import re
def natural_sort_key(path):
    # Extract folder name (e.g. train, train1, train11)
    folder = os.path.basename(os.path.dirname(path))
    # Split into text + numbers
    parts = re.split(r'(\d+)', folder)
    # Convert digits to integers, leave text as lowercase
    key = [int(p) if p.isdigit() else p.lower() for p in parts]
    return key
# Base path
base_path = '/user/christoph.wald/u15287/insect_pest_detection/2_5_self_training/runs/detect/'

result_files = sorted(
    glob.glob(os.path.join(base_path, 'train*/results.csv')),
    key=natural_sort_key
)

# Track where runs start
run_boundaries = []

# Load and concatenate results
all_dfs = []
current_epoch_offset = 0

for file in result_files:
    df = pd.read_csv(file)
    df['epoch'] = df['epoch'] + current_epoch_offset  # shift epochs to be continuous
    all_dfs.append(df)
    current_epoch_offset = df['epoch'].iloc[-1] + 1  # update offset
    run_boundaries.append(current_epoch_offset)

# Combine into one DataFrame
df_all = pd.concat(all_dfs, ignore_index=True)

# Create subplots (2 rows, 1 column so each spans full width)
fig, axs = plt.subplots(2, 1, figsize=(12, 10))

# Top plot: Precision, Recall, mAP
axs[0].plot(df_all['epoch'], df_all['metrics/precision(B)'], label='Precision (B)', marker='o')
axs[0].plot(df_all['epoch'], df_all['metrics/recall(B)'], label='Recall (B)', marker='o')
axs[0].plot(df_all['epoch'], df_all['metrics/mAP50(B)'], label='mAP50 (B)', color='red', marker='o')

axs[0].axhline(y=0.9, color='blue', linestyle=':', label='Precision Threshold')
axs[0].axhline(y=0.8, color='orange', linestyle=':', label='Recall Threshold')

# Add vertical lines for run boundaries
for boundary in run_boundaries[:-1]:  # skip last since it's after last run
    axs[0].axvline(x=boundary, color='red', linestyle=':', alpha=0.7)

axs[0].set_xlabel('Epoch (cumulative)')
axs[0].set_ylabel('Score')
axs[0].set_title('Precision and Recall vs Epochs (All Runs)')
axs[0].legend()
axs[0].grid(True)

# Bottom plot: Training and Validation losses
axs[1].plot(df_all['epoch'], df_all['train/box_loss'], label='Train Box Loss', marker='o')
axs[1].plot(df_all['epoch'], df_all['train/cls_loss'], label='Train Cls Loss', marker='o')
axs[1].plot(df_all['epoch'], df_all['val/box_loss'], label='Val Box Loss', linestyle='--', marker='o')
axs[1].plot(df_all['epoch'], df_all['val/cls_loss'], label='Val Cls Loss', linestyle='--', marker='o')

# Add vertical lines for run boundaries
for boundary in run_boundaries[:-1]:
    axs[1].axvline(x=boundary, color='red', linestyle=':', alpha=0.7)

axs[1].set_xlabel('Epoch (cumulative)')
axs[1].set_ylabel('Loss')
axs[1].set_title('Training and Validation Losses vs Epochs (All Runs)')
axs[1].legend()
axs[1].grid(True)

# Save figure
plt.tight_layout()
save_path = "/user/christoph.wald/u15287/insect_pest_detection/2_5_self_training/metrics_plots"
os.makedirs(save_path, exist_ok=True)
save_path = os.path.join(save_path, 'combined_results.jpg')
plt.savefig(save_path)
plt.close()

print(f"Combined results plot saved to {save_path}")
