import pandas as pd
import matplotlib.pyplot as plt
import os

base_dir = "/user/christoph.wald/u15287/insect_pest_detection/training/runs/detect"
out_dir = "/user/christoph.wald/u15287/insect_pest_detection/training/metrics"

# Count how many runs
num_runs = len(os.listdir(base_dir))


# Create a figure with rows = num_runs, cols = 2
fig, axs = plt.subplots(num_runs, 2, figsize=(14, 5*num_runs))
if num_runs == 1:
    axs = [axs]  # keep consistent structure when only 1 run
else:
    axs = axs.reshape(num_runs, 2)

for i in range(num_runs):
    if i == 0:
        file_path = f'{base_dir}/train/results.csv'
    else:
        file_path = f'{base_dir}/train{i+1}/results.csv'
    
    df = pd.read_csv(file_path)

    # --- Left subplot: Precision & Recall ---
    ax = axs[i][0]
    ax.plot(df['epoch'], df['metrics/precision(B)'], label='Precision (B)', marker='o')
    ax.plot(df['epoch'], df['metrics/recall(B)'], label='Recall (B)', marker='o')
    ax.plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP50 (B)', color='red', marker='o')
    ax.axhline(y=0.9, color='blue', linestyle=':', label='Precision Thresh')
    ax.axhline(y=0.8, color='orange', linestyle=':', label='Recall Thresh')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Score')
    ax.set_title(f'Run {i+1}: Precision/Recall vs Epoch')
    ax.legend()
    ax.grid(True)

    # --- Right subplot: Losses ---
    ax = axs[i][1]
    ax.plot(df['epoch'], df['train/box_loss'], label='Train Box Loss', marker='o')
    ax.plot(df['epoch'], df['train/cls_loss'], label='Train Cls Loss', marker='o')
    ax.plot(df['epoch'], df['val/box_loss'], label='Val Box Loss', linestyle='--', marker='o')
    ax.plot(df['epoch'], df['val/cls_loss'], label='Val Cls Loss', linestyle='--', marker='o')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(f'Run {i+1}: Training vs Validation Losses')
    ax.legend()
    ax.grid(True)

plt.tight_layout()
save_path = os.path.join(out_dir, "all_runs_rows.png")
plt.savefig(save_path, dpi=300)
plt.close()

print(f"Saved combined plot at {save_path}")
