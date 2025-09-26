import json
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path
import re

# ---------- SETTINGS ----------
folder = Path("/user/christoph.wald/u15287/insect_pest_detection/2_5_self_training/predictions")   # change this to your folder path
output_folder = Path("/user/christoph.wald/u15287/insect_pest_detection/2_5_self_training/metrics_plots")
output_img = output_folder / "all_histograms.jpg"
output_txt = output_folder / "corrections.txt"

# Mapping from filename prefix to true class
class_map = {"BRAIIM": 0, "LIRIBO": 1, "TRIAVA": 3}

# ---------- NATURAL SORT HELPER ----------
def natural_key(path):
    """Split string into text + number chunks for natural sorting"""
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', path.stem)]

# ---------- LOAD & PROCESS ALL ----------
all_confidences = []   # store (json_name, confidences dict)
all_corrections = {}   # store {json_name: {class_id: count}}

for json_file in sorted(folder.glob("*.json"), key=natural_key):
    with open(json_file, "r") as f:
        data = json.load(f)

    corrections = defaultdict(int)
    confidences = defaultdict(list)

    # Process each image entry in JSON
    for filename, preds in data.items():
        # Determine true class from filename prefix
        true_class = None
        for prefix, class_id in class_map.items():
            if filename.startswith(prefix):
                true_class = class_id
                break
        if true_class is None:
            continue

        # Collect predictions
        for entry in preds:
            pred_class = entry["prediction"][0]
            confidence = entry["prediction"][-1]

            if pred_class != true_class:
                corrections[true_class] += 1
                entry["prediction"][0] = true_class

            confidences[true_class].append(confidence)

    # Save results for later
    all_confidences.append((json_file.stem, confidences))
    all_corrections[json_file.stem] = dict(corrections)

# ---------- SAVE CORRECTIONS AS TXT ----------
with open(output_txt, "w") as f:
    for json_name, corr in all_corrections.items():
        f.write(f"{json_name}:\n")
        for prefix, class_id in class_map.items():
            count = corr.get(class_id, 0)
            f.write(f"  {prefix} (class {class_id}): {count} corrections\n")
        f.write("\n")

print(f"Saved corrections report → {output_txt}")

# ---------- PLOT ALL HISTOGRAMS ----------
n_files = len(all_confidences)
fig, axes = plt.subplots(n_files, 3, figsize=(15, 4 * n_files), sharey=True)

if n_files == 1:
    axes = [axes]  # ensure iterable

for row, (json_name, confs) in enumerate(all_confidences):
    for col, (prefix, class_id) in enumerate(class_map.items()):
        ax = axes[row][col] if n_files > 1 else axes[col]
        values = confs.get(class_id, [])
        if values:
            ax.hist(values, bins=20, alpha=0.7, color="steelblue", edgecolor="black")
        ax.set_title(f"{json_name} - {prefix}")
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Frequency")

plt.tight_layout()
plt.savefig(output_img, dpi=150)
plt.close()

print(f"Saved combined histogram grid → {output_img}")
