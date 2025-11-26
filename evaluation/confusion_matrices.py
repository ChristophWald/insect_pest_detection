import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import os

'''
saves two plotted confusion matrices next to each other,
both normalized, one for recall, one for precision
expects a results.json as given by predict on fixed thresholds
'''


def create_confusion_matrix(data, #results.json created by predict_on_fixed_thresholds
                            species, #species names
                            label_map=None):
    num_classes = len(species)
    cm = np.zeros((num_classes + 1, num_classes + 1), dtype=int)

    # Default label map
    if label_map is None:
        label_map = {cls: cls for cls in species}
    labels = [label_map.get(cls, cls) for cls in species] + [label_map.get("background", "background")]

    for entry in data:
        filename = entry["filename"]
        # Derive GT class from filename; fallback to last index if unknown
        row_index = next((i for i, sp in enumerate(species) if filename.startswith(sp)), num_classes)

        # True positives (correct predictions)
        for cls_id in entry["true_positives"]["classes"]:
            cm[row_index, cls_id] += 1

        # False negatives (missed GT objects)
        cm[row_index, num_classes] += len(entry["false_negatives"]["classes"])

        # False positives
        for cls_id in entry["false_positives"]["classes"]:
            if cls_id == row_index:
                # Background FP (wrong location for correct class)
                cm[num_classes, cls_id] += 1
            else:
                # Misclassification (wrong class)
                cm[row_index, cls_id] += 1

    return cm, labels

def normalize_confusion_matrix(cm, #confusion matrix
                               axis=1 #row or column normalization
                               ):
    
    cm = np.array(cm, dtype=float)
    if axis is None:
        total = cm.sum()
        return cm / total if total > 0 else cm
    else:
        sums = cm.sum(axis=axis, keepdims=True)
        sums[sums == 0] = 1  # avoid division by zero
        return cm / sums

def plot_confusion_matrix(cm, #confusion matrix 
                          labels, 
                          output_path=None, 
                          title="Confusion Matrix"):
   
    df = pd.DataFrame(cm, index=labels, columns=labels)

    plt.figure(figsize=(8, 6))
    plt.imshow(df, interpolation="nearest", cmap="Blues")
    plt.title(title, fontsize = 12)
    plt.colorbar()

    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45, ha="right", fontsize = 12)
    plt.yticks(tick_marks, labels, fontsize = 12)

    # annotate counts
    for i in range(len(labels)):
        for j in range(len(labels)):
            plt.text(
                j, i, df.iat[i, j],
                ha="center", va="center",
                color="white" if df.iat[i, j] > df.values.max()/2 else "black",
                fontsize = 12
            )

    plt.ylabel("True Species", fontsize = 12)
    plt.xlabel("Predicted Species", fontsize = 12)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path)
    #plt.show()

def give_confusion_infos(file, 
                         species, 
                         label_map=None, 
                         output_path=None, 
                         include_thrips=True):
    """
    Load results, create confusion matrix, normalize, and plot with optional label mapping.
    Can optionally skip the 'Thrips' class.
    """
    with open(file, "r") as f:
        data = json.load(f)

    # create confusion matrix
    cm, labels = create_confusion_matrix(data, species, label_map=label_map)

    # Optionally remove 'Thrips'
    if not include_thrips:
        if label_map and "FRANOC" in label_map:
            thrips_label = label_map["FRANOC"]
        else:
            thrips_label = "FRANOC"  # fallback
        if thrips_label in labels:
            idx = labels.index(thrips_label)
            cm = np.delete(cm, idx, axis=0)  # remove row
            cm = np.delete(cm, idx, axis=1)  # remove column
            labels = [l for i,l in enumerate(labels) if i != idx]

    print("Confusion Matrix with FP/FN (last row = FP, last col = FN):\n", cm)

    # Normalize
    norm_cm_row = normalize_confusion_matrix(cm, axis=1)
    norm_cm_col = normalize_confusion_matrix(cm, axis=0)

    # Plot side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    for ax, norm_cm, title in zip(
        axes,
        [norm_cm_row, norm_cm_col],
        ["Recall for self-training on validation set", "Precision for for self-training on validation set"]
    ):
        df = pd.DataFrame(np.round(norm_cm, 2), index=labels, columns=labels)
        im = ax.imshow(df, interpolation="nearest", cmap="Blues")
        ax.set_title(title)
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize = 12)
        ax.set_yticklabels(labels, fontsize = 12)
        ax.set_ylabel("True Species", fontsize = 12)
        ax.set_xlabel("Predicted Species", fontsize = 12)

        # annotate cells
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(
                    j, i, f"{df.iat[i, j]:.2f}",
                    ha="center", va="center",
                    color="white" if df.iat[i, j] > df.values.max()/2 else "black",
                    fontsize=12
                )

    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, label="Normalized Value")
    if output_path:
        plt.savefig(output_path)
    plt.show()

    return norm_cm_row, norm_cm_col

species = ["BRAIIM", "LIRIBO", "FRANOC", "TRIAVA"]
label_map = {
    "BRAIIM": "Fungus gnats",
    "LIRIBO": "Leaf miner flies",
    "FRANOC": "Thrips",
    "TRIAVA": "Whiteflies",
    "background": "Background"
}


file = ".../results.json"
output_path = ".../confusions.png"
_,_ = give_confusion_infos(file, species,  label_map, output_path)