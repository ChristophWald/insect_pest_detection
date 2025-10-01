from modules import draw_box
import cv2
import os
from collections import defaultdict
import json
import pandas as pd
import matplotlib.pyplot as plt
import glob
import re
from pathlib import Path




def make_image_with_boxes(
    image,
    tp_data,  # (boxes, classes)
    fp_data,  # (boxes, classes)
    fn_data,  # (boxes, classes)
    output_image_path=None,
    filename=None
):
    
    drawn = image.copy()

    tp_boxes, tp_classes, _ = tp_data
    fp_boxes, fp_classes, _ = fp_data
    fn_boxes, fn_classes = fn_data

    # Draw true positives in green
    for box, cls in zip(tp_boxes, tp_classes):
        draw_box(drawn, box, (0, 255, 0), f"TP: {cls}")

    # Draw false positives in blue
    for box, cls in zip(fp_boxes, fp_classes):
        draw_box(drawn, box, (255, 0, 0), f"FP: {cls}")

    # Draw false negatives in red
    for box, cls in zip(fn_boxes, fn_classes):
        draw_box(drawn, box, (0, 0, 255), f"FN: {cls}")

    if output_image_path:
        cv2.imwrite(os.path.join(output_image_path,os.path.splitext(filename)[0] + "_w_boxes.jpg"), drawn)



def compute_metrics(results):
    """
    Compute total and per-class metrics from a list of detection results.

    Args:
        results: list of [filename, tp, fp, fn] where each of tp, fp, fn is (boxes, class_ids)

    Returns:
        Dictionary with overall 'summary' and per-class 'per_class' metrics.
    """
    class_stats = defaultdict(lambda: {"TP": 0, "FP": 0, "FN": 0})

    # Aggregate counts
    for _, tp, fp, fn in results:
        for cls in tp[1]:
            class_stats[cls]["TP"] += 1
        for cls in fp[1]:
            class_stats[cls]["FP"] += 1
        for cls in fn[1]:
            class_stats[cls]["FN"] += 1

    # Compute per-class precision and recall
    for cls, stats in class_stats.items():
        tp = stats["TP"]
        fp = stats["FP"]
        fn = stats["FN"]
        stats["precision"] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        stats["recall"] = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # Compute totals
    total_tp = sum(stats["TP"] for stats in class_stats.values())
    total_fp = sum(stats["FP"] for stats in class_stats.values())
    total_fn = sum(stats["FN"] for stats in class_stats.values())

    total_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    total_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0

    return {
        "summary": {
            "TP": total_tp,
            "FP": total_fp,
            "FN": total_fn,
            "precision": total_precision,
            "recall": total_recall
        },
        "per_class": dict(class_stats)
    }

def save_results_to_json(output_path,results):
    """
    Save the list of detection results to a JSON file.

    Args:
        results: List of [filename, tp, fp, fn] entries.
        output_path: Path to the output JSON file.
    """
    formatted = []
    for filename, tp, fp, fn in results:
        entry = {
            "filename": filename,
            "true_positives": {
                "boxes": tp[0],
                "classes": tp[1],
            },
            "false_positives": {
                "boxes": fp[0],
                "classes": fp[1],
                "scores": fp[2],
            },
            "false_negatives": {
                "boxes": fn[0],
                "classes": fn[1],
            }
        }
        formatted.append(entry)

    with open(os.path.join(output_path, "results.json"), "w") as f:
        json.dump(formatted, f, indent=4)


def natural_sort_key(path):
    # Extract folder name (e.g. train, train1, train11)
    folder = os.path.basename(os.path.dirname(path))
    # Split into text + numbers
    parts = re.split(r'(\d+)', folder)
    # Convert digits to integers, leave text as lowercase
    key = [int(p) if p.isdigit() else p.lower() for p in parts]
    return key

def plot_prec_recall(output_path):
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
    save_path = os.path.join(output_path, 'training_curves.jpg')
    plt.savefig(save_path)
    plt.close()

    print(f"Combined results plot saved to {save_path}")


def plot_histograms(input_folder, output_folder):
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    output_img = output_folder / "all_histograms.jpg"
    output_txt = output_folder / "false_negatives.txt"

    class_map = {"BRAIIM": 0, "LIRIBO": 1, "TRIAVA": 2}

    def natural_key(path):
        return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", path.stem)]

    all_confidences = []   # (json_name, {species: {"TP": [...], "FP": [...]}})
    all_corrections = {}   # {json_name: {species: FN_count}}

    for json_file in sorted(input_folder.glob("*.json"), key=natural_key):
        with open(json_file, "r") as f:
            data = json.load(f)

        confidences = defaultdict(lambda: {"TP": [], "FP": []})
        corrections = defaultdict(int)

        for label_type in ["TP", "FP", "FN"]:
            if label_type not in data:
                continue
            for species, images in data[label_type].items():
                for image_name, entries in images.items():
                    for entry in entries:
                        if label_type in ["TP", "FP"]:
                            # Use second element of prediction as confidence
                            pred_list = entry.get("prediction", [])
                            if len(pred_list) > 1:
                                confidences[species][label_type].append(pred_list[1])
                        else:  # FN
                            corrections[species] += len(entries)

        all_confidences.append((json_file.stem, confidences))
        all_corrections[json_file.stem] = dict(corrections)

    # Save corrections
    with open(output_txt, "w") as f:
        for json_name, corr in all_corrections.items():
            f.write(f"{json_name}:\n")
            for species in class_map.keys():
                f.write(f"  {species}: {corr.get(species, 0)} FN\n")
            f.write("\n")

    print(f"Saved corrections report → {output_txt}")

    # Plot histograms
    n_files = len(all_confidences)
    n_species = len(class_map)
    fig, axes = plt.subplots(n_files, n_species, figsize=(5*n_species, 4*n_files), sharey=True)

    # Make axes always a 2D list
    if n_files == 1 and n_species == 1:
        axes = [[axes]]
    elif n_files == 1:
        axes = [list(axes)]
    elif n_species == 1:
        axes = [[ax] for ax in axes]
    else:
        axes = axes.tolist()

    for row_idx, (json_name, confs) in enumerate(all_confidences):
        for col_idx, species in enumerate(class_map.keys()):
            ax = axes[row_idx][col_idx]
            tp_vals = confs[species]["TP"]
            fp_vals = confs[species]["FP"]

            if tp_vals:
                ax.hist(tp_vals, bins=20, alpha=0.6, label="TP", color="green", edgecolor="black")
            if fp_vals:
                ax.hist(fp_vals, bins=20, alpha=0.6, label="FP", color="orange", edgecolor="black")

            ax.set_title(f"{json_name} - {species}")
            ax.set_xlabel("Confidence")
            ax.set_ylabel("Frequency")
            ax.legend()
            ax.grid(True)

    plt.tight_layout()
    plt.savefig(output_img, dpi=150)
    plt.close()
    print(f"Saved combined histogram grid → {output_img}")

