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
import numpy as np
from modules import save_cropped_boxes



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


import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path
import json
import re

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path
import json
import re



def plot_histograms_dynamic_fn(input_folder, output_folder):
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    output_img = output_folder / "all_histograms_dynamic_fn.jpg"

    class_map = {"BRAIIM": 0, "LIRIBO": 1, "TRIAVA": 2}

    def natural_key(path):
        return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", path.stem)]

    all_confidences = []
    all_fns = {}

    # Load JSON files
    for json_file in sorted(input_folder.glob("*.json"), key=natural_key):
        with open(json_file, "r") as f:
            data = json.load(f)

        confidences = defaultdict(lambda: {"TP": [], "FP": []})
        fns = defaultdict(int)

        for label_type in ["TP", "FP", "FN"]:
            if label_type not in data:
                continue
            for species, images in data[label_type].items():
                for image_name, entries in images.items():
                    if label_type in ["TP", "FP"]:
                        for entry in entries:
                            pred_list = entry.get("prediction", [])
                            if len(pred_list) > 5:
                                conf = pred_list[-1]
                                confidences[species][label_type].append(conf)
                    else:  # FN at conf=0
                        fns[species] += len(entries)

        all_confidences.append((json_file.stem, confidences))
        all_fns[json_file.stem] = dict(fns)

    # Plot histograms
    n_files = len(all_confidences)
    n_species = len(class_map)
    fig, axes = plt.subplots(n_files, n_species, figsize=(5*n_species, 4*n_files), sharey=True)

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
            tp_vals = np.array(confs[species]["TP"])
            fp_vals = np.array(confs[species]["FP"])
            total_gt = len(tp_vals) + all_fns[json_name].get(species, 0)

            # Use bins derived from data
            all_vals = np.concatenate([tp_vals, fp_vals]) if tp_vals.size + fp_vals.size > 0 else np.array([0,1])
            bins = np.histogram_bin_edges(all_vals, bins=20)
            bin_centers = 0.5 * (bins[:-1] + bins[1:])

            # --- TP histogram ---
            if tp_vals.size > 0:
                ax.hist(tp_vals, bins=bins, alpha=0.6, label="TP", color="green", edgecolor="black")
                ax.axvline(np.mean(tp_vals), color="green", linestyle="--", linewidth=2, label=f"TP mean={np.mean(tp_vals):.2f}")

            # --- FP histogram ---
            if fp_vals.size > 0:
                ax.hist(fp_vals, bins=bins, alpha=0.6, label="FP", color="orange", edgecolor="black")
                ax.axvline(np.mean(fp_vals), color="orange", linestyle="--", linewidth=2, label=f"FP mean={np.mean(fp_vals):.2f}")

            # --- Compute FN per bin dynamically ---
            fn_per_bin = []
            for thresh in bins[:-1]:
                tp_above_thresh = tp_vals[tp_vals >= thresh]
                fn_in_bin = total_gt - len(tp_above_thresh)
                fn_per_bin.append(fn_in_bin)

            ax.plot(bin_centers, fn_per_bin, 'r-', linewidth=2, label="FN (dynamic)")

            ax.set_title(f"{json_name} - {species}")
            ax.set_xlabel("Confidence")
            ax.set_ylabel("Frequency")
            ax.grid(True)
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles, labels, loc="upper left", fontsize="small")

    plt.tight_layout()
    plt.savefig(output_img, dpi=150)
    plt.close()
    print(f"Saved combined histogram grid with dynamic FN → {output_img}")



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
    all_max_fp_per_image = {}  # {json_name: {image_name: {"species": ..., "conf": ..., "box": [...]}}}

    for json_file in sorted(input_folder.glob("*.json"), key=natural_key):
        with open(json_file, "r") as f:
            data = json.load(f)

        confidences = defaultdict(lambda: {"TP": [], "FP": []})
        corrections = defaultdict(int)
        max_fp_per_image = {}  # Track highest FP per image for this JSON

        #we already have the highest FP box per image
        #it can be saved (see below)
        #but coordinates are relative to the tile
        #either tile id has to be saved or coordinates relative to the image
        #this belongs into another function

        for label_type in ["TP", "FP", "FN"]:
            if label_type not in data:
                continue

            for species, images in data[label_type].items():
                for image_name, entries in images.items():

                    if label_type in ["TP", "FP"]:
                        for entry in entries:
                            pred_list = entry.get("prediction", [])
                            if len(pred_list) > 5:  # [cls, xmin, ymin, xmax, ymax, conf]
                                conf = pred_list[-1]   # confidence
                                box = pred_list[1:5]

                                # Append to species confidences for histogram
                                confidences[species][label_type].append(conf)

                                # Track highest FP per image
                                if label_type == "FP":
                                    if image_name not in max_fp_per_image or conf > max_fp_per_image[image_name]["conf"]:
                                        max_fp_per_image[image_name] = {
                                            "species": species,
                                            "conf": conf,
                                            "box": box
                                        }

                    else:  # FN
                        corrections[species] += len(entries)

        all_confidences.append((json_file.stem, confidences))
        all_corrections[json_file.stem] = dict(corrections)
        all_max_fp_per_image[json_file.stem] = max_fp_per_image
    
    '''
    for json_name, image_dict in all_max_fp_per_image.items():
        out_data = {}

        for image_name, info in image_dict.items():
            out_data[image_name] = {
                "species": info["species"],
                "box": info["box"],        # [xmin, ymin, xmax, ymax]
                "confidence": info["conf"]
            }

        out_path = output_folder / f"{json_name}_max_fp.json"
        with open(out_path, "w") as f:
            json.dump(out_data, f, indent=4)

        print(f"Saved max FP JSON for {json_name} → {out_path}")
    '''

    '''
    # Save corrections
    with open(output_txt, "w") as f:
        for json_name, corr in all_corrections.items():
            f.write(f"{json_name}:\n")
            for species in class_map.keys():
                f.write(f"  {species}: {corr.get(species, 0)} FN\n")
            f.write("\n")

    print(f"Saved corrections report → {output_txt}")
    '''
    

  
    '''
    #old code
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
                mean_tp = np.mean(tp_vals)
                ax.axvline(mean_tp, color="green", linestyle="--", linewidth=2, label=f"TP mean={mean_tp:.2f}")
            if fp_vals:
                ax.hist(fp_vals, bins=20, alpha=0.6, label="FP", color="orange", edgecolor="black")
                mean_fp = np.mean(fp_vals)
                ax.axvline(mean_fp, color="orange", linestyle="--", linewidth=2, label=f"FP mean={mean_fp:.2f}")
            fn_count = all_corrections[json_name].get(species, 0)
            if fn_count > 0:
                ax.axhline(fn_count, color="red", linestyle=":", linewidth=2, label=f"FN={fn_count}")



            ax.set_title(f"{json_name} - {species}")
            ax.set_xlabel("Confidence")
            ax.set_ylabel("Frequency")
            ax.legend()
            ax.grid(True)

    plt.tight_layout()
    plt.savefig(output_img, dpi=150)
    plt.close()
    print(f"Saved combined histogram grid → {output_img}")
    '''
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
            tp_vals = np.array(confs[species]["TP"])
            fp_vals = np.array(confs[species]["FP"])

            # --- Plot original histograms ---
            if tp_vals.size > 0:
                n_tp, bins_tp, _ = ax.hist(tp_vals, bins=20, alpha=0.6, label="TP", color="green", edgecolor="black")
                mean_tp = np.mean(tp_vals)
                ax.axvline(mean_tp, color="green", linestyle="--", linewidth=2, label=f"TP mean={mean_tp:.2f}")
            else:
                bins_tp = np.linspace(0,1,21)
            if fp_vals.size > 0:
                n_fp, bins_fp, _ = ax.hist(fp_vals, bins=20, alpha=0.6, label="FP", color="orange", edgecolor="black")
                mean_fp = np.mean(fp_vals)
                ax.axvline(mean_fp, color="orange", linestyle="--", linewidth=2, label=f"FP mean={mean_fp:.2f}")
            else:
                bins_fp = np.linspace(0,1,21)

            # --- Horizontal FN line ---
            fn_count = all_corrections[json_name].get(species, 0)
            if fn_count > 0:
                ax.axhline(fn_count, color="red", linestyle=":", linewidth=2, label=f"FN={fn_count}")

            # --- Compute mean confidence & precision per bin (aligned to histogram bins) ---
            bins = bins_tp  # use same bins for simplicity
            bin_centers = 0.5 * (bins[:-1] + bins[1:])
            mean_conf_per_bin = []
            precision_per_bin = []

            for i in range(len(bins)-1):
                tp_in_bin = tp_vals[(tp_vals >= bins[i]) & (tp_vals < bins[i+1])]
                fp_in_bin = fp_vals[(fp_vals >= bins[i]) & (fp_vals < bins[i+1])]
                all_in_bin = np.concatenate((tp_in_bin, fp_in_bin))

                mean_conf = np.mean(all_in_bin) if len(all_in_bin) > 0 else np.nan
                precision = len(tp_in_bin) / (len(tp_in_bin) + len(fp_in_bin)) if (len(tp_in_bin) + len(fp_in_bin)) > 0 else np.nan

                mean_conf_per_bin.append(mean_conf)
                precision_per_bin.append(precision)

            # --- Plot mean confidence & precision lines on secondary axis ---
            ax2 = ax.twinx()
            ax2.plot(bin_centers, mean_conf_per_bin, "b--", linewidth=2, label="Mean confidence")
            ax2.plot(bin_centers, precision_per_bin, "r-", linewidth=2, label="Precision")

            # --- Labels, title, grid ---
            ax.set_title(f"{json_name} - {species}")
            ax.set_xlabel("Confidence")
            ax.set_ylabel("Frequency")
            ax.grid(True)

            # --- Merge legends ---
            handles1, labels1 = ax.get_legend_handles_labels()
            handles2, labels2 = ax2.get_legend_handles_labels()
            ax2.legend(handles1 + handles2, labels1 + labels2, loc="upper left", fontsize="small")

    plt.tight_layout()
    plt.savefig(output_img, dpi=150)
    plt.close()
    print(f"Saved combined histogram grid with mean conf & precision → {output_img}")
