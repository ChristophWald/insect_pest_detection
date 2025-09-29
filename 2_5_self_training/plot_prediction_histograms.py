from collections import defaultdict
import json
from pathlib import Path
import matplotlib.pyplot as plt
import re


def plot_histograms():
    folder = Path("/user/christoph.wald/u15287/insect_pest_detection/2_5_self_training/predictions")
    output_folder = Path("/user/christoph.wald/u15287/insect_pest_detection/2_5_self_training/metrics_plots")
    output_folder.mkdir(exist_ok=True)
    output_img = output_folder / "all_histograms.jpg"
    output_txt = output_folder / "corrections.txt"

    class_map = {"BRAIIM": 0, "LIRIBO": 1, "TRIAVA": 2}

    def natural_key(path):
        return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", path.stem)]

    all_confidences = []   # (json_name, {species: {"TP": [...], "FP": [...]}})
    all_corrections = {}   # {json_name: {species: FN_count}}

    for json_file in sorted(folder.glob("*.json"), key=natural_key):
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

