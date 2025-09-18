import os
import ast
import numpy as np
import matplotlib.pyplot as plt


# --- Folder containing rectangle files ---
rect_folder = "/user/christoph.wald/u15287/insect_pest_detection/image_processing_in_progress/data_unlabeled_training_set/rectangles"
rect_files = sorted([f for f in os.listdir(rect_folder) if f.endswith(".txt")])

num_files = len(rect_files)
fig, axes = plt.subplots(num_files, 2, figsize=(12, num_files * 4))  # 2 histograms per file

for i, file_name in enumerate(rect_files):
    areas = []
    square_devs = []

    with open(os.path.join(rect_folder, file_name), "r") as f:
        for line in f:
            line = line.strip()
            if line:  # skip empty lines
                rects = ast.literal_eval(line)
                for x, y, w, h in rects:
                    areas.append(w * h)
                    square_devs.append(abs(w / h - 1))  # deviation from square

    areas = np.array(areas)
    square_devs = np.array(square_devs)

    # --- Step 2: Compute percentiles ---
    print(f"{file_name} stats:")
    print(f"95th percentile area, max: {np.percentile(areas,95):.0f}, {np.max(areas)}")

    print(f"Square deviation 95th: { np.percentile(square_devs, 95):.2f}")

    # Histogram for areas
    axes[i, 0].hist(areas, bins=40, color='skyblue', edgecolor='black')
    axes[i, 0].axvline(np.percentile(areas, 95), color='green', linestyle='--', label='95th percentile')
    axes[i, 0].set_title(f"{file_name} - Rectangle Areas")
    axes[i, 0].set_xlabel("Area")
    axes[i, 0].set_ylabel("Count")
    axes[i, 0].legend()

    # Histogram for square deviation
    axes[i, 1].hist(square_devs, bins=40, color='salmon', edgecolor='black')
    axes[i, 1].axvline(np.percentile(square_devs, 95), color='green', linestyle='--', label='95th percentile')
    axes[i, 1].set_title(f"{file_name} - Square Deviation")
    axes[i, 1].set_xlabel("Square Deviation")
    axes[i, 1].set_ylabel("Count")
    axes[i, 1].legend()

plt.tight_layout()
plt.savefig("all_rectangle_distributions.jpg", dpi=300)
plt.show()
