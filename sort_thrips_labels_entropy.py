import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy

def compute_hue_entropy(img, hist_bins=60):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0].flatten()
    hue_filtered = hue[(hue >= 0) & (hue <= 60)]
    bins = np.linspace(0, 60, hist_bins + 1)
    hist, _ = np.histogram(hue_filtered, bins=bins, density=True)
    return entropy(hist + 1e-10)  # avoid log(0)

def plot_image_and_hue_histogram(img, save_path, hist_bins=60):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0].flatten()
    hue_filtered = hue[(hue >= 0) & (hue <= 60)]

    bins = np.linspace(0, 60, hist_bins + 1)
    hist, _ = np.histogram(hue_filtered, bins=bins, density=True)

    mean_hue = np.mean(hue_filtered) if hue_filtered.size > 0 else np.nan
    std_hue = np.std(hue_filtered) if hue_filtered.size > 0 else np.nan
    entropy_hue = entropy(hist + 1e-10) if hue_filtered.size > 0 else np.nan

    fig, axs = plt.subplots(1, 2, figsize=(14, 4))
    axs[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axs[0].set_title("Original Image")
    axs[0].axis('off')

    axs[1].bar(bins[:-1], hist, width=(bins[1] - bins[0]), color='orange', edgecolor='black')
    axs[1].set_title(f"Hue Histogram (0–60)\nMean: {mean_hue:.1f}, Std: {std_hue:.1f}, Entropy: {entropy_hue:.2f}")
    axs[1].set_xlim(0, 60)
    axs[1].set_xlabel("Hue")
    axs[1].set_ylabel("Density")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def sort_images_by_entropy(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))]

    image_data = []

    # Compute entropy for each image
    for filename in image_files:
        path = os.path.join(input_folder, filename)
        img = cv2.imread(path)
        if img is None:
            print(f"⚠️ Skipping unreadable file: {filename}")
            continue
        ent = compute_hue_entropy(img)
        image_data.append((img, filename, ent))

    # Sort by entropy (ascending)
    sorted_images = sorted(image_data, key=lambda x: x[2])

    # Save images with entropy plots
    for i, (img, filename, ent) in enumerate(sorted_images, 1):
        name, ext = os.path.splitext(filename)
        new_filename = f"{i:03d}_{name}.png"
        out_path = os.path.join(output_folder, new_filename)

        # Save the plot (original image + histogram)
        plot_image_and_hue_histogram(img, out_path)
        print(f"Saved: {new_filename} (entropy: {ent:.4f})")

input_folder = "/user/christoph.wald/u15287/big-scratch/dataset/Thrips_boxes"
output_folder = "/user/christoph.wald/u15287/big-scratch/dataset/Thrips_boxes_sorted" 


sort_images_by_entropy(input_folder, output_folder)