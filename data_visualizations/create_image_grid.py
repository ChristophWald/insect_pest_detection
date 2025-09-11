import os
import math
import matplotlib.pyplot as plt
from PIL import Image

def plot_image_grid(folder_path, cols=10, thumb_px=128, output_file="image_grid.png"):
    # Get all image files in the folder
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"}
    files = [f for f in os.listdir(folder_path) if os.path.splitext(f)[1].lower() in valid_exts]
    files.sort()

    n_images = len(files)
    if n_images == 0:
        print("No images found in the folder.")
        return

    rows = math.ceil(n_images / cols)

    # Create figure (adjust figsize to scale thumbnails nicely)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    axes = axes.flatten() if n_images > 1 else [axes]

    for ax in axes:
        ax.axis("off")

    for idx, file in enumerate(files):
        img_path = os.path.join(folder_path, file)
        print(f"Loading {img_path}")
        try:
            img = Image.open(img_path)
            img.thumbnail((thumb_px, thumb_px))  # Resize to thumbnail
            axes[idx].imshow(img)
            axes[idx].set_title(file, fontsize=6)
        except Exception as e:
            print(f"Could not load {file}: {e}")

    # Hide unused axes
    for ax in axes[n_images:]:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_file, dpi=150)  # Save instead of show
    plt.close(fig)  # Close to free memory
    print(f"Grid saved to {output_file}")

plot_image_grid("/user/christoph.wald/u15287/big-scratch/02_splitted_data/train_labeled/images_uncropped_w_labels")
