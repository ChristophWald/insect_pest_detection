import os
import matplotlib.pyplot as plt
import cv2

# Paths
first_image_path = "/user/christoph.wald/u15287/big-scratch/00_uncropped_dataset/YSTohneInsekten/IMG_5885.JPG"
image_dir = "/user/christoph.wald/u15287/insect_pest_detection/2_4_image_processing/masks"

# Load the first image
img1 = cv2.imread(first_image_path)
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)  # convert BGR → RGB for matplotlib

# Load other images from directory
all_images = sorted([f for f in os.listdir(image_dir) if f.endswith(".jpg")])
# Take first five images (skip first_image if it's also in the folder)
other_paths = [os.path.join(image_dir, f) for f in all_images[:5]]

imgs = [img1]  # start with first image
for path in other_paths:
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    imgs.append(img)

# Plot in 3x2 grid
fig, axes = plt.subplots(3, 2, figsize=(10, 12))
axes = axes.flatten()

for i, ax in enumerate(axes):
    ax.imshow(imgs[i])
    ax.axis('off')

plt.tight_layout()
plt.savefig("mask_overview.jpg")
