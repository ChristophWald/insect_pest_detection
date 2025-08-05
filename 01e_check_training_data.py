import os

def check_images_labels(base_path, split):
    img_dir = os.path.join(base_path, "images", split)
    label_dir = os.path.join(base_path, "labels", split)

    missing_labels = []
    missing_images = []

    # List images and labels
    images = {os.path.splitext(f)[0] for f in os.listdir(img_dir) if f.endswith('.jpg')}
    labels = {os.path.splitext(f)[0] for f in os.listdir(label_dir) if f.endswith('.txt')}

    # Check images without labels
    for img in images:
        if img not in labels:
            missing_labels.append(img + '.jpg')

    # Check labels without images
    for label in labels:
        if label not in images:
            missing_images.append(label + '.txt')

    print(f"Checking {split} split:")
    if missing_labels:
        print(f"  Images without labels ({len(missing_labels)}): {missing_labels}")
    else:
        print("  All images have corresponding labels.")

    if missing_images:
        print(f"  Labels without images ({len(missing_images)}): {missing_images}")
    else:
        print("  All labels have corresponding images.")


base_path = '/user/christoph.wald/u15287/big-scratch/supervised_large'  # Update if needed

for split in ['train', 'val', 'test']:
    check_images_labels(base_path, split)

