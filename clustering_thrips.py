import os
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tqdm import tqdm
import shutil
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN

def load_images(image_dir, max_size=224):
    images = []
    filenames = []
    for fname in os.listdir(image_dir):
        if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
            path = os.path.join(image_dir, fname)
            img = cv2.imread(path)
            if img is None:
                print(f"Warning: Could not read {fname}")
                continue
            # Resize preserving aspect ratio, then center crop 224x224
            h, w = img.shape[:2]
            scale = max_size / min(h, w)
            img = cv2.resize(img, (int(w*scale), int(h*scale)))
            # Center crop
            h_new, w_new = img.shape[:2]
            startx = w_new//2 - (max_size//2)
            starty = h_new//2 - (max_size//2)
            img = img[starty:starty+max_size, startx:startx+max_size]
            images.append(img)
            filenames.append(fname)
    return images, filenames

def extract_features(images, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model = resnet50(pretrained=True)
    model.eval()
    model = model.to(device)

    # Use only up to layer3
    modules = list(model.children())
    feature_extractor = torch.nn.Sequential(
        modules[0],  # conv1
        modules[1],  # bn1
        modules[2],  # relu
        modules[3],  # maxpool
        modules[4],  # layer1
        modules[5],  # layer2
        modules[6],  # layer3
    ).to(device)

    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    features = []
    with torch.no_grad():
        for img in tqdm(images, desc="Extracting features from layer3"):
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            input_tensor = preprocess(img_rgb).unsqueeze(0).to(device)
            feat = feature_extractor(input_tensor)  # shape (1, C, H, W)
            feat = feat.squeeze().cpu().numpy()
            features.append(feat.flatten())  # Flatten to 1D vector
    return np.array(features)

def apply_pca(features, n_components=50):
    pca = PCA(n_components=n_components, random_state=42)
    reduced = pca.fit_transform(features)
    print(f"PCA: reduced from {features.shape[1]} to {n_components} dimensions.")
    return reduced

def cluster_and_save_kmeans(images, filenames, features_pca, output_dir, n_clusters=4):
    os.makedirs(output_dir, exist_ok=True)

    # KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(features_pca)

    print(f"KMeans found {n_clusters} clusters")

    # Create folders for clusters
    cluster_dirs = {}
    for label in range(n_clusters):
        cluster_folder = os.path.join(output_dir, f"cluster_{label}")
        os.makedirs(cluster_folder, exist_ok=True)
        cluster_dirs[label] = cluster_folder

    # Save images to corresponding cluster folders
    for img, fname, lbl in zip(images, filenames, labels):
        out_path = os.path.join(cluster_dirs[lbl], fname)
        cv2.imwrite(out_path, img)

    # t-SNE visualization
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(features_pca)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels, cmap='tab10', alpha=0.7)
    plt.legend(*scatter.legend_elements(), title="Clusters")
    plt.title("t-SNE visualization (KMeans)")
    plt.savefig(os.path.join(output_dir, "tsne_kmeans2_w_pca.png"))
    plt.close()

input_folder = "/user/christoph.wald/u15287/big-scratch/dataset/Thrips_boxes"
output_folder = "/user/christoph.wald/u15287/big-scratch/dataset/Thrips_boxes_clustered"     

print("Loading images...")
images, filenames = load_images(input_folder)
print(f"Loaded {len(images)} images.")

print("Extracting features using ResNet-50...")
features = extract_features(images)

print("Apply PCA")
features_pca = apply_pca(features, n_components=50)

print("Clustering and saving results...")
cluster_and_save_kmeans(images, filenames, features_pca, output_folder, n_clusters=4)

print("Done!")
