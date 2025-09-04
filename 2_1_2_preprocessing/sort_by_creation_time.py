import os
from PIL import Image
from PIL.ExifTags import TAGS
from datetime import datetime
import shutil

def get_exif_creation_date(filepath):
    try:
        image = Image.open(filepath)
        exif_data = image._getexif()
        if not exif_data:
            return None
        for tag_id, value in exif_data.items():
            tag = TAGS.get(tag_id)
            if tag == 'DateTimeOriginal':
                return datetime.strptime(value, '%Y:%m:%d %H:%M:%S')
    except Exception as e:
        print(f"Error reading EXIF from {filepath}: {e}")
    return None

def find_and_sort_images(images_dir):
    jpg_files = []
    for file in os.listdir(images_dir):
        if file.lower().endswith(('.jpg', '.jpeg')):
            filepath = os.path.join(images_dir, file)
            date = get_exif_creation_date(filepath)
            if date:
                jpg_files.append((filepath, date))
            else:
                print(f"No EXIF date for: {filepath}")
    jpg_files.sort(key=lambda x: x[1])
    return jpg_files

def extract_prefix(filename):
    base = os.path.basename(filename)
    return base.split('_')[0] if '_' in base else 'img'

def process_class_subfolder(class_name, base_images='images', base_labels='labels'):
    images_subdir = os.path.join(base_images, class_name)
    labels_subdir = os.path.join(base_labels, class_name)

    images_sorted_dir = os.path.join(base_images, class_name + "_sorted")
    labels_sorted_dir = os.path.join(base_labels, class_name + "_sorted")

    os.makedirs(images_sorted_dir, exist_ok=True)
    os.makedirs(labels_sorted_dir, exist_ok=True)

    image_list = find_and_sort_images(images_subdir)
    if not image_list:
        print(f"[{class_name}] No images found.")
        return

    prefix = extract_prefix(os.path.basename(image_list[0][0]))
    print(f"\n[{class_name}] Prefix: {prefix}")

    for index, (img_path, date) in enumerate(image_list, start=1):
        new_base_name = f"{prefix}_{index:04d}"
        ext = os.path.splitext(img_path)[1].lower()
        new_img_path = os.path.join(images_sorted_dir, new_base_name + ext)

        print(f"  {date} - {img_path} -> {new_img_path}")
        shutil.copy2(img_path, new_img_path)

        # Process label
        old_label_name = os.path.splitext(os.path.basename(img_path))[0] + ".txt"
        old_label_path = os.path.join(labels_subdir, old_label_name)
        new_label_path = os.path.join(labels_sorted_dir, new_base_name + ".txt")

        if os.path.exists(old_label_path):
            print(f"         label: {old_label_path} -> {new_label_path}")
            shutil.copy2(old_label_path, new_label_path)
        else:
            print(f"         label: {old_label_path} [NOT FOUND]")

def main():
    class_folders = ["FungusGnats", "LeafMinerFlies", "Thrips", "WhiteFlies"]
    for class_name in class_folders:
        process_class_subfolder(class_name)

if __name__ == "__main__":
    main()

