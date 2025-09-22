import os
import shutil

#setup base folder
source_folder = "/user/christoph.wald/u15287/big-scratch/02_splitted_data/train_labeled/tiles"
dest_folder = "/user/christoph.wald/u15287/big-scratch/SSL_training_data/training_data"

#setup file structure for destination
file_types = ["images", "labels"]
use_types = ["train", "val"]

for f in file_types:
    for u in use_types:
        path = os.path.join(dest_folder, f, u)
        os.makedirs(path, exist_ok= True)



#copy all tiles and labels from train/val source folders, but only if they contain objects
for u in use_types:
    
    print(f"Copying {u} data.")
    skipped = 0
    
    label_path = os.path.join(source_folder, u, "labels")
    img_path = os.path.join(source_folder, u, "images")
    label_dest_path = os.path.join(dest_folder, "labels", u)
    img_dest_path = os.path.join(dest_folder, "images", u)
    label_files = os.listdir(label_path)
    
    for file in label_files:
        with open(os.path.join(label_path, file), "r") as f:
            if f.read().strip() == "":
                skipped += 1
                continue
            else:
                img_file = os.path.splitext(file)[0] + ".jpg"
                shutil.copy2(os.path.join(img_path, img_file), os.path.join(img_dest_path, img_file))
                shutil.copy2(os.path.join(label_path, file), os.path.join(label_dest_path, file))

    total = len(label_files)
    print(f"Copied {total-skipped} files from {total} files (used {(total-skipped)/total*100:.2f} %)")