import os

'''
remaps class indices in yolo label files given in a folder
'''

# Define remapping of class IDs
CLASS_ID_MAP = {
    3: 0,
    0: 1,
    4: 2,
    1: 3,
}

def remap_class_ids_in_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    new_lines = []
    changed = False

    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue  # skip malformed lines

        try:
            old_id = int(parts[0])
        except ValueError:
            continue  # skip lines where class ID isn't an integer

        if old_id in CLASS_ID_MAP:
            new_id = CLASS_ID_MAP[old_id]
            if new_id != old_id:
                changed = True
            parts[0] = str(new_id)
        new_lines.append(" ".join(parts))

    if changed:
        with open(file_path, 'w') as f:
            f.write("\n".join(new_lines) + "\n")
        print(f"✅ Updated {os.path.basename(file_path)}")
    else:
        print(f"⏩ Skipped {os.path.basename(file_path)} (no changes)")

def process_folder(root_folder):
    for root, _, files in os.walk(root_folder):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                remap_class_ids_in_file(file_path)

# Change this to your folder path
folder = "/user/christoph.wald/u15287/big-scratch/02_splitted_data/test_set/test_set_w_new_labels/labels"
process_folder(folder)