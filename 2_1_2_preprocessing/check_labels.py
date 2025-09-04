import os

'''
find labels that do not match the pest class of the subset
'''

def report_mismatched_labels(root_path, subfolders):
    """
    Reports .txt filenames where the class label does not match the subfolder index.
    
    root_path: str - absolute or relative path to the root directory
    subfolders: list - list of subfolder names (relative to root_path) to process
    """
    for idx, subfolder in enumerate(subfolders):
        folder = os.path.join(root_path, subfolder)
        for root, _, files in os.walk(folder):
            for file in files:
                if file.endswith(".txt"):
                    path = os.path.join(root, file)
                    with open(path, 'r') as f:
                        lines = f.readlines()

                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) != 5:
                            continue

                        try:
                            cls = int(parts[0])
                        except ValueError:
                            continue

                        if cls != idx:
                            print(f"{file} (found class {cls}, expected {idx})")
                            break  # Only report once per file

# Example usage
root_path = "/home/wald/Schreibtisch/10_BA_Arbeit/01_data_preparation/dataset/labels"
subfolders = ["LeafMinerFlies", "WhiteFlies", "skip", "FungusGnats",  "Thrips", ]
report_mismatched_labels(root_path, subfolders)

