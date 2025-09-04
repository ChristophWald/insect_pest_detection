import os
import hashlib
from collections import defaultdict

def compute_sha256(filepath):
    sha256 = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            sha256.update(chunk)
    return sha256.hexdigest()

hash_map = defaultdict(list)
file_counter = 0  # Count of files processed

for root, _, files in os.walk(os.getcwd()):
    for file in files:
        filepath = os.path.join(root, file)
        file_hash = compute_sha256(filepath)
        hash_map[file_hash].append(filepath)
        file_counter += 1
        if file_counter % 100 == 0:
            print(f"Processed {file_counter} files...")

print(f"\nTotal files processed: {file_counter}")

# Find and print duplicates
duplicates_found = False
for hash_val, paths in hash_map.items():
    if len(paths) > 1:
        duplicates_found = True
        print(f"\n🔁 Duplicate files (SHA256: {hash_val}):")
        for path in paths:
            print(f"  - {path}")

if not duplicates_found:
    print("✅ No duplicate files found.")

