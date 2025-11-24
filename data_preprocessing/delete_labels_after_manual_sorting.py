import os

'''
deletes all lines in label files given by single .img-files with [species]_[filenumber]_[linenumber].jpg
these are single bounding boxes with insects for manual inspection that are sorted into folders 
used for cleaning the thrips labels
'''

source_path = "/user/christoph.wald/u15287/big-scratch/01_dataset/marcella_no_thrips"
dest_path = "/user/christoph.wald/u15287/big-scratch/01_dataset/labels/Thrips"

filenames = os.listdir(source_path)

result = {}

for f in filenames:
    base = f.replace('.jpg', '')
    string, filenumber, linenumber = base.split('_')
    key = f"{string}_{filenumber}"
    line = int(linenumber)
    result.setdefault(key, []).append(line)

print(result)

for key, lines_to_delete in result.items():
    txt_file = os.path.join(dest_path, f"{key}.txt")

    with open(txt_file, "r") as f:
        lines = f.readlines()

    
    with open(txt_file, "r") as f:
        lines = f.readlines()

    # delete safely: sort unique line numbers in descending order
    for line_no in sorted(set(lines_to_delete), reverse=True):
        if 1 <= line_no <= len(lines):
            del lines[line_no - 1]  # filenames are 1-based
            print(f"Deleted line {line_no} from {txt_file}")
        else:
            print(f"Line {line_no} out of range in {txt_file}")

    # save back the updated file
    with open(txt_file, "w") as f:
        f.writelines(lines)

    print(f"Updated {txt_file}")