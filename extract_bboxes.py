from modules import * 

'''
takes all labels from a directory and puts out all bounding boxes as individual jpgs
'''

image_path = "/user/christoph.wald/u15287/big-scratch/dataset/images/Thrips"
label_path = "/user/christoph.wald/u15287/big-scratch/dataset/labels/Thrips"
output_path = "/user/christoph.wald/u15287/big-scratch/dataset/Thrips_boxes"

if not os.path.exists(output_path):
    os.makedirs(output_path)

txt_files = [f for f in os.listdir(label_path) if f.endswith('.txt')]

for txt_file in txt_files:
    base_name = os.path.splitext(txt_file)[0]

    img_path = os.path.join(image_path, base_name + ".jpg")
    if not os.path.exists(img_path):
        print(f"Image not found: {img_path}, skipping.")
        continue
    print(f"Processing {img_path}")
    labels_path = os.path.join(label_path, txt_file)
    extract_boxes(img_path, labels_path, output_path)

