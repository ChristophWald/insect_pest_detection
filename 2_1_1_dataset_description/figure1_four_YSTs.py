import cv2
import numpy as np

'''
creates figure 1 - four YST with the four pest classes

'''

#input and output paths
image1_path = "/home/wald/Schreibtisch/10_BA_Arbeit/00_data_cleaning/dataset_uncropped_after_cleaning/images/FungusGnats_sorted/BRAIIM_0008.jpg"
image2_path = "/home/wald/Schreibtisch/10_BA_Arbeit/00_data_cleaning/dataset_uncropped_after_cleaning/images/LeafMinerFlies_sorted/LIRIBO_0007.jpg"
image3_path = "/home/wald/Schreibtisch/10_BA_Arbeit/00_data_cleaning/dataset_uncropped_after_cleaning/images/Thrips_sorted/FRANOC_0021.jpg"
image4_path = "/home/wald/Schreibtisch/10_BA_Arbeit/00_data_cleaning/dataset_uncropped_after_cleaning/images/WhiteFlies_sorted/TRIAVA_0008.jpg"

output_path = "/home/wald/Schreibtisch/10_BA_Arbeit/images/figure1.jpg"

#create the figure
images = [cv2.imread(path) for path in [image1_path, image2_path, image3_path, image4_path]]

top_row = np.hstack((images[0], images[1]))
bottom_row = np.hstack((images[2], images[3]))
grid_image = np.vstack((top_row, bottom_row))

scale_factor = 0.1  
new_width = int(grid_image.shape[1] * scale_factor)
new_height = int(grid_image.shape[0] * scale_factor)
grid_resized = cv2.resize(grid_image, (new_width, new_height), interpolation=cv2.INTER_AREA)

cv2.imwrite(output_path, grid_resized)
print(f"Saved 2x2 grid image as {output_path}")