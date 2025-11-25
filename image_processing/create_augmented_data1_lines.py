import sys
sys.path.append("/user/christoph.wald/u15287/insect_pest_detection/modules")
import cv2
#import matplotlib.pyplot as plt
from modules_segmentation import *
from modules_augmentation import *
import random
import os

'''
augments all images by adding two randomly shifted background structures
background of image is separated from foreground (= insects)
background structures taken from empty yellow sticky traps are placed on them 
finally insects are restored.
'''

output_path = "/user/christoph.wald/u15287/big-scratch/02_splitted_data/train_unlabeled/06_masked_augmented2_reduced"
os.makedirs(output_path, exist_ok = True)

image_path = "/user/christoph.wald/u15287/big-scratch/02_splitted_data/train_unlabeled/03_images_masked_reduced"
filenames = os.listdir(image_path)

#select the first three of the empty images as masks for the lines
empty_folder = "/user/christoph.wald/u15287/big-scratch/emptyYST"
filenames = os.listdir(empty_folder)[:3]
masks = []
for f in filenames:
    img = cv2.imread(os.path.join(empty_folder, f))
    empty_YST, _ = crop(img)
    mask = create_binary_mask(empty_YST) 
    cleaned_mask = denoise_mask(mask)
    h, w = cleaned_mask.shape

    top = 75
    bottom = h - 100
    left = 100
    right = w * 4 // 5  # remove the rightmost 1/5

    cleaned_mask = cleaned_mask[top:bottom, left:right] #cut borders to prevent large black areas
    print(cleaned_mask.shape)
    masks.append(cleaned_mask)


#create the images
for f in filenames:
    img = cv2.imread(os.path.join(image_path, f))
    
    #create first lines
    mask = masks[random.randint(0,len(masks)-1)]
    mask = circular_shift_mask_2d(mask)
    mask = fit_mask_to_image(mask, img.shape)
    mask0 = mask
    #creates second lines
    
    mask = masks[random.randint(0,len(masks)-1)]
    mask = circular_shift_mask_2d(mask)
    mask = cv2.rotate(mask, cv2.ROTATE_180) #rotated
    mask = fit_mask_to_image(mask, img.shape)
    mask1 = mask

    #create black lines mask from shifted masks ---
    line_mask = (mask0 == 0).astype(np.uint8) * 255
    line_mask_3c0 = cv2.merge([line_mask]*3)

    line_mask = (mask1 == 0).astype(np.uint8) * 255
    line_mask_3c1 = cv2.merge([line_mask]*3)
    
    #create background from image
    yellow_mask = create_binary_mask(img)  # 255 = background
    yellow_mask_3c = cv2.merge([yellow_mask]*3)

    background_layer = img.copy()
    background_layer[yellow_mask_3c == 0] = 0

    # overlay black lines
    merged = background_layer.copy()
    merged[line_mask_3c0 == 255] = 0  # black lines
    merged[line_mask_3c1 == 255] = 0  # black lines

    #restore foreground
    foreground_layer = np.zeros_like(img)
    foreground_layer[yellow_mask_3c == 0] = img[yellow_mask_3c == 0]
    merged[yellow_mask_3c == 0] = img[yellow_mask_3c == 0]

    cv2.imwrite(os.path.join(output_path, f), merged)