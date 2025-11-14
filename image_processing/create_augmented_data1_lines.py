import sys
sys.path.append("/user/christoph.wald/u15287/insect_pest_detection/modules")
import cv2
#import matplotlib.pyplot as plt
from modules_segmentation import *
from modules_augmentation import *
import random
import os

output_path = "/user/christoph.wald/u15287/big-scratch/02_splitted_data/train_unlabeled/06_masked_augmented2_reduced"
os.makedirs(output_path, exist_ok = True)


def crop(image):
    mask = create_binary_mask(image)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        cropped_image = image[y:y+h, x:x+w]
        return cropped_image, (x, y, w, h)
    else:
        return None, None
    
def fit_mask_to_image(mask, img_shape):
    """
    Resize/crop or place mask to fit the image shape.
    If mask is smaller: place randomly.
    If mask is larger: crop to fit.
    """
    mask_h, mask_w = mask.shape
    img_h, img_w = img_shape[:2]

    # If mask is larger, crop it
    if mask_h > img_h:
        start_y = (mask_h - img_h) // 2
        mask = mask[start_y:start_y+img_h, :]
    if mask_w > img_w:
        start_x = (mask_w - img_w) // 2
        mask = mask[:, start_x:start_x+img_w]

    # If mask is smaller, place it randomly
    mask_h, mask_w = mask.shape
    result_mask = np.ones((img_h, img_w), dtype=mask.dtype) * 255  # white background
    max_y = img_h - mask_h
    max_x = img_w - mask_w
    start_y = random.randint(0, max_y) if max_y > 0 else 0
    start_x = random.randint(0, max_x) if max_x > 0 else 0
    result_mask[start_y:start_y+mask_h, start_x:start_x+mask_w] = mask
    return result_mask

def circular_shift_mask_2d(mask, min_shift=500):
    h, w = mask.shape
    max_shift_h = h // 2
    max_shift_w = w // 2
    shift_y = random.randint(min_shift, max_shift_h) * random.choice([-1, 1])
    shift_x = random.randint(min_shift, max_shift_w) * random.choice([-1, 1])
    shifted_mask = np.roll(mask, shift_y, axis=0)
    shifted_mask = np.roll(shifted_mask, shift_x, axis=1)
    return shifted_mask


image_path = "/user/christoph.wald/u15287/big-scratch/02_splitted_data/train_unlabeled/03_images_masked_reduced"
filenames = os.listdir(image_path)
random_filenames = filenames

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
for f in random_filenames:
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

    # Soverlay black lines
    merged = background_layer.copy()
    merged[line_mask_3c0 == 255] = 0  # black lines
    merged[line_mask_3c1 == 255] = 0  # black lines

    #restore foreground
    foreground_layer = np.zeros_like(img)
    foreground_layer[yellow_mask_3c == 0] = img[yellow_mask_3c == 0]
    merged[yellow_mask_3c == 0] = img[yellow_mask_3c == 0]

    cv2.imwrite(os.path.join(output_path, f), merged)