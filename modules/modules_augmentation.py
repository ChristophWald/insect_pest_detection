import sys
sys.path.append("/user/christoph.wald/u15287/insect_pest_detection/modules")
import cv2
import numpy as np
import random
from modules_segmentation import find_contour, find_corners, get_distance_h_mid, create_binary_mask


'''

for augmented set 1 (inserted background structures on training images)

crop: crops an image to the rectangle of the yellow sticky trap
circular_shift_mask_2d: shifts the background structure vertical and horizontal randomly
fit_mask_to_image: puts the background structure at random location of the image

for augmented set 2 (insects pasted on background structures)

random_shear: shears an image 
get_mask: returns a mask aligned to an image
enlarge_box: scales a rectangle by a given factor
keep_central_contour: keeps only the contour closest to the image center
iou: calculates IoU, used by place_insects_by_region
place_insect_at: plots an insect into an image, used by place_insects_by_region
get_candidate_positions: find a place to put the insect, used by place_insects_by_region
place_insects_by_region: places given number of insects randomly in areas selected by a mask
random_light_variation: augmentation for the final image



'''
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



def random_shear(image, max_shear=0.05):
    """
    Apply a random shear transformation to an image around its center.

    Parameters:
        image (numpy.ndarray): Input image (BGR or grayscale)
        max_shear (float): Maximum absolute shear factor for both x and y

    Returns:
        numpy.ndarray: Sheared image
    """
    rows, cols = image.shape[:2]

    # --- Random shear factors ---
    shear_x = random.uniform(-max_shear, max_shear)
    shear_y = random.uniform(-max_shear, max_shear)

    # --- Image center ---
    cx, cy = cols / 2, rows / 2

    # --- Translate to origin ---
    T1 = np.array([
        [1, 0, -cx],
        [0, 1, -cy],
        [0, 0, 1]
    ], dtype=np.float32)

    # --- Shear matrix ---
    S = np.array([
        [1, shear_x, 0],
        [shear_y, 1, 0],
        [0, 0, 1]
    ], dtype=np.float32)

    # --- Translate back ---
    T2 = np.array([
        [1, 0, cx],
        [0, 1, cy],
        [0, 0, 1]
    ], dtype=np.float32)

    # --- Combine all transformations ---
    M = T2 @ S @ T1
    M_affine = M[:2, :]

    # --- Apply affine transform ---
    sheared = cv2.warpAffine(image, M_affine, (cols, rows), flags=cv2.INTER_LINEAR)

    return sheared

def get_mask(image, processed_mask, gridcorners, mask_h_line):
    imageYST = find_contour(image)

    #save cropped image
    x, y, w, h = cv2.boundingRect(imageYST)
    

    #find corners if possible, if not skip the image
    imagecorners = find_corners(image, imageYST)
    if len(imagecorners) == 0:
        print(f"No YST found for {image_file}")
        return None

    #find transformation
    H, _ = cv2.findHomography(gridcorners, imagecorners, cv2.RANSAC)

    #first transformation: 
    mask = cv2.warpPerspective(processed_mask, H, (image.shape[1], image.shape[0]))
    
    #transform midline of mask as above
    h_line_pts = mask_h_line.reshape(-1,1,2)
    h_line_pts_warped = cv2.perspectiveTransform(h_line_pts, H)
    x1w, y1w, x2w, y2w = h_line_pts_warped.reshape(-1).astype(np.int32)
    mask_h = (x1w, y1w, x2w, y2w)

    #second transformation: correct vertical misalignment
    dy = get_distance_h_mid(create_binary_mask(image), mask_h)
    H, W = mask.shape[:2]
    M = np.float32([[1, 0, 0], [0, 1, dy]])  # translation matrix
    mask= cv2.warpAffine(mask, M, (W, H), borderValue=255)  # white background
    return mask[y:y+h, x:x+w]

def enlarge_box(box, image_shape, scale=0.2):
    """
    Enlarge a bounding box by a given percentage.

    Parameters:
        box (tuple): (x1, y1, x2, y2)
        image_shape (tuple): (height, width, channels) of the image
        scale (float): Fraction to enlarge the box (e.g., 0.2 = 20%)

    Returns:
        tuple: New enlarged box (x1_new, y1_new, x2_new, y2_new)
    """
    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1

    # Amount to expand
    dx = int(width * scale / 2)
    dy = int(height * scale / 2)

    # New coordinates
    x1_new = max(x1 - dx, 0)
    y1_new = max(y1 - dy, 0)
    x2_new = min(x2 + dx, image_shape[1] - 1)
    y2_new = min(y2 + dy, image_shape[0] - 1)

    return (x1_new, y1_new, x2_new, y2_new)

def keep_central_contour(img):
    """
    Keep only the contour closest to the image center.
    Other contours are filled white.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Threshold to binary (invert: foreground=white)
    _, binary = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)

    # Find external contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return img

    # Image center
    h, w = gray.shape
    img_center = np.array([w / 2, h / 2])

    # Find contour closest to center
    min_dist = float("inf")
    closest_contour = None
    for cnt in contours:
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
            dist = np.linalg.norm(np.array([cx, cy]) - img_center)
            if dist < min_dist:
                min_dist = dist
                closest_contour = cnt

    # Remove other contours by painting them white
    result = img.copy()
    for cnt in contours:
        if not np.array_equal(cnt, closest_contour):
            cv2.drawContours(result, [cnt], -1, (255, 255, 255), thickness=cv2.FILLED)

    return result

def iou(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    inter_x = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
    inter_y = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
    inter_area = inter_x * inter_y

    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)

    union_area = area1 + area2 - inter_area
    return inter_area / union_area if union_area > 0 else 0

def remove_border_connected(mask_binary):
    mask = mask_binary.copy().astype(np.uint8)
    num_labels, labels = cv2.connectedComponents(mask)
    output = np.zeros_like(mask)
    for label in range(1, num_labels):
        component = (labels == label)
        touches_border = np.any(component[0, :]) or np.any(component[-1, :]) or \
                         np.any(component[:, 0]) or np.any(component[:, -1])
        if not touches_border:
            output[component] = 1
    return output

def place_insects_by_region(base_image, mask_binary, insects, n_per_region=1, overlap_threshold=0.0, place_in_nonmask = False):
    """
    Place insects into 4 regions (2 rows × 2 uneven columns: 1/4 + 1/2 width),
    draw separation lines, and prevent overlapping.
    """
    h_img, w_img, _ = base_image.shape
    h_region = h_img // 2 - (h_img // 8)

    # Column boundaries
    x_col1_end = w_img // 4            # first column = 1/4 width
    x_col2_end = x_col1_end + w_img // 2  # second column = 1/2 width

    placed_count = 0
    placed_boxes = []

    # Draw separation lines
    #cv2.line(base_image, (0, h_region), (w_img, h_region), (0, 255, 0), 2)
    #cv2.line(base_image, (x_col1_end, 0), (x_col1_end, h_img), (0, 255, 0), 2)
    #cv2.line(base_image, (x_col2_end, 0), (x_col2_end, h_img), (0, 255, 0), 2)

    column_groups = [(0, x_col1_end), (x_col1_end, x_col2_end)]

    for row in range(2):
        y_start = row * h_region
        y_end = h_img if row == 1 else (row + 1) * h_region

        for i, (x_start, x_end) in enumerate(column_groups):
            region_mask = mask_binary[y_start:y_end, x_start:x_end]

            # Custom insect counts per region (example)
            if row == 0 and i == 0: n_to_place = int(n_per_region / 2)
            elif row == 0 and i == 1: n_to_place = int(n_per_region * 2)
            elif row == 1 and i == 1: n_to_place = int(n_per_region * 2)
            else: n_to_place = int(n_per_region)

            for _ in range(n_to_place):
                insect = insects.pop(0)
                #insect, _ = rotate_insect(insect)
                for _ in range(1000):  # try multiple random positions
                    

                    candidate_box = get_candidate_position(region_mask, insect, region_offset=(y_start, x_start))
                    if candidate_box is None:
                        break  # no valid pixel

                    # Check overlap
                    if all(iou(candidate_box, b) <= overlap_threshold for b in placed_boxes):
                        # Place insect
                        place_insect_at(base_image, insect, candidate_box)
                        placed_boxes.append(candidate_box)
                        placed_count += 1
                        break  # next insect
    if place_in_nonmask:
        # Use a separate list of insects for non-masked areas
        h_img, w_img, _ = base_image.shape

        for _ in range(placed_count):  # same number as placed in first pass
            insect = insects.pop(0)
            h_i, w_i, _ = insect.shape

            for _ in range(1000):
                # Pick a random position anywhere in the image
                y1 = random.randint(0, h_img - h_i)
                x1 = random.randint(0, w_img - w_i)
                candidate_box = (x1, y1, x1 + w_i, y1 + h_i)

                # Only accept if it does NOT overlap masked regions
                mask_patch = mask_binary[y1:y1 + h_i, x1:x1 + w_i]
                if np.any(mask_patch != 0):
                    continue  # skip positions that overlap the mask

                # Check overlap with previously placed insects
                if all(iou(candidate_box, b) <= overlap_threshold for b in placed_boxes):
                    place_insect_at(base_image, insect, candidate_box)
                    placed_boxes.append(candidate_box)
                    break
    #for box in placed_boxes:
    #        x1, y1, x2, y2 = box
    #        cv2.rectangle(base_image, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)  # red boxes

    return placed_count, placed_boxes

def place_insect_at(base_image, insect, box):
    """
    Paste the insect onto the base_image at the given bounding box.
    """
    x1, y1, x2, y2 = box
    region = base_image[y1:y2, x1:x2]

    # Create mask of non-white pixels
    non_white_mask = np.any(insect < 250, axis=2)
    mask_3d = np.repeat(non_white_mask[:, :, None], 3, axis=2)

    region[mask_3d] = insect[mask_3d]
    base_image[y1:y2, x1:x2] = region

def get_candidate_position(mask_binary, insect, region_offset=(0,0)):
    """
    Pick a random valid pixel from the mask and compute top-left coordinates
    for the insect placement (without actually pasting it yet).
    """
    y_offset, x_offset = region_offset
    h_insect, w_insect, _ = insect.shape
    valid_coords = np.argwhere(mask_binary == 1)
    if len(valid_coords) == 0:
        return None  # no valid place

    # Try multiple random pixels
    for _ in range(1000):
        y_local, x_local = valid_coords[np.random.randint(len(valid_coords))]
        y_center = y_offset + y_local
        x_center = x_offset + x_local

        y_top = y_center - h_insect // 2
        x_left = x_center - w_insect // 2
        y_bottom = y_top + h_insect
        x_right = x_left + w_insect

        # Check boundaries
        if x_left < 0 or y_top < 0 or x_right > mask_binary.shape[1] + x_offset or y_bottom > mask_binary.shape[0] + y_offset:
            continue

        return (x_left, y_top, x_right, y_bottom)

    return None  # failed to find position

def random_light_variation(image):
    """
    Apply a random daylight-like variation to an image by adjusting
    color balance, brightness, and saturation — without using named presets.
    
    Parameters:
        image (numpy.ndarray): Input BGR image.
    
    Returns:
        numpy.ndarray: The adjusted image.
    """
    img = image.astype(np.float32) / 255.0

    # --- Randomize approximate "temperature" range (cool ↔ warm)
    #  Lower values -> warmer (more red/yellow)
    #  Higher values -> cooler (more blue)
    t_value = random.uniform(4500, 7500)
    t_norm = np.clip(t_value, 1000, 40000) / 100.0

    # Compute simple color balance vector
    if t_norm <= 66:
        r = 255
        g = 99.47 * np.log(t_norm) - 161.12
        b = 0 if t_norm <= 19 else 138.52 * np.log(t_norm - 10) - 305.04
    else:
        r = 329.70 * ((t_norm - 60) ** -0.1332)
        g = 288.12 * ((t_norm - 60) ** -0.0755)
        b = 255
    balance = np.clip(np.array([b, g, r]) / 255.0, 0, 1)

    # --- Randomize brightness and saturation ---
    brightness_factor = random.uniform(0.8, 1.3)
    saturation_factor = random.uniform(0.8, 1.4)

    # --- Apply color balance ---
    img *= balance
    img = np.clip(img, 0, 1)

    # --- Apply brightness ---
    img *= brightness_factor
    img = np.clip(img, 0, 1)

    # --- Apply saturation ---
    hsv = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] *= saturation_factor
    hsv[..., 1] = np.clip(hsv[..., 1], 0, 255)
    result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    return result

