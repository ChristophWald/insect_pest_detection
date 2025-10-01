import cv2
import torch
import matplotlib.pyplot as plt

def draw_boxes(img, bboxes, color=(0, 255, 0), thickness=2):
    """
    Draw multiple bounding boxes on an image.

    Args:
        img (np.ndarray): OpenCV image.
        bboxes (torch.Tensor or np.ndarray): Shape (N, 4), each [x1, y1, x2, y2].
        color (tuple): BGR color for the boxes.
        thickness (int): Line thickness.

    Returns:
        np.ndarray: Image with drawn rectangles.
    """
    img_copy = img.copy()

    if isinstance(bboxes, torch.Tensor):
        bboxes = bboxes.detach().cpu().numpy().astype(int)

    for x1, y1, x2, y2 in bboxes:
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, thickness)

    return img_copy


def show(img):
    """
    Displays an OpenCV image with matplotlib (RGB format).
    """
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(8, 6))
    plt.imshow(img_rgb)
    plt.axis("off")
    plt.show()