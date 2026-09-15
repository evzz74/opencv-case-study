import cv2
import numpy as np


def make_grid_image(
    width=320,
    height=240,
    vertical_x=(60, 120, 180, 240),
    horizontal_y=(60, 120, 180),
    thickness=5,
):
    image = np.zeros((height, width, 3), dtype=np.uint8)
    for x in vertical_x:
        cv2.line(image, (x, 15), (x, height - 15), (255, 255, 255), thickness)
    for y in horizontal_y:
        cv2.line(image, (15, y), (width - 15, y), (255, 255, 255), thickness)
    return image
