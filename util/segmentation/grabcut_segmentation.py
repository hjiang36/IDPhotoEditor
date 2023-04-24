import cv2
import numpy as np

"""
Image segmentation with OpenCV GrabCut algorithm.
"""
def run_grabcut_segmentation(img: np.ndarray, mask: np.ndarray, rect: tuple) -> np.ndarray:
    print("GrabCut segmentation not yet implemented.")
    mask = cv2.grabCut(img, mask, rect, None, None, 5, cv2.GC_INIT_WITH_RECT)
    mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    return mask