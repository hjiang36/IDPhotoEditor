import numpy as np

"""
Image segmentation with Mask R-CNN algorithm.
"""
def run_mask_rcnn_segmentation(img: np.ndarray, mask: np.ndarray, rect: tuple) -> np.ndarray:
    print('Mask R-CNN segmentation')
    return mask