import numpy as np
import cv2
import os

from .segmentation.grabcut_segmentation import run_grabcut_segmentation
from .segmentation.mask_rcnn_segmentation import run_mask_rcnn_segmentation
from .segmentation.yolo_segmentation import run_yolo_segmentation

"""
Image data structure class to hold image / mask etc. info.
"""
class ImageDataStructs:
    """
    Initialize the empty struct
    """
    def __init__(self) -> None:
        self.img: np.ndarray = None
        self.mask: np.ndarray = None

        self.img_path = ""
        self.segmentation_options = ["GrabCut", "Mask-RCNN", "YOLO"]
        self.segmentation_index = 0

    """
    Open an image file.

    @param file_path: full path to image file.
    @returns: if the file opens properly.
    """
    def open_image(self, file_path: str) -> bool:
        if not os.path.exists(file_path):
            return False
        self.img = cv2.cvtColor(cv2.imread(filename=file_path), cv2.COLOR_BGR2RGB)
        self.img = (self.img / 255.0).astype(np.float32)
        self.mask = np.ones_like(self.img)
        self.img_path = file_path
        return True

    """
    Get width of image.
    """
    def width(self) -> int:
        if self.img is None:
            return 0
        return self.img.shape[1]
    
    """
    Get height of image.
    """
    def height(self) -> int:
        if self.img is None:
            return 0
        return self.img.shape[0]

    """
    Segment the image using the method specified in segmentation_method.

    @param erode_dilate_size: size of erode / dilate kernel. -1 means no erode / dilate.
    @return: the segmetation mask.
    """
    def run_segmentation(self, erode_dilate_size=-1) -> np.ndarray:
        if self.img is None:
            return None
        segmentation_method = self.segmentation_options[self.segmentation_index]
        if segmentation_method == "Mask R-CNN":
            self.mask = run_mask_rcnn_segmentation(self.img, self.mask, (0, 0, self.width(), self.height()))
        elif segmentation_method == "YOLO":
            self.mask = run_yolo_segmentation(self.img)
        elif segmentation_method == "GrabCut":
            self.mask = run_grabcut_segmentation(self.img, self.mask, (0, 0, self.width(), self.height()))
        else:
            print("Unknown segmentation method: " + segmentation_method)
            return None
        
        # Erode / dilate the mask to create trimap.
        # In this trimap representation, 0 means background, 1 means foreground, and 0.5 means unknown.
        if erode_dilate_size > 0:
            kernel = np.ones((erode_dilate_size, erode_dilate_size), np.uint8)
            mask = self.mask
            self.mask = 0.5 * np.ones_like(self.mask)
            self.mask[cv2.erode(mask, kernel=kernel) > 0.6] = 1.0
            self.mask[cv2.dilate(mask, kernel=kernel) < 0.4] = 0.0
        return self.mask

    """
    Mark region of image as unknown.
    """
    def mark_unknown(self, x: int, y: int, radius: int) -> None:
        if self.mask is None:
            return
        self.mask[
            max(y - radius, 0):min(y + radius, self.height()),
            max(x - radius, 0):min(x + radius, self.width())] = 0.5