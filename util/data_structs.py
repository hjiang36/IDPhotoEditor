import numpy as np
import cv2
import os

from .segmentation.grabcut_segmentation import run_grabcut_segmentation
from .segmentation.mask_rcnn_segmentation import run_mask_rcnn_segmentation
from .segmentation.yolo_segmentation import run_yolo_segmentation

from .matting import matte_former

from .pose.yolo_pose import run_yolo_pose_estimation
from .pose.mediapipe_facial_landmark import get_face_bounding_box

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

        self.image_matter = None
        self.mask_color = (0.0, 0.0, 0.0, 0.5)

    """
    Open an image file.

    @param file_path: full path to image file.
    @returns: if the file opens properly.
    """
    def open_image(self, file_path: str) -> bool:
        if not os.path.exists(file_path):
            return False
        self.img = cv2.cvtColor(cv2.imread(filename=file_path), cv2.COLOR_BGR2RGB)
        if self.img.shape[0] * self.img.shape[1] > 2000000:
            self.img = cv2.resize(self.img, (0, 0), fx=0.5, fy=0.5)
        self.img = (self.img / 255.0).astype(np.float32)
        self.open_image_with_array(self.img, file_path)
        return True

    """
    Open image with numpy array.

    @param img: image data to open.
    @param file_path: full path to image file.
    """
    def open_image_with_array(self, img: np.ndarray, file_path:str="") -> None:
        self.img = img
        self.mask = np.ones_like(self.img)
        self.segmentation_index = 0
        self.img_path = file_path


    """
    Save the data to numpy binary file.

    @param file_path: full path to save the file.
    """
    def dump_to_file(self, file_path: str) -> None:
        np.savez_compressed(
            file_path,
            img=self.img,
            mask=self.mask,
            img_path=self.img_path,
            mask_color=self.mask_color)

    """
    Load the data from numpy binary file.
    """
    def load_from_file(self, file_path: str) -> None:
        parameters = np.load(file_path)
        self.img = parameters["img"]
        self.mask = parameters["mask"]
        self.img_path = parameters["img_path"]
        self.mask_color = parameters["mask_color"]

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

    """
    Image matting to precisely determine foreground and background.
    This function will classify each unknown pixel (0.5) into foreground or background.

    @return: the matted mask (0 means background, 1 means foreground).
    """
    def run_image_matting(self) -> np.ndarray:
        if self.img is None or self.mask is None:
            return None
        if self.image_matter is None:
            self.image_matter = matte_former.MatteFormerMatting(
                "C:\\Users\\haomiao\\Downloads\\matteformer-master\\matte_former.pth")
        self.mask = self.image_matter.run_matte_former_matting(self.img, self.mask)
        return self.mask
    
    """
    Estimate the proper crop region for the image.
    TODO: we should accept a config settings about target requirements.

    """
    def propose_crop_region(self) -> np.ndarray:
        if self.img is None:
            return None
        face_bbx = get_face_bounding_box(self.img)
        face_width = face_bbx[1] - face_bbx[0]

        target_height = face_width * 2.0 / 33.0 * 48.0
        center_height = face_bbx[2] + (face_bbx[3] - face_bbx[2]) * 0.33
        return np.array(
            [face_bbx[0] - face_width * 0.5,
             face_bbx[1] + face_width * 0.5,
             center_height - target_height * 0.5,
             center_height + target_height * 0.5])
