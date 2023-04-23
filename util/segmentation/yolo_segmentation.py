import numpy as np
from ultralytics import YOLO

"""
Image segmentation with YOLO algorithm.
We only segment the person in the image.

@param img: image to segment.
@return: segmentation mask.
"""
def run_yolo_segmentation(img: np.ndarray) -> np.ndarray:
    model = YOLO("yolov8m-seg.pt")

    # Find person only.
    results = model.predict(img * 255.0, classes=[0], retina_masks=True)
    if len(results) == 0:
        print("No person found in image with YOLO.")
        return None
    assert results[0].names[0] == "person", "Model is not configured properly to find person."
    assert len(results) == 1, "More than one person found in image."
    return results[0].masks.data.detach().cpu().numpy()[0].astype(np.float32)