import numpy as np
from ultralytics import YOLO

"""
Pose estimation with YOLO algorithm.

The YOLOv8 human pose estimation model detects 17 keypoints: 
    5 keypoints for the spine
    4 keypoints for the left arm
    4 keypoints for the right arm
    2 keypoints for the left leg
    2 keypoints for the right leg

@param img: image to estimate pose.
@return: pose tensor.
"""
def run_yolo_pose_estimation(img: np.ndarray) -> np.ndarray:
    model = YOLO("yolov8m-pose.pt")

    # Find person pose.
    results = model(img * 255.0, classes=[0])
    assert len(results) == 1, "More than one person found in image."
    print(results[0])
    pose_tensor = results[0].keypoints
    return pose_tensor[0].cpu().detach().numpy().astype(np.float32)