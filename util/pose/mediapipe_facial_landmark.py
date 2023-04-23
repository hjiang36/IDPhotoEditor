import mediapipe as mp
import numpy as np

"""
Run media pipe facial landmark detection on a single image.

The facial landmark detection model detects 468 keypoints (without iris detection).
The index of key-points mapping can be found at:
https://github.com/tensorflow/tfjs-models/blob/838611c02f51159afdd77469ce67f0e26b7bbb23/face-landmarks-detection/src/mediapipe-facemesh/keypoints.ts

@param img: image to detect facial landmarks.
@return: facial landmarks numpy array.
"""
def run_mediapipe_facial_landmark_detection(img: np.ndarray) -> np.ndarray:
    holistic = mp.solutions.holistic.Holistic(static_image_mode=True, model_complexity=2)
    results = holistic.process((img * 255).astype(np.uint8))
    return results.face_landmarks.landmark


"""
Get face bounding box.

@param img: image to detect facial landmarks.
@return: facial bounding box (left-x, right-x, forehead-y, chin-y) in pixels
"""
def get_face_bounding_box(img: np.ndarray) -> np.ndarray:
    landmarks = run_mediapipe_facial_landmark_detection(img)
    left = landmarks[234].x * img.shape[1]
    right = landmarks[454].x * img.shape[1]
    forehead = landmarks[10].y * img.shape[0]
    chin = landmarks[152].y * img.shape[0]
    return np.array([left, right, forehead, chin], dtype=np.float32)