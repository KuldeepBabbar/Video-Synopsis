# 1) Load up to N frames from the input video
import cv2
import numpy as np
from tqdm import tqdm


def load_video_color(path, max_frames=200, width=640, height=380):
    cap = cv2.VideoCapture(path)
    frames = []
    for _ in tqdm(range(max_frames), desc="loading video"):
        ret, f = cap.read()
        if not ret:
            break

        f = cv2.resize(f, (width, height), interpolation=cv2.INTER_AREA)
        frames.append(f)
    cap.release()
    return frames


def load_background(path, max_frames=10000, width=640, height=380):
    cap = cv2.VideoCapture(path)
    frames = []
    for _ in tqdm(np.arange(0, max_frames, 500), desc="loading video"):
        ret, f = cap.read()
        if not ret:
            break

        f = cv2.resize(f, (width, height), interpolation=cv2.INTER_AREA)
        frames.append(f)
    cap.release()
    return frames


# rezise video
def resize_frames(frames, width=640, height=380):
    """
    Resize a list of image frames to the given low resolution.

    Parameters:
    - frames (list of np.ndarray): List of image frames.
    - width (int): Target width of resized frames.
    - height (int): Target height of resized frames.

    Returns:
    - List of resized image frames (np.ndarray).
    """
    resized = [
        cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        for frame in frames
    ]
    return resized
