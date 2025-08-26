import numpy as np
import cv2


# median
def compute_background_median(frames):
    arr = np.stack(frames, axis=3)  # shape: (H, W, 3, T)
    background = np.median(arr, axis=3).astype(np.uint8)
    return background  # shape: (H, W, 3)


def extract_background_mog2(
    video_path,
    history=500,
    varThreshold=16,
    detectShadows=True,
    learning_rate=0.05,
    width=640,
    height=380,
):
    """
    Extracts the background image from a video using MOG2 with a specified learning rate.

    Parameters:
    -----------
    video_path : str
        Path to the input video file.
    history : int
        Length of the history (number of frames) for MOG2 (default: 500).
    varThreshold : float
        Threshold on the squared Mahalanobis distance to decide
        whether a pixel is well described by the background model (default: 16).
    detectShadows : bool
        Whether to enable shadow detection (default: True).
    learning_rate : float
        The learning rate for background model updates.
        Use values > 1/history to make it adapt faster (default: 0.05).

    Returns:
    --------
    background : numpy.ndarray or None
        The final background image, or None if the video couldn’t be processed.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return None

    # Initialize MOG2 subtractor
    backSub = cv2.createBackgroundSubtractorMOG2(
        history=history, varThreshold=varThreshold, detectShadows=detectShadows
    )

    # Process all frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Apply with explicit learning rate
        frame = cv2.resize(
            frame,
            (width, height),
            interpolation=cv2.INTER_AREA,
        )
        backSub.apply(frame, learningRate=learning_rate)

    cap.release()
    # Retrieve the background image
    background = backSub.getBackgroundImage()
    return background


# 2) Compute a robust background via MOG2
def compute_background(frames, history=100, varThresh=100):
    bg_sub = cv2.createBackgroundSubtractorMOG2(
        history=history,
        varThreshold=varThresh,
        detectShadows=False,
        learningRate=1,
    )
    for f in frames:
        bg_sub.apply(f)
    bg = bg_sub.getBackgroundImage()
    if bg is None:
        arr = np.stack(frames, axis=3)
        bg = np.median(arr, axis=3).astype(np.uint8)
    return bg
