import cv2
import numpy as np


# 6) Build & write one synopsis video per class ID
def build_synopsis_with_time(
    frames, tubes, shifts, synopsis_length, background, alpha_border=20
):
    """
    Build the final synopsis video frames:
      - Soft alpha‐blend each tube’s mask over the background
      - Draw a green bbox around each object
      - Put the original source‐frame index (t_orig) above the box

    Args:
      frames           : list of original BGR frames [(H,W,3), ...]
      tubes            : list of tube‐dicts, each with keys:
                          "frames", "masks", "bboxes"
      shifts           : list of int, start‐time for each tube
      synopsis_length  : total number of synopsis frames to produce
      background       : (H,W,3) uint8 background image
      alpha_border     : width (px) over which to fade mask edges

    Returns:
      synopsis_frames  : list of (H,W,3) uint8 frames
    """
    H, W = background.shape[:2]

    # 1) precompute soft alpha masks for each tube
    soft_masks = []
    for tube in tubes:
        tube_soft = []
        for mask in tube["masks"]:
            # distance to background pixels
            dist = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 5)
            # fade over alpha_border px
            alpha = 1.0 - np.clip(dist, 0, alpha_border) / alpha_border
            tube_soft.append(alpha)
        soft_masks.append(tube_soft)

    # 2) initialize blank synopsis frames with background
    synopsis = [background.copy() for _ in range(synopsis_length)]

    # 3) composite each tube
    for tube, soft_tube, shift in zip(tubes, soft_masks, shifts):
        for idx, t_orig in enumerate(tube["frames"]):
            t_syn = shift + idx
            if not (0 <= t_syn < synopsis_length):
                continue

            mask = tube["masks"][idx]  # bool (H,W)
            alpha_map = soft_tube[idx]  # float (H,W)
            box = tube["bboxes"][idx]  # (x1,y1,x2,y2)

            # blend FG over BG
            fg = frames[t_orig].astype(np.float32) / 255.0
            bg = synopsis[t_syn].astype(np.float32) / 255.0
            A = alpha_map[:, :, None]
            comp = fg * A + bg * (1 - A)
            comp_uint8 = (comp * 255).astype(np.uint8)

            # apply only where mask==True
            synopsis[t_syn][mask] = comp_uint8[mask]

            # draw bounding box
            x1, y1, x2, y2 = box
            color = (255, 255, 0)
            cv2.rectangle(synopsis[t_syn], (x1, y1), (x2, y2), color, 2)

            # draw the original frame index
            text = f"{t_orig}"
            cv2.putText(
                synopsis[t_syn],
                text,
                (x1, max(y1 - 10, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
                cv2.LINE_AA,
            )

    return synopsis


#
def write_video(frames, out_path, fps=10):
    H, W, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    w = cv2.VideoWriter(out_path, fourcc, fps, (W, H))
    for f in frames:
        w.write(f)
    w.release()
