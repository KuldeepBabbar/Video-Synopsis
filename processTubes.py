import os

import cv2
import numpy as np


def bbox_iou(boxA, boxB):
    """
    Compute IoU between two axis‑aligned boxes.
    box = (x1,y1,x2,y2)
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    inter = interW * interH

    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union = areaA + areaB - inter
    return inter / union if union > 0 else 0.0


def merge_bike_into_person_masks(tubes, person_cls_id=0, bike_cls_id=3, iou_thresh=0.1):
    """
    For each person tube, whenever a bike tube shares a frame and
    their bboxes overlap by >= iou_thresh, union the bike mask
    into the person mask for that frame.

    Args:
      tubes          : list of tube dicts with keys
                       'frames','masks','bboxes','cls_ids'
      person_cls_id  : the class ID for person in tubes
      bike_cls_id    : the class ID for bike in tubes
      iou_thresh     : bounding-box IoU threshold to trigger merge

    Returns:
      new_tubes      : list of tubes where person tubes have been
                       augmented with overlapping bike masks
    """

    def bbox_iou(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interW = max(0, xB - xA)
        interH = max(0, yB - yA)
        inter = interW * interH
        areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        union = areaA + areaB - inter
        return inter / union if union > 0 else 0.0

    # Separate person and bike tubes
    person_tubes = [t for t in tubes if t["cls_ids"][0] == person_cls_id]
    bike_tubes = [t for t in tubes if t["cls_ids"][0] == bike_cls_id]
    others = [t for t in tubes if t["cls_ids"][0] not in (person_cls_id, bike_cls_id)]

    # For each person tube, augment its masks
    for P in person_tubes:
        # Build a quick lookup for P: frame -> index in P["frames"]
        idxP = {fr: i for i, fr in enumerate(P["frames"])}
        for B in bike_tubes:
            # Build same for B
            idxB = {fr: i for i, fr in enumerate(B["frames"])}
            # find common frames
            for fr in set(idxP) & set(idxB):
                iP = idxP[fr]
                iB = idxB[fr]
                boxP = P["bboxes"][iP]
                boxB = B["bboxes"][iB]
                if bbox_iou(boxP, boxB) >= iou_thresh:
                    # Union the bike mask into the person mask
                    P["masks"][iP] = P["masks"][iP] | B["masks"][iB]
                    # (Optionally) expand the bbox to cover both:
                    x1p, y1p, x2p, y2p = boxP
                    x1b, y1b, x2b, y2b = boxB
                    P["bboxes"][iP] = (
                        min(x1p, x1b),
                        min(y1p, y1b),
                        max(x2p, x2b),
                        max(y2p, y2b),
                    )
    # Return combined list: modified person tubes + all others (including original bike tubes)
    return person_tubes + bike_tubes + others


def refine_tubes_by_bbox_disp(tubes, min_disp=5, min_frames=1):
    """
    Subsample each tube so that consecutive kept frames have their
    bounding-box center moved at least max_disp pixels.

    Args:
      tubes    : list of tube dicts, each with keys:
                   - "frames":   list of int frame indices
                   - "bboxes":   list of (x1,y1,x2,y2) tuples
                   - "masks":    list of (H,W) bool masks
                   - "cls_ids":  list of int class IDs
      min_disp : minimum Euclidean displacement (in pixels) required
                 before keeping a new frame in the tube.

    Returns:
      refined : new list of tubes in the same format, but with each
                tube subsampled by displacement.
    """
    refined = []
    for tube in tubes:
        frs = tube["frames"]
        bbs = tube["bboxes"]
        # msks = tube["masks"]
        # cls_ids = tube.get("cls_ids", None)

        last_cx = last_cy = None
        new_tube = {k: [] for k in tube}

        for idx, t in enumerate(frs):
            x1, y1, x2, y2 = bbs[idx]
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0

            # Always keep the very first detection
            if last_cx is None or np.hypot(cx - last_cx, cy - last_cy) >= min_disp:
                # append this index to the new tube
                for key in tube:
                    new_tube[key].append(tube[key][idx])
                last_cx, last_cy = cx, cy

        # only include tubes whic has moving object lenth
        if len(new_tube["frames"]) > min_frames:
            refined.append(new_tube)

    return refined


# save tubes for debugging
def save_tubes_as_videos(frames, tubes, out_dir="tubes", fps=10):
    """
    For each tube, create a video of just that tube’s appearances.

    Arguments:
      frames : list of (H, W, 3) uint8 original frames
      tubes  : list of dicts, each with:
                 - "frames": list of frame indices
                 - "masks" : list of (H, W) bool masks
      out_dir : directory to write tube_0.mp4, tube_1.mp4, …
      fps     : output frames per second
    """
    os.makedirs(out_dir, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    H, W, _ = frames[0].shape
    for i, tube in enumerate(tubes):
        tube_len = len(tube["frames"])
        if tube_len == 0:
            continue

        path = os.path.join(out_dir, f"tube_{i}.mp4")
        writer = cv2.VideoWriter(path, fourcc, fps, (W, H), isColor=True)

        for mask, t_idx in zip(tube["masks"], tube["frames"]):

            # start with a black frame
            out_frame = np.zeros_like(frames[0])
            # print(frames[0].shape, mask.shape)
            # copy only the object pixels
            out_frame[mask] = frames[t_idx][mask]
            writer.write(out_frame)

        writer.release()
        print(f"Saved tube {i} ({tube_len} frames) → {path}")


# 4) Filter tubes by a single class ID
def filter_tubes_by_class(tubes, class_id):
    return [tube for tube in tubes if tube["cls_ids"][0] == class_id]
