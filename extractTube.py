from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm
from ultralytics import YOLO


# 3) Extract segmentation-based tubes using YOLOv8-seg + default tracker
def extract_segmentation_tubes(
    frames: list, keep_classes=["person"], min_len=5, conf=0.4
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # print(device)
    print(f"Using device : {device}")
    model = YOLO("yolo11s-seg.pt")
    model.to(device)
    # print(model.device)
    H, W, _ = frames[0].shape
    tubes = defaultdict(
        lambda: {
            "frames": [],
            "masks": [],
            "centroids": [],
            "bboxes": [],
            "cls_ids": [],
        }
    )
    results = []
    for f in tqdm(frames, desc="inference"):
        res = model.track(
            [
                f,
            ],
            device=device,
            stream=True,
            persist=True,
            half=True,
            conf=conf,
            verbose=False,
            tracker="botsort.yaml",
            imgsz=max(W, H),
            retina_masks=True,
        )
        new_res = []
        for r in res:
            if r is None or r.boxes is None or r.masks is None or r.boxes.id is None:
                continue
            ids = r.boxes.id.detach().cpu().numpy()
            cls = r.boxes.cls.detach().cpu().numpy().astype(int)
            masks = r.masks.data.detach().cpu().numpy()  # (N, H, W) floats 0..1
            boxes = r.boxes.xyxy.detach().cpu().numpy().astype(int)
            new_res.append(dict(ids=ids, cls=cls, masks=masks, boxes=boxes))
        results.extend(new_res)
    names = model.names
    # print(names)
    for t, r in tqdm(enumerate(results), desc="Tracking"):
        # if r is None or r.masks is None:
        #     continue
        # if r is None or r.boxes is None or r.masks is None or r.boxes.id is None:
        #     continue
        # ids = r.boxes.id.cpu().numpy()
        # cls = r.boxes.cls.cpu().numpy().astype(int)
        # masks = r.masks.data.cpu().numpy()  # (N, H, W) floats 0..1
        # boxes = r.boxes.xyxy.cpu().numpy().astype(int)
        ids = r["ids"]
        cls = r["cls"]
        masks = r["masks"]
        boxes = r["boxes"]

        for obj_id, cls_id, mask, box in zip(ids, cls, masks, boxes):
            # print("extract tube", mask.shape)
            label = names[int(cls_id)]
            if label not in keep_classes:
                continue
            # mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
            bin_mask = mask > 0.5

            # compute centroid of the binary mask
            ys, xs = np.where(bin_mask)
            if len(xs) == 0:
                continue
            cx, cy = int(xs.mean()), int(ys.mean())

            tubes[obj_id]["frames"].append(t)
            tubes[obj_id]["masks"].append(bin_mask)
            tubes[obj_id]["centroids"].append((cx, cy))
            tubes[obj_id]["bboxes"].append(tuple(box))
            tubes[obj_id]["cls_ids"].append(int(cls_id))
    # filter out short tubes
    out = []
    for tube in tubes.values():
        if len(tube["frames"]) >= min_len:
            out.append(tube)
        else:
            print(f" tube lenth is small than {min_len}")
    print(f"[INFO] Extracted {len(out)} tubes (min_len={min_len})")
    return out, names
