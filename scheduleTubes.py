from collections import defaultdict

import numpy as np


# 5) Greedy schedule tubes into a short synopsis
def schedule_tubes_dynamic(tubes, H, W):
    shifts = [None] * len(tubes)
    occupancy = []
    for i, tube in enumerate(tubes):
        L = len(tube["frames"])
        placed = False
        # try all existing start positions
        for s in range(max(0, len(occupancy) - L) + 1):
            if any(
                (tube["masks"][j] & occupancy[s + j]).any()
                for j in range(L)
                if s + j < len(occupancy)
            ):
                continue
            shifts[i] = s
            # extend occupancy
            while len(occupancy) < s + L:
                occupancy.append(np.zeros((H, W), dtype=bool))
            for j, m in enumerate(tube["masks"]):
                occupancy[s + j] |= m
            placed = True
            break
        if not placed:
            s = len(occupancy)
            shifts[i] = s
            for _ in range(L):
                occupancy.append(np.zeros((H, W), dtype=bool))
            for j, m in enumerate(tube["masks"]):
                occupancy[s + j] |= m
    return shifts, len(occupancy)


def groub_tubes_by_classid(all_tubes: list):
    """list of tubes
    reurn dict containing tube by class ids
    """
    class_groups = defaultdict(list)
    for tube in all_tubes:
        cid = tube["cls_ids"][0]
        class_groups[cid].append(tube)
    return class_groups
