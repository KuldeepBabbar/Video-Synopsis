from processTubesY import *
from scheduleTubesY import *
from write import *

from computeBackground import *
from loadVideo import *

# import from every module
# 7) Main
if __name__ == "__main__":
    video_path = "input_video/video_1_20250324_174822.avi"

    # parameters
    max_frames = 1000
    keep_classes = ["car"]

    min_len = 5
    conf = 0
    fps_out = 10

    frames = load_video_color(video_path, max_frames)

    # sample frame to get clean background using median
    bk_step = 5
    background_frames = [frames[i] for i in np.arange(0, len(frames), bk_step)]
    H, W, _ = frames[0].shape
    print(f"Loaded {len(frames)} resize frames ({W}×{H})")
    background = compute_background_median(background_frames)

    all_tubes, names = extract_segmentation_tubes(
        frames,
        keep_classes,
        min_len=min_len,
        conf=conf,
    )

    # save_tubes_as_videos(frames, all_tubes, "actual_tubes")
    sampled_tubes = refine_tubes_by_bbox_disp(all_tubes)

    # print(names)
    class_groups = groub_tubes_by_classid(sampled_tubes)
    names[-1] = "all_classess"
    class_groups[-1] = sampled_tubes

    for cid, tubes in class_groups.items():
        cls_name = names[int(cid)]
        output_path = f"segmented_synopsis_cls_{cls_name}.mp4"
        # save_tubes_as_videos(
        #     frames,
        #     tubes,
        #     out_dir=f"tubes_output_{cls_name}",
        #     fps=10,
        # )
        tube_lengths = [len(t["frames"]) for t in tubes]

        order = sorted(range(len(tubes)), key=lambda i: -tube_lengths[i])
        # Reorder
        tubes_sorted = [tubes[i] for i in order]

        # Schedule
        shifts = [None] * len(tubes)
        for new_idx, orig_idx in enumerate(order):
            shifts[orig_idx] = shifts_sorted[new_idx]
        # 4) Schedule
        # shifts, syn_len = schedule_tubes_dynamic(tubes, H, W) if you want to use non sorted tubes,

        print(f"Scheduled synopsis length = {syn_len} frames")

        # 5) Build with blending
        synopsis = build_synopsis_with_time(
            frames,
            tubes,
            shifts,
            syn_len,
            background,
        )
        # 6) Write out
        write_video(synopsis, output_path, fps_out)

        print(f"Saved segmented synopsis to {output_path}")
