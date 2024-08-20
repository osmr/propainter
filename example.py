"""
    Example of video inpainting based on ProPainter.
"""

import argparse
import numpy as np
from propainter.propainter_video import (ProPainterIterator, FrameIterator, MaskIterator, FilePathDirIterator,
                                         conv_propainter_frames_into_numpy, check_arrays)


def parse_args() -> argparse.Namespace:
    """
    Parse python script parameters.

    Returns
    -------
    argparse.Namespace
        Resulted args.
    """
    parser = argparse.ArgumentParser(
        description="Video inpainting",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--frames",
        type=str,
        default="tests/data/bmx-trees",
        help="path to the directories with input frames")
    parser.add_argument(
        "--masks",
        type=str,
        default="tests/data/bmx-trees_mask",
        help="path to the directories with input masks")
    parser.add_argument(
        "--output",
        type=str,
        default="tests/data/bmx-trees_result",
        help="path to the directories with output frames")
    parser.add_argument(
        "--resize_ratio",
        type=float,
        default=1.0,
        help="resize scale for input frames/masks")
    parser.add_argument(
        "--mask_dilation",
        type=int,
        default=4,
        help="mask dilation for input masks")
    parser.add_argument(
        "--save",
        action="store_true",
        help="save output frames instead of testing")

    args = parser.parse_args()
    return args


def main():
    """
    Main body of script.
    """
    args = parse_args()
    frames_dir_path = args.frames
    masks_dir_path = args.masks
    output_dir_path = args.output
    image_resize_ratio = args.resize_ratio
    mask_dilation = args.mask_dilation
    do_save = args.save

    raft_model_path = None
    pprfc_model_path = None
    pp_model_path = None

    frame_iterator = FrameIterator(
        data=FilePathDirIterator(frames_dir_path),
        image_resize_ratio=image_resize_ratio,
        use_cuda=True)

    mask_iterator = MaskIterator(
        mask_dilation=mask_dilation,
        data=FilePathDirIterator(masks_dir_path),
        image_resize_ratio=image_resize_ratio,
        use_cuda=True)

    vi_iterator = ProPainterIterator(
        frames=frame_iterator,
        masks=mask_iterator,
        raft_model=raft_model_path,
        pprfc_model=pprfc_model_path,
        pp_model=pp_model_path)

    vi_frames_np = None
    for frames_i in vi_iterator:
        frames_np_i = conv_propainter_frames_into_numpy(frames_i)
        if vi_frames_np is None:
            vi_frames_np = frames_np_i
        else:
            vi_frames_np = np.concatenate([vi_frames_np, frames_np_i])

    check_arrays(
        gt_arrays_dir_path=output_dir_path,
        pref="vi_frame_",
        tested_array=vi_frames_np,
        start_idx=0,
        end_idx=len(vi_frames_np),
        do_save=do_save,
        format="png")


if __name__ == "__main__":
    main()
