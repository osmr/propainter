"""
    Example of video inpainting based on ProPainter.
"""

import argparse
import numpy as np
from propainter.propainter_video import (FilePathDirSequencer, RawFrameSequencer, RawMaskSequencer,
                                         ScaledProPainterIterator)
from tests.common import check_arrays


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
        help="mask dilation for processed masks")
    parser.add_argument(
        "--pre_raw_mask_dilation",
        type=int,
        default=0,
        help="mask dilation for input masks in original resolution (before all calculations)")
    parser.add_argument(
        "--post_raw_mask_dilation",
        type=int,
        default=0,
        help="mask dilation for input masks in original resolution (extra dilation)")
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
    pre_raw_mask_dilation = args.pre_raw_mask_dilation
    post_raw_mask_dilation = args.post_raw_mask_dilation
    do_save = args.save

    frame_file_sequencer = FilePathDirSequencer(dir_path=frames_dir_path)
    mask_file_sequencer = FilePathDirSequencer(dir_path=masks_dir_path)

    raw_frame_sequencer = RawFrameSequencer(data=frame_file_sequencer)
    raw_mask_sequencer = RawMaskSequencer(
        data=mask_file_sequencer,
        pre_raw_mask_dilation=pre_raw_mask_dilation)

    vi_iterator = ScaledProPainterIterator(
        raw_frames=raw_frame_sequencer,
        raw_masks=raw_mask_sequencer,
        image_resize_ratio=image_resize_ratio,
        mask_dilation=mask_dilation,
        post_raw_mask_dilation=post_raw_mask_dilation)

    vi_frames = None
    for frames_i in vi_iterator:
        if vi_frames is None:
            vi_frames = frames_i
        else:
            vi_frames = np.concatenate([vi_frames, frames_i])

    check_arrays(
        gt_arrays_dir_path=output_dir_path,
        pref="vi_frame_",
        tested_array=vi_frames,
        start_idx=0,
        end_idx=len(vi_frames),
        do_save=do_save,
        file_format="png")


if __name__ == "__main__":
    main()
