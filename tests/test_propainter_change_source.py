"""
    Testing ProPainter restart on another source.
"""

import os
from propainter.propainter_video import (FilePathDirSequencer, RawFrameSequencer, RawMaskSequencer,
                                         ScaledProPainterIterator, run_streaming_propainter)
from common import check_arrays


def test_propainter_change_source():
    image_resize_ratio = 1.0
    mask_dilation = 4

    root_dir_path = os.path.dirname(__file__)
    frames_dir_path = os.path.join(root_dir_path, "data/bmx-trees")
    masks_dir_path = os.path.join(root_dir_path, "data/bmx-trees_mask")
    result_dir_path = os.path.join(root_dir_path, "data/bmx-trees_result")

    frame_file_sequencer = FilePathDirSequencer(dir_path=frames_dir_path)
    mask_file_sequencer = FilePathDirSequencer(dir_path=masks_dir_path)

    raw_frame_sequencer = RawFrameSequencer(data=frame_file_sequencer)
    raw_mask_sequencer = RawMaskSequencer(data=mask_file_sequencer)

    vi_iterator = ScaledProPainterIterator(
        raw_frames=raw_frame_sequencer,
        raw_masks=raw_mask_sequencer,
        image_resize_ratio=image_resize_ratio,
        mask_dilation=mask_dilation)

    vi_frames_np = run_streaming_propainter(vi_iterator)

    check_arrays(
        gt_arrays_dir_path=result_dir_path,
        pref="vi_frame_",
        tested_array=vi_frames_np,
        start_idx=0,
        end_idx=len(vi_frames_np),
        file_format="png")

    frames_dir_path = os.path.join(root_dir_path, "data/tennis")
    masks_dir_path = os.path.join(root_dir_path, "data/tennis_mask")
    result_dir_path = os.path.join(root_dir_path, "data/tennis_result")

    frame_file_sequencer = FilePathDirSequencer(dir_path=frames_dir_path)
    mask_file_sequencer = FilePathDirSequencer(dir_path=masks_dir_path)

    raw_frame_sequencer = RawFrameSequencer(data=frame_file_sequencer)
    raw_mask_sequencer = RawMaskSequencer(data=mask_file_sequencer)

    vi_iterator = ScaledProPainterIterator(
        raw_frames=raw_frame_sequencer,
        raw_masks=raw_mask_sequencer,
        image_resize_ratio=image_resize_ratio,
        mask_dilation=mask_dilation,
        raft_model=vi_iterator.flow_sequencer.net,
        pprfc_model=vi_iterator.comp_flow_sequencer.net,
        pp_model=vi_iterator.trans_frame_sequencer.net)

    vi_frames = run_streaming_propainter(vi_iterator)

    check_arrays(
        gt_arrays_dir_path=result_dir_path,
        pref="vi_frame_",
        tested_array=vi_frames,
        start_idx=0,
        end_idx=len(vi_frames),
        file_format="png")
