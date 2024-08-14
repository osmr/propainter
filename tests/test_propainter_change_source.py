"""
    Testing ProPainter restart on another source.
"""

from propainter.propainter_video import (ProPainterIterator, FilePathDirIterator, FrameIterator, MaskIterator,
                                         run_streaming_propainter, check_arrays)


def test_propainter_change_source():
    raft_model_path = None
    pprfc_model_path = None
    pp_model_path = None

    image_resize_ratio = 1.0
    mask_dilation = 4

    frames_dir_path = "data/bmx-trees"
    masks_dir_path = "data/bmx-trees_mask"
    result_dir_path = "data/bmx-trees_result"

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

    vi_frames_np = run_streaming_propainter(vi_iterator)

    check_arrays(
        gt_arrays_dir_path=result_dir_path,
        pref="vi_frame_",
        tested_array=vi_frames_np,
        start_idx=0,
        end_idx=len(vi_frames_np),
        format="png")

    frames_dir_path = "data/tennis"
    masks_dir_path = "data/tennis_mask"
    result_dir_path = "data/tennis_result"

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
        raft_model=vi_iterator.flow_iterator.net,
        pprfc_model=vi_iterator.comp_flow_iterator.net,
        pp_model=vi_iterator.trans_frame_iterator.net)

    vi_frames_np = run_streaming_propainter(vi_iterator)

    check_arrays(
        gt_arrays_dir_path=result_dir_path,
        pref="vi_frame_",
        tested_array=vi_frames_np,
        start_idx=0,
        end_idx=len(vi_frames_np),
        format="png")
