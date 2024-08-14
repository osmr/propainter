"""
    Testing ProPainter restart on long sequence.
"""

from propainter.propainter_video import (ProPainterIterator, FilePathDirIterator, FrameIterator, MaskIterator,
                                         run_streaming_propainter, check_arrays)


def test_propainter_long():
    raft_model_path = None
    pprfc_model_path = None
    pp_model_path = None

    factor = 100

    image_resize_ratio = 1.0
    mask_dilation = 4

    frames_dir_path = "data/bmx-trees"
    masks_dir_path = "data/bmx-trees_mask"
    result_dir_path = "data/bmx-trees_result"

    frame_file_iterator = FilePathDirIterator(frames_dir_path)
    video_length = len(frame_file_iterator)

    frame_file_iterator.file_name_list = frame_file_iterator.file_name_list * factor
    frame_iterator = FrameIterator(
        data=frame_file_iterator,
        image_resize_ratio=image_resize_ratio,
        use_cuda=True)

    mask_file_iterator = FilePathDirIterator(masks_dir_path)
    mask_file_iterator.file_name_list = mask_file_iterator.file_name_list * factor
    mask_iterator = MaskIterator(
        mask_dilation=mask_dilation,
        data=mask_file_iterator,
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
        end_idx=video_length,
        format="png")
