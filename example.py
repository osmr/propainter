"""
    Example of video inpainting based on ProPainter.
"""

import os
import argparse
import numpy as np
from propainter.propainter_video import (ProPainterIterator, FrameIterator, MaskIterator, FilePathDirIterator,
                                         conv_propainter_frames_into_numpy, check_arrays)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', '--video', type=str, default='inputs/object_removal/bmx-trees', help='Path of the input video or image folder.')
    parser.add_argument(
        '-m', '--mask', type=str, default='inputs/object_removal/bmx-trees_mask', help='Path of the mask(s) or mask folder.')
    parser.add_argument(
        '-o', '--output', type=str, default='results', help='Output folder. Default: results')
    parser.add_argument(
        "--resize_ratio", type=float, default=1.0, help='Resize scale for processing video.')
    parser.add_argument(
        '--height', type=int, default=-1, help='Height of the processing video.')
    parser.add_argument(
        '--width', type=int, default=-1, help='Width of the processing video.')
    parser.add_argument(
        '--mask_dilation', type=int, default=4, help='Mask dilation for video and flow masking.')
    parser.add_argument(
        "--ref_stride", type=int, default=10, help='Stride of global reference frames.')
    parser.add_argument(
        "--neighbor_length", type=int, default=10, help='Length of local neighboring frames.')
    parser.add_argument(
        "--subvideo_length", type=int, default=80, help='Length of sub-video for long video inference.')
    parser.add_argument(
        "--raft_iter", type=int, default=20, help='Iterations for RAFT inference.')
    parser.add_argument(
        '--mode', default='video_inpainting', choices=['video_inpainting', 'video_outpainting'], help="Modes: video_inpainting / video_outpainting")
    parser.add_argument(
        '--scale_h', type=float, default=1.0, help='Outpainting scale of height for video_outpainting mode.')
    parser.add_argument(
        '--scale_w', type=float, default=1.2, help='Outpainting scale of width for video_outpainting mode.')
    parser.add_argument(
        '--save_fps', type=int, default=24, help='Frame per second. Default: 24')
    parser.add_argument(
        '--save_frames', action='store_true', help='Save output frames. Default: False')

    args = parser.parse_args()

    frames_dir_path = args.video
    masks_dir_path = args.mask
    image_resize_ratio = args.resize_ratio
    mask_dilation = args.mask_dilation

    raft_model_path = "pytorchcv_data/test/raft-things_2.pth"
    pprfc_model_path = "pytorchcv_data/test/propainter_rfc.pth"
    pp_model_path = "pytorchcv_data/test/propainter.pth"

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
        raft_model_path=raft_model_path,
        pprfc_model_path=pprfc_model_path,
        pp_model_path=pp_model_path)

    vi_frames_np = None
    for frames_i in vi_iterator:
        frames_np_i = conv_propainter_frames_into_numpy(frames_i)
        if vi_frames_np is None:
            vi_frames_np = frames_np_i
        else:
            vi_frames_np = np.concatenate([vi_frames_np, frames_np_i])

    video_length = len(vi_frames_np)

    if True:
        save_root = os.path.join(args.output, "results")
        check_arrays(
            gt_arrays_dir_path=os.path.join(save_root, "comp_frames"),
            pref="comp_frame_",
            tested_array=vi_frames_np,
            start_idx=0,
            end_idx=video_length,
            # do_save=True,
            precise=False,
            # atol=8,
        )

        import imageio
        imageio.mimwrite(
            uri=os.path.join(save_root, "inpaint_out_2.mp4"),
            ims=vi_frames_np,
            fps=args.save_fps,
            quality=7)

    pass
