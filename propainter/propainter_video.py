"""
    Video inpainting calculator based on ProPainter.
"""

__all__ = ['ProPainterIterator', 'FrameIterator', 'MaskIterator', 'FilePathDirIterator',
           'conv_propainter_frames_into_numpy', 'run_streaming_propainter', 'check_arrays']

import os
import cv2
from PIL import Image
import scipy.ndimage
import numpy as np
import torch
from typing import Sequence
from enum import IntEnum
from pytorchcv.models.common.steam import BufferedIterator
from pytorchcv.models.propainter_stream import ProPainterIterator


class FilePathDirIterator(object):
    """
    Iterator for file paths in directory.

    Parameters
    ----------
    dir_path: str
        Directory path.
    """
    def __init__(self,
                 dir_path: str):
        super(FilePathDirIterator, self).__init__()
        assert os.path.exists(dir_path)

        self.dir_path = dir_path
        self.file_name_list = sorted(os.listdir(dir_path))

    def __len__(self):
        return len(self.file_name_list)

    def __getitem__(self,
                    index: int | slice) -> list[str]:
        selected_file_name_list = self.file_name_list[index]
        if isinstance(selected_file_name_list, str):
            return os.path.join(self.dir_path, selected_file_name_list)
        elif isinstance(selected_file_name_list, list):
            return [os.path.join(self.dir_path, x) for x in selected_file_name_list]
        else:
            raise ValueError()


class FrameIterator(BufferedIterator):
    """
    Frame buffered iterator.

    Parameters
    ----------
    image_resize_ratio : float
        Resize ratio.
    use_cuda : bool
        Whether to use CUDA.
    """
    def __init__(self,
                 image_resize_ratio: float,
                 use_cuda: bool,
                 **kwargs):
        super(FrameIterator, self).__init__(**kwargs)
        assert (image_resize_ratio > 0.0)
        self.image_resize_ratio = image_resize_ratio
        self.use_cuda = use_cuda

        self.image_scaled_size = None
        self.do_scale = False

    def _rescale_image(self,
                       image: Image,
                       resample: IntEnum | None = None) -> Image:
        """
        Rescale frame.

        Parameters
        ----------
        image : Image
            Frame.
        resample : IntEnum or None, default None
            PIL resample mode.

        Returns
        -------
        Image
            Image.
        """
        if self.image_scaled_size is None:
            image_raw_size = image.size
            self.image_scaled_size = (int(self.image_resize_ratio * image_raw_size[0]),
                                      int(self.image_resize_ratio * image_raw_size[1]))
            self.image_scaled_size = (self.image_scaled_size[0] - self.image_scaled_size[0] % 8,
                                      self.image_scaled_size[1] - self.image_scaled_size[1] % 8)
            if image_raw_size != self.image_scaled_size:
                self.do_scale = True
        if self.do_scale:
            image = image.resize(
                size=self.image_scaled_size,
                resample=resample)
        return image

    def load_frame(self,
                   frame_path: str) -> Image:
        """
        Load frame from file.

        Parameters
        ----------
        frame_path : str
            Path to frame file.

        Returns
        -------
        Image
            Frame.
        """
        frame = cv2.imread(frame_path)
        frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame = self._rescale_image(image=frame)
        return frame

    def _calc_data_items(self,
                         raw_data_chunk_list: tuple[Sequence, ...] | list[Sequence] | Sequence) -> Sequence:
        """
        Calculate/load data items.

        Parameters
        ----------
        raw_data_chunk_list : tuple(sequence, ...) or list(sequence) or sequence
            List of source data chunks.

        Returns
        -------
        sequence
            Resulted data.
        """
        assert (len(raw_data_chunk_list) == 1)

        frame_list = [self.load_frame(x) for x in raw_data_chunk_list[0]]
        frames = np.stack(frame_list)
        frames = torch.from_numpy(frames).permute(0, 3, 1, 2).contiguous()
        frames = frames.float()
        frames = frames.div(255.0)
        frames = frames * 2.0 - 1.0

        if self.use_cuda:
            frames = frames.cuda()

        return frames

    def _expand_buffer_by(self,
                          data_chunk: Sequence):
        """
        Expand buffer by extra data.

        Parameters
        ----------
        data_chunk : sequence
            Data chunk.
        """
        self.buffer = torch.cat([self.buffer, data_chunk])


class MaskIterator(FrameIterator):
    """
    Mask buffered iterator.

    Parameters
    ----------
    mask_dilation : int
        Mask dilation.
    image_resize_ratio : float
        Resize ratio.
    use_cuda : bool
        Whether to use CUDA.
    """
    def __init__(self,
                 mask_dilation: int,
                 **kwargs):
        super(MaskIterator, self).__init__(**kwargs)
        self.mask_dilation = mask_dilation
        assert (self.mask_dilation > 0)

    def load_mask(self,
                  mask_path: str) -> Image:
        """
        Load mask from file.

        Parameters
        ----------
        mask_path : str
            Path to mask file.

        Returns
        -------
        Image
            Mask.
        """
        mask = Image.open(mask_path)
        mask = self._rescale_image(image=mask, resample=Image.NEAREST)
        mask = np.array(mask.convert("L"))

        mask = scipy.ndimage.binary_dilation(input=mask, iterations=self.mask_dilation).astype(np.uint8)
        mask = Image.fromarray(mask * 255)

        return mask

    def _calc_data_items(self,
                         raw_data_chunk_list: tuple[Sequence, ...] | list[Sequence] | Sequence) -> Sequence:
        """
        Calculate/load data items.

        Parameters
        ----------
        raw_data_chunk_list : tuple(sequence, ...) or list(sequence) or sequence
            List of source data chunks.

        Returns
        -------
        sequence
            Resulted data.
        """
        assert (len(raw_data_chunk_list) == 1)

        mask_list = [self.load_mask(x) for x in raw_data_chunk_list[0]]
        masks = np.stack(mask_list)
        masks = np.expand_dims(masks, axis=-1)
        masks = torch.from_numpy(masks).permute(0, 3, 1, 2).contiguous()
        masks = masks.float()
        masks = masks.div(255.0)

        if self.use_cuda:
            masks = masks.cuda()

        return masks


def conv_propainter_frames_into_numpy(frames: torch.Tensor) -> np.ndarray:
    """
    Convert ProPainter output frames from torch to numpy format.

    Parameters
    ----------
    frames : torch.Tensor
        ProPainter output frames in torch format.

    Returns
    -------
    np.ndarray
        Resulted numpy frames.
    """
    frames = (((frames + 1.0) / 2.0) * 255).to(torch.uint8)
    frames = frames.permute(0, 2, 3, 1).cpu().detach().numpy()
    return frames


def run_streaming_propainter(vi_iterator: ProPainterIterator) -> np.ndarray:
    """
    Run ProPainter in streaming mode.

    Parameters
    ----------
    vi_iterator : ProPainterIterator
        ProPainter iterator.

    Returns
    -------
    np.ndarray
        Resulted frames.
    """
    vi_frames_np = None
    for frames_i in vi_iterator:
        frames_np_i = conv_propainter_frames_into_numpy(frames_i)
        if vi_frames_np is None:
            vi_frames_np = frames_np_i
        else:
            vi_frames_np = np.concatenate([vi_frames_np, frames_np_i])
    return vi_frames_np


def check_arrays(gt_arrays_dir_path: str,
                 pref: str,
                 tested_array: torch.Tensor | np.ndarray,
                 start_idx: int,
                 end_idx: int,
                 c_slice: slice = slice(None),
                 do_save: bool = False,
                 precise: bool = True,
                 atol: float = 1.0,
                 format: str = "npy"):
    """
    Check calculation precision by saved values.

    Parameters
    ----------
    gt_arrays_dir_path : str
        Directory path for saved values.
    pref : str
        Prefix for file name.
    tested_array : torch.Tensor or np.ndarray
        Tested array of values.
    start_idx : int
        Start index for test.
    end_idx : int
        End index for test.
    c_slice : slice, default slice(None)
        Slice value for the second dim.
    do_save : bool, default False
        Whether to save value instead to test.
    precise : bool, default True
        Whether to do precise testing.
    atol : float, default 1.0
        Absolute tolerance value.
    format : str, default 'npy'
        File extension (`npy` or `png`).

    Returns
    -------
    np.ndarray
        Resulted numpy frames.
    """
    if do_save and (not os.path.exists(gt_arrays_dir_path)):
        os.mkdir(gt_arrays_dir_path)

    for j, i in enumerate(range(start_idx, end_idx)):
        if isinstance(tested_array, torch.Tensor):
            tested_array_i = tested_array[j, c_slice].cpu().detach().numpy()
        else:
            tested_array_i = tested_array[j]
        tested_array_i = np.ascontiguousarray(tested_array_i)

        tested_array_i_file_path = os.path.join(gt_arrays_dir_path, pref + "{:05d}.{}".format(i, format))
        if do_save:
            if format == "npy":
                np.save(tested_array_i_file_path, tested_array_i)
            else:
                tested_array_i = cv2.cvtColor(tested_array_i, cv2.COLOR_BGR2RGB)
                cv2.imwrite(tested_array_i_file_path, tested_array_i)
            continue

        if format == "npy":
            gt_array_i = np.load(tested_array_i_file_path)
        else:
            gt_array_i = cv2.imread(tested_array_i_file_path)
            gt_array_i = cv2.cvtColor(gt_array_i, cv2.COLOR_BGR2RGB)

        if precise:
            if not np.array_equal(tested_array_i, gt_array_i):
                print(f"{gt_arrays_dir_path}, {pref}, {tested_array}, {start_idx}, {end_idx}, {j}, {i}")
            np.testing.assert_array_equal(tested_array_i, gt_array_i)
        else:
            if not np.allclose(tested_array_i, gt_array_i, rtol=0, atol=atol):
                print(f"{gt_arrays_dir_path}, {pref}, {tested_array}, {start_idx}, {end_idx}, {j}, {i}")
            np.testing.assert_allclose(tested_array_i, gt_array_i, rtol=0, atol=atol)
