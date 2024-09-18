"""
    Video inpainting calculator based on ProPainter.
"""

__all__ = ['ScaledProPainterIterator', 'RawFrameSequencer', 'RawMaskSequencer', 'FilePathDirSequencer',
           'run_streaming_propainter']

import os
import cv2
from PIL import Image
import scipy.ndimage
import numpy as np
import torch
from typing import Sequence
from pytorchcv.models.common.stream import Sequencer, BufferedSequencer
from pytorchcv.models.propainter_stream import ProPainterIterator


class PillowImageRescaler:
    """
    Pillow image rescaler.

    Parameters
    ----------
    image_resize_ratio : float
        Resize ratio.
    """
    def __init__(self,
                 image_resize_ratio: float):
        super(PillowImageRescaler, self).__init__()
        assert (image_resize_ratio > 0.0)
        self.image_resize_ratio = image_resize_ratio

        self.image_raw_size = None
        self.image_scaled_size = None
        self.do_scale = False

    def check_image_scale(self,
                          image: np.ndarray):
        """
        Check image scale.

        Parameters
        ----------
        image : np.ndarray
            Source image.
        """
        if self.image_raw_size is None:
            height, width = image.shape[:2]
            self.image_raw_size = (width, height)
            self.image_scaled_size = (int(self.image_resize_ratio * self.image_raw_size[0]),
                                      int(self.image_resize_ratio * self.image_raw_size[1]))
            self.image_scaled_size = (self.image_scaled_size[0] - self.image_scaled_size[0] % 8,
                                      self.image_scaled_size[1] - self.image_scaled_size[1] % 8)
            if self.image_raw_size != self.image_scaled_size:
                self.do_scale = True

    def __call__(self,
                 image: np.ndarray,
                 is_mask: bool) -> np.ndarray:
        """
        Rescale image.

        Parameters
        ----------
        image : np.ndarray
            Source image.
        is_mask : bool
            Whether to interpret processed image as a mask.

        Returns
        -------
        np.ndarray
            Target image.
        """
        self.check_image_scale(image)
        if self.do_scale:
            assert (not is_mask) or (len(image.shape) == 2)
            image = Image.fromarray(image)
            image = image.resize(
                size=self.image_scaled_size,
                resample=(Image.Resampling.NEAREST if is_mask else Image.Resampling.BICUBIC))
            image = np.array(image)
        return image

    def invert(self,
               image: np.ndarray) -> np.ndarray:
        """
        Invert rescale image.

        Parameters
        ----------
        image : np.ndarray
            Source image.

        Returns
        -------
        np.ndarray
            Target image.
        """
        if self.do_scale:
            assert (len(image.shape) == 3)
            image = Image.fromarray(image)
            image = image.resize(
                size=self.image_raw_size,
                resample=Image.Resampling.BICUBIC)
            image = np.array(image)
        return image


class ScipyMaskDilator:
    """
    Scipy binary mask dilator.

    Parameters
    ----------
    dilation : int
        Mask dilation.
    """
    def __init__(self,
                 dilation: int):
        super(ScipyMaskDilator, self).__init__()
        assert (dilation >= 0)
        self.dilation = dilation

    def __call__(self,
                 mask: np.ndarray) -> np.ndarray:
        """
        Dilate mask.

        Parameters
        ----------
        mask : np.ndarray
            Source mask.

        Returns
        -------
        np.ndarray
            Target mask (binary image).
        """
        if self.dilation > 0:
            mask = scipy.ndimage.binary_dilation(input=mask, iterations=self.dilation).astype(np.uint8)
        else:
            assert (mask.dtype == np.uint8)
            assert (np.max(mask).item() <= 1)
        return mask


class FilePathDirSequencer(object):
    """
    Sequencer for file paths in directory.

    Parameters
    ----------
    dir_path: str
        Directory path.
    """
    def __init__(self,
                 dir_path: str):
        super(FilePathDirSequencer, self).__init__()
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


class RawFrameSequencer(BufferedSequencer):
    """
    Numpy raw frame buffered sequencer.
    """
    def __init__(self,
                 **kwargs):
        super(RawFrameSequencer, self).__init__(**kwargs)

    def load_frame(self,
                   image_path: str) -> np.ndarray:
        """
        Load frame from file.

        Parameters
        ----------
        image_path : str
            Path to frame file.

        Returns
        -------
        np.ndarray
            Loaded frame.
        """
        image = cv2.imread(
            filename=image_path,
            flags=cv2.IMREAD_UNCHANGED)
        assert (len(image.shape) == 3)
        assert (image.shape[2] == 3)

        image = cv2.cvtColor(
            src=image,
            code=cv2.COLOR_BGR2RGB)

        assert (image.dtype == np.uint8)
        return image

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
        frames = np.array([self.load_frame(x) for x in raw_data_chunk_list[0]])
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
        self.buffer = np.concatenate([self.buffer, data_chunk])


class RawMaskSequencer(BufferedSequencer):
    """
    Numpy raw mask buffered sequencer.

    Parameters
    ----------
    pre_raw_mask_dilation : int, default 0
        Mask dilation.
    """
    def __init__(self,
                 pre_raw_mask_dilation: int = 0,
                 **kwargs):
        super(RawMaskSequencer, self).__init__(**kwargs)
        self.pre_raw_dilator = ScipyMaskDilator(dilation=pre_raw_mask_dilation)

    def load_mask(self,
                  image_path: str) -> np.ndarray:
        """
        Load mask from file.

        Parameters
        ----------
        image_path : str
            Path to mask file.

        Returns
        -------
        np.ndarray
            Loaded mask.
        """
        image = cv2.imread(
            filename=image_path,
            flags=cv2.IMREAD_UNCHANGED)
        assert (len(image.shape) == 2)

        image = (image > 0).astype(np.uint8)
        assert (np.max(image).item() <= 1)

        image = self.pre_raw_dilator(image)

        assert (image.dtype == np.uint8)
        return image

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
        masks = np.array([self.load_mask(x) for x in raw_data_chunk_list[0]])
        return masks

    def _expand_buffer_by(self,
                          data_chunk: Sequence):
        """
        Expand buffer by extra data.

        Parameters
        ----------
        data_chunk : sequence
            Data chunk.
        """
        self.buffer = np.concatenate([self.buffer, data_chunk])


class FrameSequencer(BufferedSequencer):
    """
    Frame buffered sequencer.

    Parameters
    ----------
    image_resize_ratio : float
        Resize ratio.
    use_cuda : bool, default True
        Whether to use CUDA.
    """
    def __init__(self,
                 image_resize_ratio: float,
                 use_cuda: bool = True,
                 **kwargs):
        super(FrameSequencer, self).__init__(**kwargs)
        self.rescaler = PillowImageRescaler(image_resize_ratio=image_resize_ratio)
        self.use_cuda = use_cuda

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
        frames = raw_data_chunk_list[0]
        assert (len(frames.shape) == 4) and ((frames.shape[-1] == 3))

        frames = np.array([self.rescaler(x, is_mask=False) for x in frames])

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


class MaskSequencer(BufferedSequencer):
    """
    Binary mask buffered sequencer.

    Parameters
    ----------
    image_resize_ratio : float
        Resize ratio.
    mask_dilation : int
        Mask dilation.
    use_cuda : bool, default True
        Whether to use CUDA.
    """
    def __init__(self,
                 image_resize_ratio: float,
                 mask_dilation: int,
                 use_cuda: bool = True,
                 **kwargs):
        super(MaskSequencer, self).__init__(**kwargs)
        assert (mask_dilation > 0)
        self.rescaler = PillowImageRescaler(image_resize_ratio=image_resize_ratio)
        self.dilator = ScipyMaskDilator(dilation=mask_dilation)
        self.use_cuda = use_cuda

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
        masks = raw_data_chunk_list[0]
        assert (len(masks.shape) == 3)

        masks = np.array([self.dilator(self.rescaler(x, is_mask=True)) for x in masks])

        masks = np.expand_dims(masks, axis=-1)
        masks = torch.from_numpy(masks).permute(0, 3, 1, 2).contiguous()
        masks = masks.float()

        if self.use_cuda:
            masks = masks.cuda()

        return masks

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


class ProPainterSIMSequencer(Sequencer):
    """
    Scaled Inpaint Masking (ProPainter-SIM) sequencer.

    Parameters
    ----------
    inp_frames : Sequence
        Inpaint masking sequencer (ProPainter-IM).
    raw_frames : RawFrameSequencer
        Numpy frame sequencer.
    raw_masks : RawMaskSequencer
        Numpy mask sequencer.
    post_raw_mask_dilation : int, default 0
        Post raw mask dilation.
    """
    def __init__(self,
                 inp_frames: Sequence,
                 raw_frames: RawFrameSequencer,
                 raw_masks: RawMaskSequencer,
                 rescaler: PillowImageRescaler,
                 post_raw_mask_dilation: int = 0):
        assert (len(raw_frames) > 0)
        super(ProPainterSIMSequencer, self).__init__(data=[inp_frames, raw_frames, raw_masks])
        self.rescaler = rescaler
        self.post_raw_dilator = ScipyMaskDilator(dilation=post_raw_mask_dilation)

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
        assert (len(raw_data_chunk_list) == 3)

        inp_frames = raw_data_chunk_list[0]
        raw_frames = raw_data_chunk_list[1]
        raw_masks = raw_data_chunk_list[2]

        assert isinstance(inp_frames, torch.Tensor)
        assert isinstance(raw_frames, np.ndarray)
        assert isinstance(raw_masks, np.ndarray)

        inp_frames_np = ProPainterSIMSequencer.conv_propainter_frames_into_numpy(inp_frames)

        do_masking = (self.post_raw_dilator.dilation > 0)

        if self.rescaler.do_scale:
            assert (inp_frames_np.shape != raw_frames.shape)
            inp_frames_np = np.array([self.rescaler.invert(x) for x in inp_frames_np])
            do_masking = True

        if do_masking:
            dilated_raw_masks = np.array([self.post_raw_dilator(x) for x in raw_masks])
            dilated_raw_masks = np.expand_dims(dilated_raw_masks, axis=-1)
            inp_frames_np = inp_frames_np * dilated_raw_masks + raw_frames * (1 - dilated_raw_masks)

        return inp_frames_np

    @staticmethod
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


class ScaledProPainterIterator(ProPainterIterator):
    """
    Video Inpainting (ProPainter) iterator for scaled frames.

    Parameters
    ----------
    raw_frames : RawFrameSequencer
        Raw frame sequencer.
    raw_masks : RawMaskSequencer
        Raw mask sequencer.
    image_resize_ratio : float
        Resize ratio.
    mask_dilation : int
        Mask dilation.
    post_raw_mask_dilation : int, default 0
        Post raw mask dilation.
    """
    def __init__(self,
                 raw_frames: RawFrameSequencer,
                 raw_masks: RawMaskSequencer,
                 image_resize_ratio: float,
                 mask_dilation: int,
                 post_raw_mask_dilation: int = 0,
                 **kwargs):
        frames = FrameSequencer(
            data=raw_frames,
            image_resize_ratio=image_resize_ratio)
        masks = MaskSequencer(
            data=raw_masks,
            image_resize_ratio=image_resize_ratio,
            mask_dilation=mask_dilation)
        super(ScaledProPainterIterator, self).__init__(
            frames=frames,
            masks=masks,
            **kwargs)
        self.sclaed_inp_frame_sequencer = ProPainterSIMSequencer(
            inp_frames=self.inp_frame_sequencer,
            raw_frames=raw_frames,
            raw_masks=raw_masks,
            rescaler=frames.rescaler,
            post_raw_mask_dilation=post_raw_mask_dilation)
        self.main_sequencer = self.sclaed_inp_frame_sequencer


def run_streaming_propainter(vi_iterator: ScaledProPainterIterator) -> np.ndarray:
    """
    Run scaled ProPainter in streaming mode on the entire data sequence.

    Parameters
    ----------
    vi_iterator : ScaledProPainterIterator
        Scaled ProPainter iterator.

    Returns
    -------
    np.ndarray
        Resulted frames.
    """
    vi_frames = None
    for frames_i in vi_iterator:
        if vi_frames is None:
            vi_frames = frames_i
        else:
            vi_frames = np.concatenate([vi_frames, frames_i])
    return vi_frames
