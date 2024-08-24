"""
    Common test utilities.
"""

__all__ = ['check_arrays']

import os
import cv2
import numpy as np
import torch


def check_arrays(gt_arrays_dir_path: str,
                 pref: str,
                 tested_array: torch.Tensor | np.ndarray,
                 start_idx: int,
                 end_idx: int,
                 c_slice: slice = slice(None),
                 do_save: bool = False,
                 precise: bool = True,
                 atol: float = 1.0,
                 file_format: str = "npy"):
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
    file_format : str, default 'npy'
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

        tested_array_i_file_path = os.path.join(gt_arrays_dir_path, pref + "{:05d}.{}".format(i, file_format))
        if do_save:
            if file_format == "npy":
                np.save(tested_array_i_file_path, tested_array_i)
            else:
                tested_array_i = cv2.cvtColor(tested_array_i, cv2.COLOR_BGR2RGB)
                cv2.imwrite(tested_array_i_file_path, tested_array_i)
            continue

        if file_format == "npy":
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
