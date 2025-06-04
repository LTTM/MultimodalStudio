# Copyright 2025 Sony Group Corporation.
# All rights reserved.
#
# Licenced under the License reported at
#
#     https://github.com/LTTM/MultimodalStudio/LICENSE.txt (the "License").
#
# This code is a modified version of the original code available at
#
#     https://github.com/autonomousvision/sdfstudio (commit 370902a10dbef08cb3fe4391bd3ed1e227b5c165)
#
# At the moment of this file creation, the original code is licensed under the Apache License, Version 2.0;
# You may obtain a copy of the Apache License, Version 2.0, at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# See the License for the specific language governing permissions and limitations under the License.
#
# Author: Federico Lincetto, Ph.D. Student at the University of Padova

"""
Miscellaneous helper code.
"""

from typing import Any, Callable, Dict, List, Optional, Union
import numpy as np

import torch


def get_dict_to_torch(stuff: Any, device: Union[torch.device, str] = "cpu", exclude: Optional[List[str]] = None):
    """Set everything in the dict to the specified torch device.

    Args:
        stuff: things to convert to torch
        device: machine to put the "stuff" on
        exclude: list of keys to skip over transferring to device
    """
    if isinstance(stuff, dict):
        for k, v in stuff.items():
            if exclude and k in exclude:
                stuff[k] = v
            else:
                stuff[k] = get_dict_to_torch(v, device)
        return stuff
    if isinstance(stuff, torch.Tensor):
        return stuff.to(device)
    return stuff


def get_dict_to_cpu(stuff: Any):
    """Set everything in the dict to CPU.

    Args:
        stuff: things to place onto cpu
    """
    if isinstance(stuff, dict):
        for k, v in stuff.items():
            stuff[k] = get_dict_to_cpu(v)
        return stuff
    if isinstance(stuff, torch.Tensor):
        return stuff.detach().cpu()
    return stuff


def get_masked_dict(d, mask):
    """Return a masked dictionary.
    (NeRFStudio comment): add more asserts/checks so this doesn't have unpredictable behavior.

    Args:
        d: dict to process
        mask: mask to apply to values in dictionary
    """
    masked_dict = {}
    for key, value in d.items():
        masked_dict[key] = value[mask]
    return masked_dict


class IterableWrapper:  # pylint: disable=too-few-public-methods
    """A helper that will allow an instance of a class to return multiple kinds of iterables bound
    to different functions of that class.

    To use this, take an instance of a class. From that class, pass in the <instance>.<new_iter_function>
    and <instance>.<new_next_function> to the IterableWrapper constructor. By passing in the instance's
    functions instead of just the class's functions, the self argument should automatically be accounted
    for.

    Args:
        new_iter: function that will be called instead as the __iter__() function
        new_next: function that will be called instead as the __next__() function
        length: length of the iterable. If -1, the iterable will be infinite.


    Attributes:
        new_iter: object's pointer to the function we are calling for __iter__()
        new_next: object's pointer to the function we are calling for __next__()
        length: length of the iterable. If -1, the iterable will be infinite.
        i: current index of the iterable.

    """

    i: int

    def __init__(self, new_iter: Callable, new_next: Callable, length: int = -1):
        self.new_iter = new_iter
        self.new_next = new_next
        self.length = length

    def __next__(self):
        if self.length != -1 and self.i >= self.length:
            raise StopIteration
        self.i += 1
        return self.new_next()

    def __iter__(self):
        self.new_iter()
        self.i = 0
        return self


def scale_dict(dictionary: Dict[Any, Any], coefficients: Dict[str, float]) -> Dict[Any, Any]:
    """Scale a dictionary in-place given a coefficients dictionary.

    Args:
        dictionary: input dict to be scaled.
        coefficients: scalar dict config for holding coefficients.

    Returns:
        Input dict scaled by coefficients.
    """
    for key in dictionary:
        if key in coefficients:
            dictionary[key] *= coefficients[key]
    return dictionary

def update_avg(prev_avg: float, new_val: float, step: int) -> float:
    """helper to calculate the running average

    Args:
        prev_avg (float): previous average value
        new_val (float): new value to update the average with
        step (int): current step number

    Returns:
        float: new updated average
    """
    return (step * prev_avg + new_val) / (step + 1)

def normalize_frame(frame: Union[np.ndarray, torch.Tensor]):
    """Normalize a frame to [0, 1] range."""
    if frame.dtype in [np.uint8, torch.uint8]:
        return frame / 255.0
    elif frame.dtype in [np.uint16]:#, torch.uint16]: To enable this Pytorch >= 2.3 is required
        return frame / 65535.0
    else:
        raise NotImplementedError(f"Normalization for {frame.dtype} not implemented. Exiting.")

def check_step(step, target_step, skip_first=True):
    """Check if the current step is a multiple of the target step."""
    if skip_first:
        return step % target_step == 0 and step != 0
    else:
        return step % target_step == 0
