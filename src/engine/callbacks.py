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
Callback code used for training iterations
Part of the code from sdfstudio: https://github.com/autonomousvision/sdfstudio
"""

from dataclasses import InitVar, dataclass
from enum import Enum, auto
from inspect import signature
from typing import Callable, Dict, List, Optional, Tuple

from configs.configs import PrintableConfig

@dataclass
class TrainingCallbackAttributes:
    """Attributes that can be used to configure training callbacks.
    The callbacks can be specified in the Dataloader or Model implementations.
    Instead of providing access to the entire Trainer object, we only provide these attributes.
    This should be least prone to errors and fairly clean from a user perspective."""

    model: Optional[InitVar]
    """reference to the model"""
    trainer: PrintableConfig
    """the trainer config"""

class TrainingCallbackLocation(Enum):
    """Enum for specifying where the training callback should be run."""

    BEFORE_TRAIN_ITERATION = auto()
    AFTER_TRAIN_ITERATION = auto()


class TrainingCallback:
    """Callback class used during training.
    The function 'func' with 'args' and 'kwargs' will be called every 'update_every_num_iters' training iterations,
    including at iteration 0. The function is called after the training iteration.

    Args:
        where_to_run: List of locations for when to run callbak (before/after iteration)
        func: The function that will be called.
        update_every_num_iters: How often to call the function `func`.
        iters: Tuple of iteration steps to perform callback
        args: args for the function 'func'.
        kwargs: kwargs for the function 'func'.
    """

    def __init__(
        self,
        where_to_run: List[TrainingCallbackLocation],
        func: Callable,
        update_every_num_iters: Optional[int] = None,
        iters: Optional[Tuple[int, ...]] = None,
        args: Optional[List] = None,
        kwargs: Optional[Dict] = None,
    ):
        assert (
            "step" in signature(func).parameters.keys()
        ), f"'step: int' must be an argument in the callback function 'func': {func.__name__}"
        self.where_to_run = where_to_run
        self.update_every_num_iters = update_every_num_iters
        self.iters = iters
        self.func = func
        self.args = args if args is not None else []
        self.kwargs = kwargs if kwargs is not None else {}

    def run_callback(self, step: int):
        """Callback to run after training step

        Args:
            step: current iteration step
        """
        if self.update_every_num_iters is not None:
            if step % self.update_every_num_iters == 0:
                self.func(*self.args, **self.kwargs, step=step)
        elif self.iters is not None:
            if step in self.iters:
                self.func(*self.args, **self.kwargs, step=step)

    def run_callback_at_location(self, step: int, location: TrainingCallbackLocation):
        """Runs the callback if it's supposed to be run at the given location.

        Args:
            step: current iteration step
            location: when to run callback (before/after iteration)
        """
        if location in self.where_to_run:
            self.run_callback(step=step)
