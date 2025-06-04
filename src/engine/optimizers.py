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
Optimizers class.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Type

import torch

from configs.configs import PrintableConfig
from utils import writer

@dataclass
class OptimizerConfig(PrintableConfig):
    """Basic optimizer config with RAdam"""
    _target: Type = torch.optim.Adam
    lr: float = 0.0005
    eps: float = 1e-08

    # (Comment from NeRFStudio authors): Somehow make this more generic. I don't like the idea of overriding the setup
    # function but also not sure how to go about passing things into predefined torch objects.
    def setup(self, params) -> Any:
        """Returns the instantiated object using the config."""
        kwargs = vars(self).copy()
        kwargs.pop("_target")
        return self._target(params, **kwargs)

@dataclass
class AdamOptimizerConfig(OptimizerConfig):
    """Basic optimizer config with Adam"""
    _target: Type = torch.optim.Adam
    weight_decay: float = 0

@dataclass
class AdamWOptimizerConfig(OptimizerConfig):
    """Basic optimizer config with AdamW"""
    _target: Type = torch.optim.AdamW
    weight_decay: float = 0

@dataclass
class RAdamOptimizerConfig(OptimizerConfig):
    """Basic optimizer config with RAdam"""
    _target: Type = torch.optim.RAdam

class Optimizers:
    """Class to manage a set of optimizers and their own schedulers.

    Args:
        optimizers: dictionary of optimizers
        schedulers: dictionary of schedulers
    """

    def __init__(self, optimizers: Dict[str, Any], schedulers: Dict[str, Any]):
        self.optimizers = optimizers
        self.schedulers = schedulers

    def optimizer_step(self, param_group_name: str) -> None:
        """Fetch and step corresponding optimizer.

        Args:
            param_group_name: name of optimizer to step forward
        """
        self.optimizers[param_group_name].step()

    def scheduler_step(self, param_group_name: str) -> None:
        """Fetch and step corresponding scheduler.

        Args:
            param_group_name: name of scheduler to step forward
        """
        if param_group_name in self.schedulers:  # type: ignore
            self.schedulers[param_group_name].step()

    def zero_grad_all(self) -> None:
        """Zero the gradients for all optimizer parameters."""
        for _, optimizer in self.optimizers.items():
            optimizer.zero_grad()

    def optimizer_step_all(self):
        """Run step for all optimizers."""
        for _, optimizer in self.optimizers.items():
            # note that they key is the parameter name
            optimizer.step()

    def scheduler_step_all(self, step: int) -> None:
        """Run step for all schedulers.

        Args:
            step: the current step
        """
        for param_group_name, scheduler in self.schedulers.items():
            scheduler.step()
            lr = scheduler.get_last_lr()
            writer.put_scalar(name=f"learning_rate/{param_group_name}", scalar=lr, step=step)

    def load_optimizers(self, loaded_state: Dict[str, Any]) -> None:
        """Helper to load the optimizer state from previous checkpoint

        Args:
            loaded_state: the state from the previous checkpoint
        """
        for k, v in loaded_state.items():
            self.optimizers[k].load_state_dict(v)

    def load_schedulers(self, loaded_state: Dict[str, Any]) -> None:
        """Helper to load the schedulers state from previous checkpoint

        Args:
            loaded_state: the state from the previous checkpoint
        """
        for k, v in loaded_state.items():
            self.schedulers[k].load_state_dict(v)
