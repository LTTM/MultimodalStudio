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
The field module baseclass.
"""

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Type, List

from torch import nn
from torchtyping import TensorType

from configs.configs import InstantiateConfig
from engine.callbacks import TrainingCallbackAttributes, TrainingCallback

@dataclass
class FieldComponentConfig(InstantiateConfig):
    """Configuration class for field components. Any neural module is a field component."""

    _target: Type = field(default_factory=lambda: FieldComponent)
    input_dim: int = None
    """Input dimension of the module."""
    output_dim: int = None
    """Output dimension of the module."""

class FieldComponent(nn.Module):
    """Base field module. Any neural module is a field component and inherits from this class.

    Args:
        input_dim: Input dimension of the module.
        output_dim: Output dimension of the module.
    """

    def __init__(self,
            config: FieldComponentConfig,
            input_dim: Optional[int] = None,
            output_dim: Optional[int] = None
    ) -> None:
        super().__init__()
        self.config = config
        self.input_dim = input_dim if input_dim is not None else self.config.input_dim
        self.output_dim = output_dim if output_dim is not None else self.config.output_dim

    def set_in_dim(self, input_dim: int) -> None:
        """Sets input dimension of encoding

        Args:
            input_dim: input dimension
        """
        if input_dim <= 0:
            raise ValueError("Input dimension should be greater than zero")
        self.input_dim = input_dim

    def get_out_dim(self) -> int:
        """Calculates output dimension of encoding."""
        if self.output_dim is None:
            raise ValueError("Output dimension has not been set")
        return self.output_dim

    @abstractmethod
    def forward(self, input_tensor: TensorType["bs":..., "input_dim"]) -> TensorType["bs":..., "output_dim"]:
        """
        Returns processed tensor

        Args:
            input_tensor: Input tensor to process
        """
        raise NotImplementedError

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        """Returns the training callbacks."""
        callbacks = []
        return callbacks
