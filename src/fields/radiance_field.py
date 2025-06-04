# Copyright 2025 Sony Group Corporation.
# All rights reserved.
#
# Licenced under the License reported at
#
#     https://github.com/LTTM/MultimodalStudio/LICENSE.txt (the "License").
#
# See the License for the specific language governing permissions and limitations under the License.
#
# Author: Federico Lincetto, Ph.D. Student at the University of Padova

"""
Base radiance field module.
"""

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Type

import torch

from field_components.base_field_component import FieldComponentConfig, FieldComponent
from field_components.mlp import MLPConfig

@dataclass
class BaseRadianceFieldConfig(FieldComponentConfig):
    """Base radiance field config."""
    _target: Type = field(default_factory=lambda: BaseRadianceField)

@dataclass
class RadianceFieldConfig(BaseRadianceFieldConfig):
    """Radiance field config."""

    _target: Type = field(default_factory=lambda: RadianceField)
    base_field: FieldComponentConfig = field(default_factory=lambda: MLPConfig)
    """Base field component config for radiance field. This is the field that estimates the radiance of the scene."""

class BaseRadianceField(FieldComponent):
    """Base radiance field module. Any radiance field inherits from this class."""

    config: BaseRadianceFieldConfig

    def __init__(
            self,
            config: BaseRadianceFieldConfig,
            input_dim: int = 6,
            output_dim: int = 3,
    ):
        super().__init__(config, input_dim=input_dim, output_dim=output_dim)

    @abstractmethod
    def forward(self, input_tensor):
        raise NotImplementedError

class RadianceField(BaseRadianceField):
    """Radiance field module. This is the field that estimates the radiance of the scene."""

    config: RadianceFieldConfig

    def __init__(
            self,
            config: RadianceFieldConfig,
            position_dim=3,
            view_direction_dim=3,
            additional_input_dim=0,
            output_dim: int = 3,
    ):
        input_dim = position_dim + view_direction_dim + additional_input_dim
        super().__init__(config, input_dim=input_dim, output_dim=output_dim)
        self.base_field = self.config.base_field.setup(input_dim=self.input_dim, output_dim=self.output_dim)

    def forward(self, positions, view_directions, additional_inputs):
        """Estimate the radiance of the scene given the input positions, view directions and additional inputs."""
        inputs = torch.cat([positions, view_directions, additional_inputs], dim=-1)
        output = self.base_field(inputs)

        return output

    def get_model_parameters(self):
        """Returns the model parameters."""
        return self.base_field.get_model_parameters()
