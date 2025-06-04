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
Vanilla NeRF field module.
"""

from dataclasses import dataclass, field
from typing import Type

import torch

from field_components.base_field_component import FieldComponentConfig
from field_components.encodings import EncodingConfig, NeRFEncodingConfig
from field_components.field_heads import ModalityHeadConfig
from field_components.mlp import MLPConfig

@dataclass
class NeRFFieldConfig(FieldComponentConfig):
    """Vanilla NeRF field config."""

    _target: Type = field(default_factory=lambda: NeRFField)
    base_field: FieldComponentConfig = field(default_factory=lambda: MLPConfig)
    """Base field component config for NeRF. This is the field that estimates the density of the scene."""
    head_field: FieldComponentConfig = field(default_factory=lambda: MLPConfig)
    """Head field component config for NeRF. This is the field that estimates the radiance of the scene."""
    use_position_encoding: bool = True
    """Whether to use position encoding or not."""
    position_encoding: EncodingConfig = field(default_factory=lambda: NeRFEncodingConfig)
    """Position encoding module to be used"""
    use_direction_encoding: bool = True
    """Whether to use direction encoding or not."""
    direction_encoding: EncodingConfig = field(default_factory=lambda: NeRFEncodingConfig)
    """Direction encoding module to be used"""

class NeRFField(torch.nn.Module):
    """Vanilla NeRF field module."""

    config: NeRFFieldConfig

    def __init__(
            self,
            config: NeRFFieldConfig,
            radiance_output_dim: int = 3,
    ):
        super().__init__()
        self.config = config
        self.position_encoding = self.config.position_encoding.setup(in_dim=3)
        self.direction_encoding = self.config.direction_encoding.setup(in_dim=3)

        base_input = self.position_encoding.get_out_dim() if self.config.use_position_encoding else 3
        head_input = self.config.base_field.output_dim + self.direction_encoding.get_out_dim() \
            if self.config.use_direction_encoding \
            else 3 + self.config.base_field.output_dim

        self.base_field = self.config.base_field.setup(
            input_dim=base_input,
            output_dim=self.config.base_field.output_dim
        )

        self.head_field = self.config.head_field.setup(
            input_dim=head_input,
            output_dim=radiance_output_dim
        )

        self.density_head = ModalityHeadConfig(
            field=MLPConfig(
                num_layers=1,
                hidden_dim=64,
                weight_norm=True,
                out_activation="Softplus",
            )
        ).setup(input_dim=self.base_field.output_dim, output_dim=1)

    def forward(self, x, viewing_direction):
        """Estimates the density and radiance of the scene given the input coordinates and viewing direction."""
        if self.config.use_position_encoding:
            x = self.position_encoding(x)
        if self.config.use_direction_encoding:
            viewing_direction = self.direction_encoding(viewing_direction)

        feature = self.base_field(x)
        density = self.density_head(feature)

        head_input = torch.cat([feature, viewing_direction], dim=-1)
        feature = self.head_field(head_input)

        return density, feature
