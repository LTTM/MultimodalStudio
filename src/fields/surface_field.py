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
Surface field module.
"""

from dataclasses import dataclass, field
from typing import Type, List

import torch

from configs.configs import InstantiateConfig
from engine.callbacks import TrainingCallbackAttributes, TrainingCallback
from field_components.base_field_component import FieldComponentConfig
from field_components.encodings import EncodingConfig, NeRFEncodingConfig
from field_components.mlp import MLPConfig

@dataclass
class SurfaceFieldConfig(InstantiateConfig):
    """Base surface field config."""

    _target: Type = field(default_factory=lambda: SurfaceField)
    use_position_encoding: bool = True
    """whether to use positional encoding as input for geometric network"""
    position_encoding: EncodingConfig = field(default_factory=lambda: NeRFEncodingConfig)
    """Positional encoding module to be used"""
    geo_feature_dim: int = 256
    """Dimension of geometric feature to be returned"""
    field: FieldComponentConfig = field(default_factory=lambda: MLPConfig)
    """Field component config for surface field. This is the field that estimates the density of the scene."""

@dataclass
class SDFFieldConfig(SurfaceFieldConfig):
    """Signed distance function field config."""

    _target: Type = field(default_factory=lambda: SDFField)
    inside_outside: bool = False
    """Whether to revert signed distance value, set to True for indoor scene"""

class SurfaceField(torch.nn.Module):
    """Base surface field module. Any surface field inherits from this class."""

    config: SurfaceFieldConfig

    def __init__(
            self,
            config: SurfaceFieldConfig,
    ):
        super().__init__()
        self.config = config

        self.position_encoding = self.config.position_encoding.setup(in_dim=3)

        self.input_dim = self.position_encoding.get_out_dim() if self.config.use_position_encoding else 3
        self.output_dim = 1 + self.config.geo_feature_dim if self.config.geo_feature_dim is not None else 1

    def forward(self, x):
        """Estimates the density of the scene."""
        raise NotImplementedError

    def single_output(self, x):
        """Returns the first output of the field. This is used to return only the density value"""
        return self.forward(x)[0]

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        """Returns the training callbacks."""
        field_callbacks = self.field.get_training_callbacks(training_callback_attributes)
        callbacks = field_callbacks
        return callbacks

    def get_model_parameters(self):
        """Returns the model parameters of the field."""
        return self.field.get_model_parameters()

class SDFField(SurfaceField):
    """Estimates the signed distance function of the scene."""

    config: SDFFieldConfig

    def __init__(
            self,
            config: SDFFieldConfig,
    ):
        super().__init__(config)

        self.field = self.config.field.setup(input_dim=self.input_dim, output_dim=self.output_dim)

    def forward(self, x):
        """
        Estimates and returns the signed distance function of the scene (and the geometry feature) given the input
        coordinates.
        """

        if self.config.use_position_encoding:
            x = self.position_encoding(x)

        out = self.field(x)

        if self.config.geo_feature_dim is not None:
            sdf, geo_feature = torch.split(out, [1, self.config.geo_feature_dim], dim=-1)
        else:
            sdf = out
            geo_feature = None

        return sdf, geo_feature
