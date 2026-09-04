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
Collection of modality heads for radiance estimation
"""

from dataclasses import dataclass, field
from typing import Optional, Type, Dict
from torchtyping import TensorType

import torch

from field_components.base_field_component import FieldComponent, FieldComponentConfig
from field_components.mlp import MLPConfig
from model_components.polarizer import align_polarization_filters, stokes_to_intensity

@dataclass
class ModalityHeadConfig(FieldComponentConfig):
    """Default modality head config."""

    _target: Type = field(default_factory=lambda: ModalityHead)
    field: Optional[FieldComponentConfig] = field(
        default_factory=lambda: MLPConfig(
            num_layers=1,
            hidden_dim=64,
            weight_norm=True,
            out_activation="Sigmoid",
        )
    )
    """Field component config for modality heads."""

@dataclass
class PolarizationHeadConfig(ModalityHeadConfig):
    """Modality head config for polarization radiance"""
    _target: Type = field(default_factory=lambda: PolarizationHead)
    field: Optional[FieldComponentConfig] = field(
        default_factory=lambda: MLPConfig(
            num_layers=1,
            hidden_dim=64,
            weight_norm=True,
            out_activation="None",
        )
    )
    """Field component config for polarization modality head."""

@dataclass
class ModalityHeadFieldConfig(FieldComponentConfig):
    """Field config for modality heads"""
    _target: Type = field(default_factory=lambda: ModalityHeadField)
    decoder: Optional[FieldComponentConfig] = None
    """Field component config for modality heads."""
    decoder_output_dim: int = 16
    """Output dimension of the field component before modality heads."""
    modality_heads: Optional[Dict[str, FieldComponentConfig]] = field(default_factory=lambda: {})
    """Modality heads config. These are the fields that estimate the multimodal radiance of the scene"""

class ModalityHead(FieldComponent):
    """Base class for modality heads"""

    config: ModalityHeadConfig

    def __init__(
            self, config: ModalityHeadConfig,
            input_dim: int = None,
            output_dim: int = None,
            **kwargs
    ):
        super().__init__(config, input_dim=input_dim, output_dim=output_dim)
        self.config = config
        assert input_dim is not None, "input_dim must be provided"
        assert output_dim is not None, "output_dim must be provided"
        self.field = self.config.field.setup(input_dim=input_dim, output_dim=output_dim)

    def forward(self, input_tensor: TensorType["N", "C"], **kwargs) -> TensorType["N", "C"]:
        """Forward pass of the modality head"""
        return self.field(input_tensor, **kwargs)

class PolarizationHead(ModalityHead):
    """Modality head for polarization"""

    config: PolarizationHeadConfig

    def __init__(
            self,
            config: PolarizationHeadConfig,
            input_dim: int = None,
            output_dim: int = 3,
            **kwargs
    ):
        super().__init__(config, input_dim=input_dim, output_dim=output_dim)
        self.config = config
        self.field = self.config.field.setup(input_dim=input_dim, output_dim=3, **kwargs)

    def forward(self, input_tensor: TensorType["N", "Cs"], directions, up_directions, **kwargs) -> TensorType["N", "Cp"]:
        """
        Forward pass of the polarization head

        Args:
            input_tensor: Input tensor to process
            directions: Directions of the rays
            up_directions: Up directions of the camera

        Returns:
            polarization_channels: radiance intensity polarized at 0, 45, 90, 135 degrees
        """
        stokes = self.field(input_tensor, **kwargs)
        stokes[..., 0] = torch.nn.functional.leaky_relu(stokes[..., 0].clone())
        aligned_stokes = align_polarization_filters(stokes, directions, up_directions)
        polarization_channels, _ = stokes_to_intensity(aligned_stokes)
        return polarization_channels

class ModalityHeadField(FieldComponent):
    """Field component for modality heads"""

    config: ModalityHeadFieldConfig

    def __init__(
            self,
            config: ModalityHeadFieldConfig,
            input_dim: int,
            modalities: Dict[str, int],
            **kwargs
    ):
        super().__init__(config, input_dim=input_dim, output_dim=None)
        self.config = config
        self.modalities = modalities

        decoder_input_dim = input_dim
        modality_heads_input_dim = input_dim
        if self.config.decoder is not None:
            self.decoder = self.config.decoder.setup(input_dim=decoder_input_dim, output_dim=self.config.decoder_output_dim, **kwargs)
            modality_heads_input_dim = self.config.decoder_output_dim

        self.modality_heads = torch.nn.ParameterDict({
            mod: self.config.modality_heads.get(mod, ModalityHeadConfig()).setup(
                input_dim=modality_heads_input_dim,
                output_dim=self.modalities[mod],
                **kwargs
            ) for mod in self.modalities
        })

    def forward(
            self,
            input_tensor: TensorType["N", "C"],
            directions: TensorType["N", 3],
            up_directions: TensorType["N", 3],
            modality: str,
            **kwargs
    ) -> Dict[str, TensorType["N", "C"]]:

        if self.config.decoder is not None:
            input_tensor = self.decoder(input_tensor, **kwargs)

        radiance_output = self.modality_heads[modality](
            input_tensor,
            directions=directions,
            up_directions=up_directions,
            **kwargs
        )

        return radiance_output
