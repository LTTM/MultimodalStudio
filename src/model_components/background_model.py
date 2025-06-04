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
Background model module.
"""

from dataclasses import dataclass, field
from typing import Type, Union, List, Dict, Optional

import torch
from torch.nn import Parameter

from cameras.rays import RaySamples
from configs.configs import InstantiateConfig
from engine.callbacks import TrainingCallbackAttributes, TrainingCallback
from field_components.base_field_component import FieldComponentConfig
from field_components.field_heads import ModalityHeadConfig
from field_components.spatial_distortions import SpatialDistortionConfig
from fields.nerf_field import NeRFFieldConfig
from utils import profiler

@dataclass
class BackgroundModelConfig(InstantiateConfig):
    """Background model config."""

    _target: Type = field(default_factory=lambda: BackgroundModel)
    background_field: NeRFFieldConfig = field(default_factory=lambda: NeRFFieldConfig)
    """Background field config. This is the field that estimates the background of the scene."""
    modality_heads: Optional[Dict[str, FieldComponentConfig]] = field(default_factory=lambda: {})
    """Modality heads config. These are the fields that estimate the multimodal radiance of the scene"""
    spatial_distortion: Union[None, SpatialDistortionConfig] = None
    """Spatial distortion module to use"""
    radiance_feature_dim: int = 256
    """Dimension of radiance feature to pass to the modality heads"""

class BackgroundModel(torch.nn.Module):
    """
    Background model module. This is the field that estimates the density and the radiance of points outside the
    region of interest of the scene
    """

    config: BackgroundModelConfig

    def __init__(
            self,
            config: BackgroundModelConfig,
            modalities: Dict[str, int],
    ):
        super().__init__()
        self.config = config
        self.modalities = modalities
        self.spatial_distortion = self.config.spatial_distortion.setup() \
            if self.config.spatial_distortion is not None \
            else None
        self.background_field = self.config.background_field.setup(radiance_output_dim=self.config.radiance_feature_dim)
        self.modality_heads = torch.nn.ParameterDict({
            mod: self.config.modality_heads.get(mod, ModalityHeadConfig()).setup(
                input_dim=self.config.radiance_feature_dim,
                output_dim=self.modalities[mod]
            ) for mod in self.modalities
        })

    @profiler.time_function
    def forward(self, ray_samples: RaySamples):
        """
        Estimates the density and the radiance of points outside the region of interest of the scene.
        it computes and returns the pixel radiance by integrating the radiance along the rays

        Args:
            ray_samples (RaySamples): Ray samples to compute the density and the radiance.

        Returns:
            outputs (Dict[str, torch.Tensor]): Dictionary containing the pixel radiance for each modality.
        """
        inputs = ray_samples.frustums.get_start_positions()
        inputs = inputs.view(-1, 3)

        directions = ray_samples.frustums.directions
        directions = directions.reshape(-1, 3)

        if self.spatial_distortion is not None:
            inputs = self.spatial_distortion(inputs)

        density, radiance_feature = self.background_field(inputs, directions)

        density = density.view(*ray_samples.frustums.directions.shape[:-1], -1)
        alphas = ray_samples.get_alphas(density)
        weights = ray_samples.get_weights_from_alphas(alphas)

        outputs = {}
        up_directions = ray_samples.frustums.up_directions
        up_directions = up_directions.reshape(-1, 3)
        for mod in self.modalities:
            radiance_output = self.modality_heads[mod](
                radiance_feature,
                directions=directions,
                up_directions=up_directions
            )
            radiance = radiance_output.view(*ray_samples.frustums.directions.shape[:-1], -1)
            outputs[mod] = torch.sum(weights * radiance, dim=1)

        return outputs

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Returns the parameter groups of the model."""
        param_groups = {
            "background_field": list(self.background_field.parameters()) + list(self.modality_heads.parameters()),
        }
        return param_groups

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        """Returns the training callbacks from both the Dataloader and the Model."""
        callbacks = []
        return callbacks

    def get_model_parameters(self):
        """Returns the field module parameters."""
        return {}
