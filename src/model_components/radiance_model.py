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
Radiance model.
"""

from dataclasses import dataclass, field
from typing import Type, Union, List, Dict, Optional

import torch
from torch.nn import Parameter
from torchtyping import TensorType

from cameras.rays import RaySamples
from configs.configs import InstantiateConfig
from engine.callbacks import TrainingCallbackAttributes, TrainingCallback
from field_components.base_field_component import FieldComponentConfig
from field_components.encodings import EncodingConfig, NeRFEncodingConfig
from field_components.field_heads import ModalityHeadConfig
from field_components.spatial_distortions import SpatialDistortionConfig
from fields.radiance_field import RadianceFieldConfig, BaseRadianceFieldConfig
from utils import profiler

@dataclass
class RadianceModelConfig(InstantiateConfig):
    """Radiance model config."""

    _target: Type = field(default_factory=lambda: RadianceModel)
    spatial_distortion: Union[None, SpatialDistortionConfig] = None
    """Spatial distortion module to use"""
    radiance_field: BaseRadianceFieldConfig = field(default_factory=lambda: RadianceFieldConfig)
    """Radiance field config. This is the base field that estimates the radiance of the scene."""
    modality_heads: Optional[Dict[str, FieldComponentConfig]] = field(default_factory=lambda: {})
    """Modality heads config. These are the fields that estimate the multimodal radiance of the scene"""
    use_direction_encoding: bool = True
    """Whether to use positional encoding as input for radiance network"""
    direction_encoding: EncodingConfig = field(default_factory=lambda: NeRFEncodingConfig)
    """Positional encoding module to be used for direction encoding"""
    use_n_dot_v: bool = False
    """Whether to use dot product between surface normal and view direction as input for radiance network"""
    use_reflection_direction: bool = False
    """Whether to use reflection direction insead of viewing direction as input for radiance network"""
    geo_feature_dim: int = 256
    """Dimension of geometric feature to receive as input"""
    radiance_feature_dim: int = 256
    """Dimension of radiance feature to pass to the modality heads"""

class RadianceModel(torch.nn.Module):
    """
    Radiance model module. This is the field that estimates the radiance of the scene.
    """

    config: RadianceModelConfig

    def __init__(
            self,
            config: RadianceModelConfig,
            modalities: Dict[str, int],
    ):
        super().__init__()
        self.config = config
        self.modalities = modalities
        self.spatial_distortion = self.config.spatial_distortion.setup() \
            if self.config.spatial_distortion is not None \
            else None
        self.direction_encoding = self.config.direction_encoding.setup(in_dim=3)
        position_input_dim = 3
        direction_input_dim = self.direction_encoding.get_out_dim() if self.config.use_direction_encoding else 3
        additional_input_dim = self.config.geo_feature_dim
        additional_input_dim += 1 if self.config.use_n_dot_v else 0
        self.radiance_field = self.config.radiance_field.setup(
            position_dim=position_input_dim,
            view_direction_dim=direction_input_dim,
            additional_input_dim=additional_input_dim,
            output_dim=self.config.radiance_feature_dim,
        )
        self.modality_heads = torch.nn.ParameterDict({
            mod: self.config.modality_heads.get(mod, ModalityHeadConfig()).setup(
                input_dim=self.config.radiance_feature_dim,
                output_dim=self.modalities[mod],
            ) for mod in self.modalities
        })

    @profiler.time_function
    def forward(
            self,
            ray_samples: RaySamples,
            normals: TensorType["batch_size", 3],
            geo_feature: TensorType["batch_size", "geo_feature_dim"]
    ):
        """
        Estimates the radiance of the scene given the input positions, view directions, surface normals and
        additional inputs.
        """
        inputs = ray_samples.frustums.get_start_positions()
        position_input = inputs.view(-1, 3)
        directions = ray_samples.frustums.directions.reshape(-1, 3)
        direction_input = directions.clone()
        normals = normals.view(-1, 3)

        if self.spatial_distortion is not None:
            position_input = self.spatial_distortion(position_input)
        position_input.requires_grad_(True)

        additional_input = [geo_feature]
        if self.config.use_n_dot_v:
            n_dot_v = torch.sum(normals * -directions, dim=-1, keepdim=True)
            additional_input.append(n_dot_v)

        if self.config.use_reflection_direction:
            if self.config.use_n_dot_v:
                direction_input = 2 * (n_dot_v * normals) + direction_input
            else:
                direction_input = 2 * (torch.sum(
                    normals * -direction_input,
                    dim=-1,
                    keepdim=True
                ) * normals) + direction_input

        if self.config.use_direction_encoding:
            direction_input = self.direction_encoding(direction_input)

        additional_input = torch.cat(additional_input, dim=-1)

        radiance_feature = self.radiance_field(
            positions=position_input,
            view_directions=direction_input,
            additional_inputs=additional_input,
        )

        outputs = {}
        up_directions = ray_samples.frustums.up_directions
        up_directions = up_directions.reshape(-1, 3)
        for mod in self.modalities:
            radiance_output = self.modality_heads[mod](
                radiance_feature,
                directions=directions,
                up_directions=up_directions
            )
            outputs[mod] = radiance_output.view(*ray_samples.frustums.directions.shape[:-1], -1)

        return outputs

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Returns the parameter groups of the model to be passed to the optimizer."""
        param_groups = {
            "radiance_field": list(self.radiance_field.parameters()) + list(self.modality_heads.parameters()),
        }
        return param_groups

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        """Returns the training callbacks from both the Dataloader and the Model."""
        callbacks = self.radiance_field.get_training_callbacks(training_callback_attributes)
        return callbacks

    def get_model_parameters(self):
        """Returns the model parameters."""
        return self.radiance_field.get_model_parameters()
