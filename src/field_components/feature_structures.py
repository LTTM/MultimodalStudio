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
Feature grid and MLP field component.
"""

from dataclasses import dataclass, field
from typing import Type, List, Union
from torchtyping import TensorType

import torch

from engine.callbacks import TrainingCallbackAttributes, TrainingCallback, TrainingCallbackLocation
from field_components.base_field_component import FieldComponent, FieldComponentConfig
from field_components.encodings import EncodingConfig
from field_components.mlp import MLPConfig, FullyFusedMLPConfig


@dataclass
class FeatureGridConfig(FieldComponentConfig):
    """Feature grid configuration."""

    _target: Type = field(default_factory=lambda: FeatureGrid)
    encoding: EncodingConfig = field(default_factory=lambda: EncodingConfig)
    """feature grid encoding type"""
    coarse_to_fine: bool = True
    """Whether to use mask to enable coarse to fine training"""
    steps_per_level_ratio: float = 1.0
    """Number of steps per level ratio"""
    level_init: int = 1
    """Initial level for training"""
    radius: float = 1
    """Radius to rescale the input in the feature grid"""

@dataclass
class FeatureGridAndMLPConfig(FieldComponentConfig):
    """Feature grid and MLP configuration."""

    _target: Type = field(default_factory=lambda: FeatureGridAndMLP)
    feature_grid: FeatureGridConfig = field(default_factory=lambda: FeatureGridConfig)
    """feature grid"""
    mlp_head: Union[MLPConfig, FullyFusedMLPConfig] = field(default_factory=lambda: MLPConfig)
    """MLP stacked after hash grid"""
    return_features: bool = False
    """Whether to return features"""

class FeatureGrid(FieldComponent):
    """Feature grid field component. Extract and return local features from a 3D point."""

    def __init__(
            self,
            config: FeatureGridConfig,
            input_dim: int = None,
            output_dim: int = None,
    ):
        """Initialize multi-layer perceptron."""
        super().__init__(config, input_dim=input_dim, output_dim=output_dim)
        self.config = config
        self.radius = self.config.radius

        # feature encoding
        self.encoding = self.config.encoding.setup(in_dim = 3)
        self.output_dim = self.encoding.get_out_dim()
        self.hash_encoding_mask = torch.ones(
            self.config.encoding.num_levels * self.config.encoding.features_per_level,
            dtype=torch.float32
        )

    def forward(self, input_tensor: TensorType["bs":..., "input_dim"]) -> TensorType["bs":..., "output_dim"]:
        """Extract features from input tensor."""
        rescaled_input = (input_tensor + self.radius) / (2 * self.radius)
        features = self.encoding(rescaled_input)
        features = features * self.hash_encoding_mask
        return features

    def update_mask(self, level: int):
        """Update the coarse-to-fine mask"""
        self.hash_encoding_mask[:] = 1.0
        self.hash_encoding_mask[level * self.config.encoding.features_per_level:] = 0

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        """Returns the training callbacks."""
        callbacks = super().get_training_callbacks(training_callback_attributes)
        # anneal for cos in NeuS
        if self.config.coarse_to_fine:
            def set_mask(step):
                steps_per_level = int(
                    training_callback_attributes.trainer.max_num_iterations * self.config.steps_per_level_ratio
                )
                steps_per_level = min(
                    steps_per_level,
                    int(training_callback_attributes.trainer.max_num_iterations / self.config.encoding.num_levels)
                )
                level = int(step / steps_per_level) + 1
                level = max(level, self.config.level_init)
                level = min(level, self.config.encoding.num_levels)
                self.update_mask(level)

            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=set_mask,
                )
            )
        return callbacks

    def get_model_parameters(self):
        """Returns the model parameters."""
        parameters = {
            "num_levels": self.config.encoding.num_levels,
            "min_res": self.config.encoding.min_res,
            "max_res": self.config.encoding.max_res,
            "steps_per_level_ratio": self.config.steps_per_level_ratio,
            "level_init": self.config.level_init,
        }
        return parameters

class FeatureGridAndMLP(FieldComponent):
    """
    Feature grid and MLP field component.
    Extract local features from a 3D point and pass them through an MLP.
    """

    def __init__(
            self,
            config: FeatureGridAndMLPConfig,
            input_dim: int = None,
            output_dim: int = None,
    ):
        """Initialize multi-layer perceptron."""
        super().__init__(config, input_dim=input_dim, output_dim=output_dim)
        self.config = config

        # feature grid
        self.feature_grid = self.config.feature_grid.setup(input_dim = 3)

        mlp_input_dim = input_dim + self.feature_grid.encoding.get_out_dim()
        self.mlp_head = self.config.mlp_head.setup(input_dim=mlp_input_dim, output_dim=output_dim)
        self.output_dim = self.mlp_head.get_out_dim()

    def forward(self, input_tensor: TensorType["bs":..., "input_dim"]) -> TensorType["bs":..., "output_dim"]:
        """Extract features from input tensor and pass them through MLP."""
        auxiliary_input = None
        if input_tensor.shape[-1] > 3:
            auxiliary_input = input_tensor[..., 3:]
            input_tensor = input_tensor[..., :3]

        features = self.feature_grid(input_tensor)
        if auxiliary_input is not None:
            mlp_input = torch.cat([input_tensor, auxiliary_input, features], dim=-1)
        else:
            mlp_input = torch.cat([input_tensor, features], dim=-1)
        output = self.mlp_head(mlp_input)

        if self.config.return_features:
            return output, features
        return output

    def get_model_parameters(self):
        """Returns the model parameters."""
        return self.feature_grid.get_model_parameters()
