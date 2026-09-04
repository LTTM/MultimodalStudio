# Copyright 2025 Sony Group Corporation.
# All rights reserved.
#
# Licenced under the License reported at
#
#     https://github.com/LTTM/MultimodalStudio/LICENSE.txt (the "License").
#
# This code is inspired by the code available at
#
#     https://github.com/autonomousvision/factor-fields (commit 21ea155d70efce5f96399830cb424c444c977948)
#
# At the moment of this file creation, the original code is licensed under the MIT License,
# Copyright (c) 2023 autonomousvision; a copy of the MIT License, and the list of the files it
# applies to, is reported at
#
#     https://github.com/LTTM/MultimodalStudio/LICENSE_FACTOR_FIELDS.txt
#
# See the License for the specific language governing permissions and limitations under the License.
#
# Author: Federico Lincetto, Ph.D. Student at the University of Padova

from typing import List, Type, Literal, Optional
from dataclasses import dataclass, field

import numpy as np
import torch
from torchtyping import TensorType

from data.scene_box import SceneBox
from engine.callbacks import TrainingCallbackAttributes, TrainingCallback
from field_components.base_field_component import FieldComponentConfig, FieldComponent
from field_components.encodings import EncodingConfig
from field_components.mlp import MLPConfig
from utils.math import discrete_cosine_transform_dict

@dataclass
class CoBaFactorConfig(FieldComponentConfig):

    _target: Type = field(default_factory=lambda: CoBaFactor)
    field_type: Literal["grid", "mlp"] = "grid"
    """Type of field to be used"""
    base_field: FieldComponentConfig = field(default_factory=lambda: MLPConfig)
    """Field component to be used"""
    coordinate_mapping: EncodingConfig = field(default_factory=lambda: EncodingConfig)
    """Factor component encoding type"""
    init_strategy: Literal["random", "value", "dct"] = "random"
    """Initialization strategy"""
    init_value: Optional[float] = 0.5
    """Initialization value"""
    standardize_output: bool = False
    """Whether to standardize the output of the field component"""

@dataclass
class DictionaryFieldConfig(FieldComponentConfig):

    _target: type = field(default_factory=lambda: DictionaryField)
    basis_field: FieldComponentConfig = field(default_factory=lambda: CoBaFactorConfig)
    """Basis field component to be used"""
    coefficient_field: FieldComponentConfig = field(default_factory=lambda: CoBaFactorConfig)
    """Coefficient field component to be used"""
    projection_function: FieldComponentConfig = field(default_factory=lambda: MLPConfig)
    """Projection function to be used"""
    output_latents: bool = False
    """Whether to output the intermediate latent vectors"""

@dataclass
class RadianceDictionaryFieldConfig(DictionaryFieldConfig):

    _target: Type = field(default_factory=lambda: RadianceDictionaryField)
    radiance_predictor: FieldComponentConfig = field(default_factory=lambda: MLPConfig)
    """Radiance predictor to be used"""
    projection_feature_dim: int = 128
    """Dimension of projection feature to be input to radiance predictor"""

class CoBaFactor(FieldComponent):

    def __init__(
            self,
            config: CoBaFactorConfig,
            input_dim: int = None,
            output_dim: int = None,
            scene_box: SceneBox = None,
            **kwargs
    ):
        """Initialize multi-layer perceptron."""

        super().__init__(
            config,
            input_dim=input_dim,
            output_dim=output_dim,
        )

        self.config = config
        self.scene_box = scene_box.aabb if scene_box is not None else None
        self.base_field = self.config.base_field.setup(input_dim=input_dim, output_dim=output_dim)
        self.coordinate_mapping = self.config.coordinate_mapping.setup(input_dim=3, aabb=self.scene_box)

        if self.config.field_type == "grid":
            self.init_grid_values()

    def forward(self, input_tensor: TensorType["bs":..., "input_dim"], **kwargs) -> TensorType["bs":..., "output_dim"]:
        mapped_tensor = self.coordinate_mapping(input_tensor)
        output = self.base_field(mapped_tensor, **kwargs)
        if self.config.standardize_output:
            output = (output - output.mean(dim=-1, keepdim=True)) / (output.std(dim=-1, keepdim=True) + 1e-6)
        return output

    def get_training_callbacks(self, training_callback_attributes: TrainingCallbackAttributes) -> List[
        TrainingCallback]:
        """Returns the training callbacks."""
        callbacks = self.base_field.get_training_callbacks(training_callback_attributes)
        return callbacks

    def get_model_parameters(self):
        parameters = self.base_field.get_model_parameters()
        return parameters

    def init_grid_values(self):
        state_dict = self.base_field.state_dict()
        for key, params in state_dict.items():
            if self.config.init_strategy == "value":
                new_params = torch.ones_like(params) * self.config.init_value
                new_params.requires_grad_(True)
                state_dict[key] = new_params
            elif self.config.init_strategy == "dct":
                dimension = params.shape[1]
                resolution = params.shape[-1]
                n_atoms = int(np.power(dimension, 1.0 / 3) + 1)
                new_params = discrete_cosine_transform_dict(
                    n_atoms=n_atoms,
                    size=resolution,
                    max_n_basis=dimension,
                    dimensions=3
                ).reshape(1, dimension, resolution, resolution, resolution)
                new_params = torch.tensor(new_params, device=params.device, dtype=params.dtype)
                new_params.requires_grad_(True)
                state_dict[key] = new_params
            elif self.config.init_strategy == "random":
                new_params = torch.rand_like(params)
                new_params.requires_grad_(True)
                state_dict[key] = new_params
            else:
                raise ValueError(f"Unknown initialization strategy: {self.config.init_strategy}")
        self.base_field.load_state_dict(state_dict)

    def get_out_dim(self) -> int:
        """Calculates output dimension of encoding."""
        return self.base_field.get_out_dim()

class DictionaryField(FieldComponent):

    def __init__(
            self,
            config: DictionaryFieldConfig,
            scene_box: SceneBox = None,
            input_dim: int = None,
            output_dim: int = None,
            **kwargs
    ):
        super().__init__(config=config, input_dim=input_dim, output_dim=output_dim)
        self.config = config
        self.basis_field = self.config.basis_field.setup(
            input_dim=3,
            scene_box=scene_box,
            **kwargs
        )
        self.coefficient_field = self.config.coefficient_field.setup(
            input_dim=3,
            scene_box=scene_box,
            **kwargs
        )

        projection_input_dim = self.basis_field.get_out_dim() + input_dim - 3

        self.projection_function = self.config.projection_function.setup(
            input_dim=projection_input_dim,
            output_dim=output_dim,
            **kwargs
        )

    def forward(self, inputs, **kwargs):
        inputs, additional_inputs = torch.split(inputs, [3, inputs.shape[1] - 3], dim=-1)
        # Get Output from the basis and radiance field
        basis_features = self.basis_field(inputs, **kwargs)
        coefficient_features = self.coefficient_field(inputs, **kwargs)
        # Compute Hadamard product
        hadamard = basis_features * coefficient_features
        inputs = torch.cat([hadamard, additional_inputs], dim=-1)
        # Compute the output
        output = self.projection_function(inputs, **kwargs)
        return output

    def get_training_callbacks(self, training_callback_attributes: TrainingCallbackAttributes) -> List[
        TrainingCallback]:
        """Returns the training callbacks."""
        basis_field_callbacks = self.basis_field.get_training_callbacks(training_callback_attributes)
        coeff_field_callbacks = self.coefficient_field.get_training_callbacks(training_callback_attributes)
        callbacks = basis_field_callbacks + coeff_field_callbacks
        return callbacks

    def get_model_parameters(self):
        basis_parameters = self.basis_field.get_model_parameters()
        coefficient_parameters = self.coefficient_field.get_model_parameters()
        # Coefficient parameters are ignored because the multi-res structure is only in the basis field
        parameters = {
            "min_res": basis_parameters["min_res"],
            "max_res": basis_parameters["max_res"],
            "num_levels": basis_parameters["num_levels"],
            "steps_per_level_ratio": basis_parameters["steps_per_level_ratio"],
            "level_init": basis_parameters["level_init"],
            "radius": basis_parameters["radius"],
            "growth_factor_list": basis_parameters["growth_factor_list"],
        }
        assert basis_parameters["radius"] == coefficient_parameters["radius"], "Radius of basis and coefficient field must be the same"
        return parameters

class RadianceDictionaryField(DictionaryField):

    def __init__(
            self,
            config: RadianceDictionaryFieldConfig,
            scene_box: SceneBox = None,
            position_dim=3,
            view_direction_dim=3,
            additional_input_dim=0,
            output_dim: int = None,
            **kwargs
    ):
        # input_dim = position_dim + view_direction_dim + additional_input_dim
        super().__init__(
            config=config,
            scene_box=scene_box,
            input_dim=position_dim,
            output_dim=config.projection_feature_dim,
            **kwargs
        )
        self.config = config

        self.projection_function = self.config.projection_function.setup(
            input_dim=self.basis_field.get_out_dim(),
            output_dim=self.config.projection_feature_dim,
            **kwargs
        )

        radiance_predictor_input_dim = self.config.projection_feature_dim + view_direction_dim + additional_input_dim
        self.radiance_predictor = self.config.radiance_predictor.setup(
            input_dim=radiance_predictor_input_dim,
            output_dim=output_dim,
            **kwargs
        )

    def forward(self, positions, view_directions, additional_inputs, **kwargs):
        # Get Output from the basis and radiance field
        basis_features = self.basis_field(positions, **kwargs)
        coefficient_features = self.coefficient_field(positions, **kwargs)
        # Compute Hadamard product
        hadamard = basis_features * coefficient_features
        # Compute the output
        projection_output = self.projection_function(hadamard, **kwargs)
        inputs = torch.cat([projection_output, view_directions, additional_inputs], dim=-1)
        output = self.radiance_predictor(inputs, **kwargs)
        latents = {
            "radiance_hadamard_latent": hadamard,
            "radiance_projection_latent": projection_output
        }
        if self.config.output_latents:
            return output, latents
        else:
            return output
