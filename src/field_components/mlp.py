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
MLP network.
"""

from dataclasses import dataclass, field
from typing import Type, Optional, Tuple, Literal
from torchtyping import TensorType

import numpy as np
import torch
from torch import nn

from field_components.base_field_component import FieldComponent, FieldComponentConfig

try:
    import tinycudann as tcnn
    TCNN_EXISTS = True
except ImportError:
    TCNN_EXISTS = False

@dataclass
class MLPConfig(FieldComponentConfig):
    """MLP implementation config"""

    _target: Type = field(default_factory=lambda: MLP)
    num_layers: int = 8
    """Number of layers"""
    hidden_dim: int = 128
    """Hidden dimension of each layer"""
    weight_norm: bool = True
    """Whether to use weight normalization"""
    activation: str = "ReLU"
    """Hidden layer activation function"""
    activation_params: dict = field(default_factory=dict)
    """Activation function parameters"""
    out_activation: Optional[str] = "Sigmoid"
    """Output layer activation function"""
    skip_connections: Optional[Tuple[int]] = field(default_factory=lambda: [])
    """Skip connections for the MLP. Tuple containing the layer indices where to add skip connections."""
    geometric_init: bool = False
    """Whether to use geometric initialization"""
    geometric_init_bias: float = 0.5
    """Defines the radius of the sphere centered in the origin at initialization."""

@dataclass
class FullyFusedMLPConfig(FieldComponentConfig):
    """FullyFusedMLP implementation config of tiny-cuda-nn"""

    _targte: Type = field(default_factory=lambda: FullyFusedMLP)
    num_layers: int = 4
    """Number of layers"""
    hidden_dim: int = 128
    """Hidden dimension of each layer"""
    activation: str = "ReLU"
    """Hidden layer activation function"""
    out_activation: Optional[str] = "None"
    """Output layer activation function"""

@dataclass
class NetworkWithInputEncodingConfig(FieldComponentConfig):
    """Network with input encoding configuration"""

    _target: Type = field(default_factory=lambda: NetworkWithInputEncoding)

    num_layers: int = 4
    """Number of layers"""
    hidden_dim: int = 128
    """Hidden dimension of each layer"""
    activation: str = "ReLU"
    """Hidden layer activation function"""
    out_activation: Optional[str] = "None"
    """Output layer activation function"""
    num_levels: int = 16
    """number of levels for multi-resolution hash grids"""
    features_per_level: int = 2
    """number of features per level for multi-resolution hash grids"""
    min_res: int = 16
    """min resolution for multi-resolution hash grids"""
    max_res: int = 2048
    """max resolution for multi-resolution hash grids"""
    log2_hashmap_size: int = 19
    """log2 hash map size for multi-resolution hash grids"""
    hash_init_scale: float = 0.001
    """value to initialize hash grid."""
    interpolation: Optional[Literal["Nearest", "Linear", "Smoothstep"]] = "Smoothstep"
    """interpolation override for tcnn hashgrid. Not supported for torch unless linear."""

class MLP(FieldComponent):
    """Standard MLP implementation"""

    def __init__(
            self,
            config: MLPConfig,
            input_dim: int = None,
            output_dim: int = None,
    ):
        """Initialize multi-layer perceptron."""
        self.config = config

        super().__init__(config, input_dim=input_dim, output_dim=output_dim)
        if self.output_dim is None:
            self.output_dim = self.config.hidden_dim

        dims = []
        for i in range(self.config.num_layers - 1):
            if i + 1 in self.config.skip_connections:
                dims.append(self.config.hidden_dim + self.input_dim)
            else:
                dims.append(self.config.hidden_dim)
        dims = [self.input_dim] + dims + [self.output_dim]

        layers = []

        for i in range(0, len(dims) - 1):
            if i + 1 in self.config.skip_connections:
                out_dim = dims[i + 1] - dims[0]
            else:
                out_dim = dims[i + 1]

            layer = nn.Linear(dims[i], out_dim)
            layers.append(layer)

        self.layers = nn.ModuleList(layers)

        if self.config.geometric_init:
            if self.input_dim > 3:
                self.geometric_init(bias=self.config.geometric_init_bias, additional_input=True)
            else:
                self.geometric_init(bias=self.config.geometric_init_bias, additional_input=False)
        else:
            self.standard_init()

        if self.config.weight_norm:
            self.weight_norm()

        self.activation = getattr(nn, self.config.activation)(**self.config.activation_params)
        self.out_activation = getattr(nn, self.config.out_activation)() \
            if self.config.out_activation != "None" \
            else None

    def forward(self, input_tensor: TensorType["bs":..., "in_dim"]) -> TensorType["bs":..., "out_dim"]:
        """Process input with a multilayer perceptron.

        Args:
            input_tensor: Network input

        Returns:
            MLP network output
        """
        x = input_tensor
        for i, layer in enumerate(self.layers):
            # as checked in `build_nn_modules`, 0 should not be in `_skip_connections`
            if i in self.config.skip_connections:
                x = torch.cat([x, input_tensor], -1) / np.sqrt(2)
            x = layer(x)
            if i < len(self.layers) - 1:
                x = self.activation(x)
        if self.out_activation is not None:
            x = self.out_activation(x)
        return x

    def geometric_init(self, bias=0.5, inside_outside=False, additional_input=True):
        """Geometric initializer"""

        for l in range(0, len(self.layers)):

            in_dim = self.layers[l].in_features
            out_dim = self.layers[l].out_features

            if l == len(self.layers) - 1:
                if not inside_outside:
                    torch.nn.init.normal_(self.layers[l].weight, mean=np.sqrt(np.pi) / np.sqrt(in_dim), std=0.0001)
                    torch.nn.init.constant_(self.layers[l].bias, -bias)
                else:
                    torch.nn.init.normal_(self.layers[l].weight, mean=-np.sqrt(np.pi) / np.sqrt(in_dim), std=0.0001)
                    torch.nn.init.constant_(self.layers[l].bias, bias)
            elif additional_input and l == 0:
                torch.nn.init.constant_(self.layers[l].bias, 0.0)
                torch.nn.init.constant_(self.layers[l].weight[:, 3:], 0.0)
                torch.nn.init.normal_(self.layers[l].weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
            elif additional_input and l in self.config.skip_connections:
                torch.nn.init.constant_(self.layers[l].bias, 0.0)
                torch.nn.init.normal_(self.layers[l].weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                torch.nn.init.constant_(self.layers[l].weight[:, -(self.layers[0].in_features - 3):], 0.0)
            else:
                torch.nn.init.constant_(self.layers[l].bias, 0.0)
                torch.nn.init.normal_(self.layers[l].weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

    def standard_init(self):
        """Standard weight and bias initializer"""
        for l in range(0, len(self.layers)):
            torch.nn.init.kaiming_uniform_(self.layers[l].weight.data)
            torch.nn.init.zeros_(self.layers[l].bias.data)

    def weight_norm(self):
        """Apply weight normalization to the layers of the MLP."""
        for l in range(len(self.layers)):
            self.layers[l] = torch.nn.utils.parametrizations.weight_norm(self.layers[l])

class FullyFusedMLP(FieldComponent):
    """FullyFusedMLP implementation of tiny-cuda-nn"""

    def __init__(
        self,
        config: FullyFusedMLPConfig,
        input_dim: int = None,
        output_dim: int = None,
    ):
        """Initialize multi-layer perceptron."""
        super().__init__(config, input_dim=input_dim, output_dim=output_dim)

        self.mlp = tcnn.Network(
            n_input_dims=self.input_dim,
            n_output_dims=self.output_dim,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": self.config.activation,
                "output_activation": self.config.out_activation,
                "n_neurons": self.config.hidden_dim,
                "n_hidden_layers": self.config.num_layers - 1,
            },
        )

    def forward(self, input_tensor: TensorType["bs":..., "in_dim"]) -> TensorType["bs":..., "out_dim"]:
        """Process input with a multilayer perceptron.

        Args:
            input_tensor: Network input

        Returns:
            MLP network output
        """
        return self.mlp(input_tensor)

class NetworkWithInputEncoding(FieldComponent):
    """Network with input encoding implementation of tiny-cuda-nn"""

    def __init__(
        self,
        config: NetworkWithInputEncodingConfig,
        input_dim: int = None,
        output_dim: int = None,
    ):
        """Initialize network with input encoding."""
        super().__init__(config, input_dim=input_dim, output_dim=output_dim)

        self.growth_factor = np.exp(
            (np.log(self.config.max_res) - np.log(self.config.min_res)) / (self.config.num_levels - 1)
        )
        encoding_config = {
            "otype": "HashGrid",
            "n_levels": self.config.num_levels,
            "n_features_per_level": self.config.features_per_level,
            "log2_hashmap_size": self.config.log2_hashmap_size,
            "base_resolution": self.config.min_res,
            "per_level_scale": self.growth_factor,
            "interpolation": self.config.interpolation,
        }
        network_config = {
            "otype": "FullyFusedMLP",
            "n_hidden_layers": self.config.num_layers - 1,
            "n_neurons": self.config.hidden_dim,
            "activation": self.config.activation,
            "output_activation": self.config.out_activation,
        }
        self.mlp = tcnn.NetworkWithInputEncoding(
            n_input_dims=self.input_dim,
            n_output_dims=self.output_dim,
            encoding_config=encoding_config,
            network_config=network_config,
        )

    def forward(self, input_tensor: TensorType["bs":..., "in_dim"]) -> TensorType["bs":..., "out_dim"]:
        """Process input with a network with input encoding.

        Args:
            input_tensor: Network input

        Returns:
            Network with input encoding output
        """
        return self.mlp(input_tensor)
