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
Encoding functions
"""

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Type, Literal
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
class EncodingConfig(FieldComponentConfig):
    """Base class for encoding configurations. It is used to define the encoding"""
    _target: Type = field(default_factory=lambda: Encoding)

@dataclass
class HashEncodingConfig(EncodingConfig):
    """Hash grid encoding configuration. Proposed by Instant-NGP."""

    _target: Type = field(default_factory=lambda: HashEncoding)
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
    implementation: str = "tcnn"
    """implementation of hash encoding. Fallback to "torch" if "tcnn" not available."""

@dataclass
class DenseEncodingConfig(EncodingConfig):
    """Dense grid encoding configuration."""
    _target: Type = field(default_factory=lambda: DenseEncoding)
    num_levels: int = 16
    """number of levels for multi-resolution hash grids"""
    features_per_level: int = 2
    """number of features per level for multi-resolution hash grids"""
    min_res: int = 16
    """min resolution for multi-resolution hash grids"""
    max_res: int = 2048
    """max resolution for multi-resolution hash grids"""
    hash_init_scale: float = 0.001
    """value to initialize hash grid."""
    interpolation: Optional[Literal["Nearest", "Linear", "Smoothstep"]] = "Smoothstep"
    """interpolation override for tcnn hashgrid. Not supported for torch unless linear."""
    implementation: str = "tcnn"
    """implementation of hash encoding. Fallback to "torch" if "tcnn" not available."""

@dataclass
class NeRFEncodingConfig(EncodingConfig):
    """Positional encoding configuration. Proposed by NeRF."""
    _target: Type = field(default_factory=lambda: NeRFEncoding)
    num_frequencies: int = 6
    """Number of frequencies to use to compute the encoding"""
    min_freq_exp: float = 0.0
    """Minimum frequency exponent"""
    max_freq_exp: int = 5
    """Maximum frequency exponent"""
    include_input: bool = True
    """Whether to include the input in the encoding"""

@dataclass
class SHEncodingConfig(EncodingConfig):
    """Spherical harmonic encoding configuration. Proposed by mip-NeRF 360."""

    _target: Type = field(default_factory=lambda: SHEncoding)
    degree: int = 4
    """Degree of spherical harmonics to use for encoding"""

class Encoding(FieldComponent):
    """Encode an input tensor. Intended to be subclassed

    Args:
        in_dim: Input dimension of tensor
    """

    def __init__(self, config: EncodingConfig, in_dim: int = 3) -> None:
        if in_dim <= 0:
            raise ValueError("Input dimension should be greater than zero")
        super().__init__(config, input_dim=in_dim)
        self.config = config

    @abstractmethod
    def forward(self, input_tensor: TensorType["bs":..., "input_dim"]) -> TensorType["bs":..., "output_dim"]:
        """Call forward and returns and processed tensor

        Args:
            input_tensor: the input tensor to process
        """
        raise NotImplementedError

class NeRFEncoding(Encoding):
    """Multi-scale sinusoidal encodings.
    Each axis is encoded with frequencies ranging from 2^min_freq_exp to 2^max_freq_exp.

    Args:
        in_dim: Input dimension of tensor
    """

    def __init__(
            self,
            config: NeRFEncodingConfig,
            in_dim: int = 3,
    ) -> None:

        super().__init__(config, in_dim=in_dim)

        self.num_frequencies = self.config.num_frequencies
        self.min_freq = self.config.min_freq_exp
        self.max_freq = self.config.max_freq_exp
        self.include_input = self.config.include_input

    def get_out_dim(self) -> int:
        if self.input_dim is None:
            raise ValueError("Input dimension has not been set")
        out_dim = self.input_dim * self.num_frequencies * 2

        if self.include_input:
            out_dim += self.input_dim
        return out_dim

    def forward(
        self,
        input_tensor: TensorType["bs":..., "input_dim"],
    ) -> TensorType["bs":..., "output_dim"]:
        """Calculates NeRF encoding.

        Args:
            input_tensor: For best performance, the input tensor should be between 0 and 1.
        Returns:
            Output values will be between -1 and 1
        """

        freqs = 2 ** torch.linspace(self.min_freq, self.max_freq, self.num_frequencies).to(input_tensor.device)

        scaled_inputs = input_tensor[..., None] * freqs  # [..., "input_dim", "num_scales"]
        scaled_inputs = scaled_inputs.view(*scaled_inputs.shape[:-2], -1)  # [..., "input_dim" * "num_scales"]

        encoded_inputs = torch.sin(torch.cat([scaled_inputs, scaled_inputs + torch.pi / 2.0], dim=-1))

        if self.include_input:
            encoded_inputs = torch.cat([input_tensor, encoded_inputs], dim=-1)
        return encoded_inputs

class HashEncoding(Encoding):
    """Multi-resolution hash grid encoding"""

    def __init__(
        self,
        config: HashEncodingConfig,
        in_dim: int = 3,
    ) -> None:

        super().__init__(config, in_dim=in_dim)

        self.growth_factor = np.exp(
            (np.log(self.config.max_res) - np.log(self.config.min_res)) / (self.config.num_levels - 1)
        )
        self.implementation = self.config.implementation
        self.tcnn_encoding = None

        if self.implementation == "tcnn":
            if not TCNN_EXISTS:
                print("WARNING: TCNN not found. Using \"torch\" as slow implementation of tcnn."
                      "Install tcnn for speedups")
                self.implementation = "torch"
            else:
                encoding_config = {
                    "otype": "HashGrid",
                    "n_levels": self.config.num_levels,
                    "n_features_per_level": self.config.features_per_level,
                    "log2_hashmap_size": self.config.log2_hashmap_size,
                    "base_resolution": self.config.min_res,
                    "per_level_scale": self.growth_factor,
                }
                if self.config.interpolation is not None:
                    encoding_config["interpolation"] = self.config.interpolation

                self.tcnn_encoding = tcnn.Encoding(
                    n_input_dims=3,
                    encoding_config=encoding_config,
                )

        if self.implementation == "torch":
            self.hash_table_size = 2 ** self.config.log2_hashmap_size
            levels = torch.arange(self.config.num_levels)

            self.scalings = torch.floor(self.config.min_res * self.growth_factor ** levels)

            self.hash_offset = levels * self.hash_table_size
            self.hash_table = torch.rand(
                size=(self.hash_table_size * self.config.num_levels, self.config.features_per_level)) * 2 - 1
            self.hash_table *= self.config.hash_init_scale
            self.hash_table = nn.Parameter(self.hash_table)

        if not TCNN_EXISTS or self.tcnn_encoding is None:
            assert (
                self.config.interpolation is None or self.config.interpolation == "Linear"
            ), f"interpolation '{self.config.interpolation}' is not supported for torch encoding backend"

    def get_out_dim(self) -> int:
        """Calculates output dimension of encoding."""
        return self.config.num_levels * self.config.features_per_level

    def hash_fn(self, in_tensor: TensorType["bs":..., "num_levels", 3]) -> TensorType["bs":..., "num_levels"]:
        """Returns hash tensor using method described in Instant-NGP

        Args:
            in_tensor: Tensor to be hashed
        """

        # min_val = torch.min(in_tensor)
        # max_val = torch.max(in_tensor)
        # assert min_val >= 0.0
        # assert max_val <= 1.0

        in_tensor = in_tensor * torch.tensor([1, 2654435761, 805459861]).to(in_tensor.device)
        x = torch.bitwise_xor(in_tensor[..., 0], in_tensor[..., 1])
        x = torch.bitwise_xor(x, in_tensor[..., 2])
        x %= self.hash_table_size
        x += self.hash_offset.to(x.device)
        return x

    def pytorch_fwd(self, in_tensor: TensorType["bs":..., "input_dim"]) -> TensorType["bs":..., "output_dim"]:
        """Forward pass using pytorch. Significantly slower than TCNN implementation."""

        assert in_tensor.shape[-1] == 3
        in_tensor = in_tensor[..., None, :]  # [..., 1, 3]
        scaled = in_tensor * self.scalings.view(-1, 1).to(in_tensor.device)  # [..., L, 3]
        scaled_c = torch.ceil(scaled).type(torch.int32)
        scaled_f = torch.floor(scaled).type(torch.int32)

        offset = scaled - scaled_f

        hashed_0 = self.hash_fn(scaled_c)  # [..., num_levels]
        hashed_1 = self.hash_fn(torch.cat([scaled_c[..., 0:1], scaled_f[..., 1:2], scaled_c[..., 2:3]], dim=-1))
        hashed_2 = self.hash_fn(torch.cat([scaled_f[..., 0:1], scaled_f[..., 1:2], scaled_c[..., 2:3]], dim=-1))
        hashed_3 = self.hash_fn(torch.cat([scaled_f[..., 0:1], scaled_c[..., 1:2], scaled_c[..., 2:3]], dim=-1))
        hashed_4 = self.hash_fn(torch.cat([scaled_c[..., 0:1], scaled_c[..., 1:2], scaled_f[..., 2:3]], dim=-1))
        hashed_5 = self.hash_fn(torch.cat([scaled_c[..., 0:1], scaled_f[..., 1:2], scaled_f[..., 2:3]], dim=-1))
        hashed_6 = self.hash_fn(scaled_f)
        hashed_7 = self.hash_fn(torch.cat([scaled_f[..., 0:1], scaled_c[..., 1:2], scaled_f[..., 2:3]], dim=-1))

        f_0 = self.hash_table[hashed_0]  # [..., num_levels, features_per_level]
        f_1 = self.hash_table[hashed_1]
        f_2 = self.hash_table[hashed_2]
        f_3 = self.hash_table[hashed_3]
        f_4 = self.hash_table[hashed_4]
        f_5 = self.hash_table[hashed_5]
        f_6 = self.hash_table[hashed_6]
        f_7 = self.hash_table[hashed_7]

        f_03 = f_0 * offset[..., 0:1] + f_3 * (1 - offset[..., 0:1])
        f_12 = f_1 * offset[..., 0:1] + f_2 * (1 - offset[..., 0:1])
        f_56 = f_5 * offset[..., 0:1] + f_6 * (1 - offset[..., 0:1])
        f_47 = f_4 * offset[..., 0:1] + f_7 * (1 - offset[..., 0:1])

        f0312 = f_03 * offset[..., 1:2] + f_12 * (1 - offset[..., 1:2])
        f4756 = f_47 * offset[..., 1:2] + f_56 * (1 - offset[..., 1:2])

        encoded_value = f0312 * offset[..., 2:3] + f4756 * (
            1 - offset[..., 2:3]
        )  # [..., num_levels, features_per_level]

        return torch.flatten(encoded_value, start_dim=-2, end_dim=-1)  # [..., num_levels * features_per_level]

    def forward(self, input_tensor: TensorType["bs":..., "input_dim"]) -> TensorType["bs":..., "output_dim"]:
        """Calculates hash encoding."""
        if TCNN_EXISTS and self.tcnn_encoding is not None:
            return self.tcnn_encoding(input_tensor)
        return self.pytorch_fwd(input_tensor)

class DenseEncoding(Encoding):
    """Dense multi-resolution grid encoding"""

    def __init__(
        self,
        config: DenseEncodingConfig,
        in_dim: int = 3,
    ) -> None:

        super().__init__(config, in_dim=in_dim)

        self.growth_factor = np.exp(
            (np.log(self.config.max_res) - np.log(self.config.min_res)) / (self.config.num_levels - 1)
        )
        self.implementation = self.config.implementation
        self.tcnn_encoding = None

        if self.implementation == "tcnn":
            if not TCNN_EXISTS:
                print("WARNING: TCNN not found. Using \"torch\" as slow implementation of tcnn."
                      "Install tcnn for speedups")
                self.implementation = "torch"
            else:
                encoding_config = {
                    "otype": "DenseGrid",
                    "n_levels": self.config.num_levels,
                    "n_features_per_level": self.config.features_per_level,
                    "base_resolution": self.config.min_res,
                    "per_level_scale": self.growth_factor,
                }
                if self.config.interpolation is not None:
                    encoding_config["interpolation"] = self.config.interpolation

                self.tcnn_encoding = tcnn.Encoding(
                    n_input_dims=3,
                    encoding_config=encoding_config,
                )

        if self.implementation == "torch":
            raise NotImplementedError

        if not TCNN_EXISTS or self.tcnn_encoding is None:
            assert (
                self.config.interpolation is None or self.config.interpolation == "Linear"
            ), f"interpolation '{self.config.interpolation}' is not supported for torch encoding backend"

    def get_out_dim(self) -> int:
        """Calculates output dimension of encoding."""
        return self.config.num_levels * self.config.features_per_level

    def forward(self, input_tensor: TensorType["bs":..., "input_dim"]) -> TensorType["bs":..., "output_dim"]:
        """Calculates dense encoding."""
        if TCNN_EXISTS and self.tcnn_encoding is not None:
            return self.tcnn_encoding(input_tensor)
        return self.pytorch_fwd(input_tensor)

class SHEncoding(Encoding):
    """Spherical harmonic encoding"""

    def __init__(
        self,
        config: SHEncodingConfig,
        in_dim: int = 3,
    ):
        super().__init__(config, in_dim=in_dim)
        self.direction_encoding = tcnn.Encoding(
            n_input_dims=in_dim,
            encoding_config={
                "otype": "SphericalHarmonics",
                "degree": self.config.degree + 1,
            },
        )

    def get_out_dim(self) -> int:
        """Calculates output dimension of encoding."""
        return (self.config.degree + 1)**2

    def forward(self, input_tensor: TensorType["bs":..., "input_dim"]) -> TensorType["bs":..., "output_dim"]:
        """Calculates spherical harmonic encoding."""
        input_tensor = (input_tensor + 1) / 2 # Needed due to tiny-cuda-nn implementation
        return self.direction_encoding(input_tensor)
