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

"""Space distortions."""

from abc import abstractmethod
from typing import Union, Type
from dataclasses import dataclass, field

import torch
from torch import nn
from torchtyping import TensorType

from configs.configs import InstantiateConfig
from utils.math import Gaussians

@dataclass
class SpatialDistortionConfig(InstantiateConfig):
    """Base configuration for spatial distortions."""
    _target: Type = field(default_factory=lambda: SpatialDistortion)

@dataclass
class SceneContractionConfig(SpatialDistortionConfig):
    """Scene contraction proposed in MipNeRF-360."""

    _target: Type = field(default_factory=lambda: SceneContraction)
    order: Union[None, int, float] = None
    """Order of the norm to use for contraction. If None, use the Frobenius norm. If L_inf, use the L_inf norm."""

class SpatialDistortion(nn.Module):
    """Apply spatial distortions"""

    config: SpatialDistortionConfig

    @abstractmethod
    def forward(
        self, positions: Union[TensorType["bs":..., 3], Gaussians]
    ) -> Union[TensorType["bs":..., 3], Gaussians]:
        """
        Args:
            positions: Sample to distort

        Returns:
            Union: distorted sample
        """
        raise NotImplementedError

class SceneContraction(SpatialDistortion):
    """Contract unbounded space using the contraction was proposed in MipNeRF-360.
        We use the following contraction equation:

        .. math::

            f(x) = \\begin{cases}
                x & ||x|| \\leq 1 \\\\
                (2 - \\frac{1}{||x||})(\\frac{x}{||x||}) & ||x|| > 1
            \\end{cases}

        If the order is not specified, we use the Frobenius norm, this will contract the space to a sphere of
        radius 1. If the order is L_inf (order=float("inf")), we will contract the space to a cube of side length 2.
        If using voxel based encodings such as the Hash encoder, we recommend using the L_inf norm.

        Args:
            order: Order of the norm. Default to the Frobenius norm. Must be set to None for Gaussians.

    """
    config: SceneContractionConfig

    def __init__(self, config) -> None:
        super().__init__()
        self.order = config.order

    def forward(self, positions):
        """Contract the space using the contraction proposed in MipNeRF-360."""
        mag = torch.linalg.norm(positions, ord=self.order, dim=-1)
        mask = mag >= 1
        x_new = positions.clone()
        x_new[mask] = (2 - (1 / mag[mask][..., None])) * (positions[mask] / mag[mask][..., None])

        return x_new
