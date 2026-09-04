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
Latent estimators
"""

from dataclasses import dataclass, field
from typing import Type, Dict

import torch
from torch.nn import Module
from torchtyping import TensorType

from configs.configs import InstantiateConfig
from field_components.mlp import MLPConfig


@dataclass
class LatentEstimatorConfig(InstantiateConfig):
    """Base model Config"""

    _target: Type = field(default_factory=lambda: LatentEstimator)
    input_latent: Dict[str, int] = None
    """Input latent name and dimensions."""
    output_latent: Dict[str, int] = None
    """Output latent name and dimensions."""
    field: MLPConfig = field(default_factory=lambda: MLPConfig)
    """Module to estimate the latent output from the latent input."""

class LatentEstimator(Module):
    """Standard multimodal model"""

    def __init__(
            self,
            config: LatentEstimatorConfig,
            **kwargs
    ):
        super().__init__()
        self.config = config
        self.field = self.config.field.setup(
            input_dim=list(self.config.input_latent.values())[0],
            output_dim=list(self.config.output_latent.values())[0],
            **kwargs
        )

    def forward(self, x: TensorType[..., "input_dim"]) -> TensorType[..., "output_dim"]:
        """Estimate the latent code from the input latent code."""
        return self.field(x.detach())
