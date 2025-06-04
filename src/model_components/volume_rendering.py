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
Collection of volume rendering models
"""

from dataclasses import dataclass, field
from typing import Type, List, Dict

import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
from torchtyping import TensorType

from cameras.rays import RaySamples
from configs.configs import InstantiateConfig
from engine.callbacks import TrainingCallbackAttributes, TrainingCallback, TrainingCallbackLocation
from field_components.single_variance import SingleVarianceNetwork

@dataclass
class DensityConfig(InstantiateConfig):
    """Base density function config"""
    _target: Type = field(default_factory=lambda: Density)
    """Target class for instantiation"""
    init_val: float = 0.3
    """Initial value for variance network"""

@dataclass
class NeuSDensityConfig(DensityConfig):
    """NeuS density function config"""
    _target: Type = field(default_factory=lambda: NeuSDensity)
    """Target class for instantiation"""

@dataclass
class LaplaceDensityConfig(DensityConfig):
    """Laplace density function config"""
    _target: Type = field(default_factory=lambda: LaplaceDensity)
    """Target class for instantiation"""
    beta: float = 0.1
    """Initial value for beta parameter"""
    beta_min: float = 0.0001
    """Minimum value for beta parameter"""

@dataclass
class VolumeRenderingConfig(InstantiateConfig):
    """Base volume rendering config"""
    _target: Type = field(default_factory=lambda: VolumeRendering)
    """Target class for instantiation"""
    density_fn: DensityConfig = field(default_factory=lambda: DensityConfig)
    """Density function configuration"""

@dataclass
class NeuSVolumeRenderingConfig(VolumeRenderingConfig):
    """NeuS volume rendering config"""
    _target: Type = field(default_factory=lambda: NeuSVolumeRendering)
    """Target class for instantiation"""
    anneal_end_ratio: int = 0.05
    """Annealing end ratio with respect to total number of iterations"""

@dataclass
class VolSDFVolumeRenderingConfig(VolumeRenderingConfig):
    """VolSDF volume rendering config"""
    _target: Type = field(default_factory=lambda: VolSDFVolumeRendering)
    """Target class for instantiation"""

class Density(nn.Module):
    """Density model

    Args:
        config: Density configuration
    """

    def __init__(self, config: DensityConfig):
        super().__init__()
        self.config = config

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Returns the parameter groups for the optimizer."""
        param_groups = {"density_fn": list(self.parameters())}
        return param_groups

class NeuSDensity(Density):
    """NeuS density model"""

    def __init__(self, config: NeuSDensityConfig):
        super().__init__(config)
        self.variance_network = SingleVarianceNetwork(init_val=self.config.init_val)

    def forward(self, sdf: TensorType[..., 1]):
        """Compute s-density from sdf as in NeuS paper"""
        s = self.variance_network.get_inv_variance()
        density = (s * torch.exp(-sdf * s)) / (1 + torch.exp(-sdf * s) ** 2)
        return density

class LaplaceDensity(Density):
    """VolSDF density model"""

    def __init__(self, config: LaplaceDensityConfig):
        super().__init__(config)
        self.beta = Parameter(torch.tensor(config.beta, requires_grad=True))
        self.beta_min = self.config.beta_min

    def get_beta(self):
        """Get beta value"""
        return self.beta.abs() + self.beta_min

    def forward(self, sdf: TensorType[..., 1], **kwargs):
        """Compute density from sdf as in VolSDF paper"""
        beta = self.get_beta()
        density = (0.5 + 0.5 * sdf.sign() * torch.expm1(-sdf.abs() / beta)) / beta
        return density

class VolumeRendering(nn.Module):
    """Volume rendering base model

    Args:
        config: Density configuration
    """

    def __init__(self, config: DensityConfig):
        super().__init__()
        self.config = config
        self.density_fn = self.config.density_fn.setup()

    def get_density(self, sdf: TensorType[..., 1]):
        """Compute density from sdf"""
        density = self.density_fn(sdf)
        return density

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        """Returns the training callbacks for the density function"""
        callbacks = []
        return callbacks

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Returns the parameter groups for the optimizer"""
        density_param_group = self.density_fn.get_param_groups()
        return density_param_group

class NeuSVolumeRendering(VolumeRendering):
    """NeuS volume rendering model

    Args:
        config: NeuS density configuration
    """

    def __init__(self, config: NeuSDensityConfig):
        super().__init__(config)

    def forward(self, ray_samples: RaySamples, sdf: TensorType[..., 1], gradients: TensorType[..., 3]):
        """Compute weights from sdf as in NeuS"""
        alphas = self.get_alphas(ray_samples, sdf, gradients)
        weights = self.get_weights(alphas)
        return weights

    def get_weights(self, alphas: TensorType[..., 1]):
        """compute weights from alphas as in NeuS"""
        transmittance = torch.cumprod(
            torch.cat([torch.ones((alphas.shape[0], 1), device=alphas.device), 1.0 - alphas + 1e-7], 1), 1
        )
        weights = (alphas * transmittance[:, :-1]).unsqueeze(-1)
        return weights

    def get_alphas(self, ray_samples: RaySamples, sdf: TensorType[..., 1], gradients: TensorType[..., 3]):
        """Compute alpha from sdf as in NeuS"""

        s = self.density_fn.variance_network.get_inv_variance()  # Single parameter

        true_cos = (ray_samples.frustums.directions * gradients).sum(-1, keepdim=True)

        # anneal as NeuS
        cos_anneal_ratio = self._cos_anneal_ratio

        # "cos_anneal_ratio" grows from 0 to 1 in the beginning training iterations. The anneal strategy below makes
        # the cos value "not dead" at the beginning training iterations, for better convergence.
        iter_cos = -(
                F.relu(-true_cos * 0.5 + 0.5) * (1.0 - cos_anneal_ratio) + F.relu(-true_cos) * cos_anneal_ratio
        )  # always non-positive

        # Estimate signed distances at section points
        estimated_next_sdf = sdf + iter_cos * ray_samples.deltas * 0.5
        estimated_prev_sdf = sdf - iter_cos * ray_samples.deltas * 0.5

        prev_cdf = torch.sigmoid(estimated_prev_sdf * s)
        next_cdf = torch.sigmoid(estimated_next_sdf * s)

        p = prev_cdf - next_cdf
        c = prev_cdf

        alpha = ((p + 1e-5) / (c + 1e-5)).clip(0.0, 1.0).squeeze(dim=-1)

        return alpha

    def set_cos_anneal_ratio(self, anneal: float) -> None:
        """Set the anneal value for the proposal network."""
        self._cos_anneal_ratio = anneal

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        """Returns the training callbacks for the density function"""
        callbacks = super().get_training_callbacks(training_callback_attributes)
        # anneal for cos in NeuS
        if self.config.anneal_end_ratio > 0:

            def set_anneal(step):
                anneal_end = int(training_callback_attributes.trainer.max_num_iterations * self.config.anneal_end_ratio)
                anneal = min([1.0, step / anneal_end])
                self.set_cos_anneal_ratio(anneal)

            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=set_anneal,
                )
            )
        return callbacks

class VolSDFVolumeRendering(VolumeRendering):
    """VolSDF volume rendering model

    Args:
        config: Laplace density configuration
    """

    def __init__(self, config: VolSDFVolumeRenderingConfig):
        super().__init__(config)

    def forward(self, ray_samples: RaySamples, sdf: TensorType[..., 1], **kwargs):
        """Compute weights from sdf as in VolSDF"""
        density = self.get_density(sdf)
        weights = self.get_weights(ray_samples, density)
        return weights

    def get_weights(self, ray_samples: RaySamples, density: TensorType[..., 1]):
        """Compute weights from density as in VolSDF"""
        delta_density = ray_samples.deltas * density
        alphas = 1 - torch.exp(-delta_density)

        transmittance = torch.cumsum(delta_density[..., :-1, :], dim=-2)
        transmittance = torch.cat(
            [torch.zeros((*transmittance.shape[:1], 1, 1), device=density.device), transmittance], dim=-2
        )
        transmittance = torch.exp(-transmittance)  # [..., "num_samples"]
        weights = alphas * transmittance  # [..., "num_samples"]
        return weights
