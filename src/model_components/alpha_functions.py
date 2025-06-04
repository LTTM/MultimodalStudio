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
Functions to compute alpha terms for volume rendering
"""

from typing import Type, List, Dict
from dataclasses import dataclass, field
from torchtyping import TensorType

import torch
from torch import nn
from torch.nn import functional as F, Parameter

from cameras.rays import RaySamples
from configs.configs import InstantiateConfig
from engine.callbacks import TrainingCallbackAttributes, TrainingCallback, TrainingCallbackLocation
from field_components.single_variance import SingleVarianceNetwork

@dataclass
class AlphaConfig(InstantiateConfig):
    """Base alpha function config."""

    _target: Type = field(default_factory=lambda: Alpha)
    s_init: float = 0.3
    """Initial value for the single variance network."""

@dataclass
class NeuSAlphaConfig(AlphaConfig):
    """NeuS alpha function config."""

    _target: Type = field(default_factory=lambda: NeuSAlpha)
    anneal_end: int = 10000
    """Number of iterations to anneal the cos value. If 0, no annealing is performed."""

class Alpha(nn.Module):
    """Base class for alpha functions."""

    config: AlphaConfig

    def __init__(self, config):
        super().__init__()
        self.deviation_network = SingleVarianceNetwork(init_val=config.s_init)
        self._cos_anneal_ratio = 1.0

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        """Returns the training callbacks for the alpha function."""
        callbacks = []
        return callbacks

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Returns the parameter groups for the optimizer."""
        param_groups = {
            "single_variance_network": list(self.deviation_network.parameters()),
        }
        return param_groups

class NeuSAlpha(Alpha):  # alpha * Laplace(loc=0, scale=beta).cdf(-sdf)
    """Volume rendering alphas computed as in NeuS"""

    config: NeuSAlphaConfig

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.anneal_end = self.config.anneal_end

    def forward(self, ray_samples: RaySamples, sdf: TensorType[..., 1], gradients: TensorType[..., 1]):
        """compute alpha from sdf as in NeuS"""

        inv_s = self.deviation_network.get_variance()  # Single parameter

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

        prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
        next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)

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
        """Returns the training callbacks for the alpha function."""
        callbacks = super().get_training_callbacks(training_callback_attributes)
        # anneal for cos in NeuS
        if self.anneal_end > 0:

            def set_anneal(step):
                anneal = min([1.0, step / self.anneal_end])
                self.set_cos_anneal_ratio(anneal)

            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=set_anneal,
                )
            )
        return callbacks
