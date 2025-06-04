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
Collection of Loss functions.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Type, Dict, Union, List
from torchtyping import TensorType

import numpy as np
import torch
from torch import nn

from configs.configs import InstantiateConfig
from engine.schedulers import SchedulerConfig

LOSSES = {"L1": nn.L1Loss, "MSE": nn.MSELoss}
EPS = 1.0e-7

@dataclass
class LossConfig(InstantiateConfig):
    """General loss configuration class"""

    _target: Type = field(default_factory=lambda: Loss)
    loss: str = "L1"
    """Loss function"""
    weight: float = 1.0
    """Loss weight"""
    scheduler: SchedulerConfig = None
    """Loss scheduler"""
    per_channel_probability: List[float] = None
    """Probability of supervising each channel. If None, every channel is supervised"""

@dataclass
class EikonalLossConfig(LossConfig):
    """Eikonal loss configuration class"""

    _target: Type = field(default_factory=lambda: EikonalLoss)
    loss: str = "MSE"
    """Loss function"""
    weight: float = 0.1
    """Loss weight"""
    scheduler: SchedulerConfig = None
    """Loss scheduler"""

@dataclass
class CurvatureLossConfig(LossConfig):
    """Curvature loss configuration class"""

    _target: Type = field(default_factory=lambda: CurvatureLoss)
    loss: str = "L1"
    """Loss function"""
    weight: float = 5e-4
    """Loss weight"""
    scheduler: SchedulerConfig = None
    """Loss scheduler"""

@dataclass
class SkipSaturationLossConfig(LossConfig):
    """Loss configuration class that skip the supervision of saturated pixels"""

    _target: Type = field(default_factory=lambda: SkipSaturationLoss)
    saturation_threshold: float = 0.9999
    """Saturation threshold: above it the pixel is cosidered as saturated"""

class Loss(nn.Module):
    """General loss class"""

    def __init__(self, config: LossConfig, reduction: str='mean', **kwargs):
        super().__init__()
        self.config = config
        self.loss_fn = globals()["LOSSES"][self.config.loss](reduction=reduction)
        if self.config.scheduler is not None and "num_iterations" in kwargs:
            self.scheduler = self.config.scheduler.setup(num_iterations=kwargs["num_iterations"])
        if self.config.per_channel_probability is not None:
            self.config.per_channel_probability = torch.tensor(self.config.per_channel_probability)

    def select_channel(self, output, target):
        """Randomly select the channel to supervise"""
        assert len(self.config.per_channel_probability) == output.shape[1]
        indexes = torch.multinomial(self.config.per_channel_probability, output.shape[0], replacement=True).view(-1, 1)
        output = output[torch.arange(output.shape[0]), indexes.view(-1, 1)]
        target = target[torch.arange(target.shape[0]), indexes.view(-1, 1)]
        return output, target

    def forward(self, *args, **kwargs):
        """Compute the loss"""
        output, target, step = args
        weight = self.config.weight
        if self.config.scheduler is not None:
            weight *= self.scheduler.get_update_factor(step)
        if self.config.per_channel_probability is not None:
            output, target = self.select_channel(output, target)
        return self.loss_fn(output, target), weight

class EikonalLoss(Loss):
    """Eikonal loss class"""

    def __init__(self, config: EikonalLossConfig, num_iterations: int, **kwargs):
        super().__init__(config, num_iterations=num_iterations)

    def forward(self, gradients, step):
        grad_norm = torch.norm(gradients, 2, dim=-1)
        loss = self.loss_fn(grad_norm, torch.ones_like(grad_norm, device=grad_norm.device))
        weight = self.config.weight
        if self.config.scheduler is not None:
            weight *= self.scheduler.get_update_factor(step)
        return loss, weight

class CurvatureLoss(Loss):
    """Curvature loss class"""

    def __init__(self, config: CurvatureLossConfig, num_iterations: int, **kwargs):
        super().__init__(config)
        self.model = kwargs.get("model")
        model_parameters = self.model.get_model_parameters()

        steps_per_level = int(num_iterations * model_parameters["steps_per_level_ratio"])
        self.steps_per_level = min(steps_per_level, int(num_iterations /  model_parameters["num_levels"]))
        self.grow_factor = np.exp((np.log(model_parameters["max_res"]) - np.log(model_parameters["min_res"])) / \
                                  (model_parameters["num_levels"] - 1))

        if self.config.scheduler is not None:
            self.scheduler = self.config.scheduler.setup(
                num_iterations=num_iterations,
                grow_factor=self.grow_factor,
                level_init=model_parameters["level_init"],
                num_levels=model_parameters["num_levels"],
                steps_per_level=self.steps_per_level,
            )

    def forward(self, hessians, step):
        """Comoutes the curvature loss"""
        laplacian = hessians.sum(dim=-1)
        loss = self.loss_fn(laplacian, torch.zeros_like(laplacian, device=laplacian.device))
        weight = self.config.weight
        if self.config.scheduler is not None:
            weight *= self.scheduler.get_update_factor(step)
        return loss, weight

class SkipSaturationLoss(Loss):
    """SkipSaturation loss class. Do not compute loss to pixels with saturation."""

    def __init__(self, config: SkipSaturationLossConfig, num_iterations: int, **kwargs):
        super().__init__(config, num_iterations=num_iterations)

    def forward(self, output, target, step, **kwargs):
        """Computes the loss only on non-saturated pixels"""
        mask = target > self.config.saturation_threshold
        if mask.any():
            value = target[mask].flatten()[0]
            output = output.masked_fill(mask, value)
        return super().forward(output, target, step, **kwargs)

@dataclass
class LossManagerConfig(InstantiateConfig):
    """
    Configuration for loss manager.
    """

    _target: Type = field(default_factory=lambda: LossManager)
    radiance_losses: Dict[str, Union[str, LossConfig]] = field(default_factory=lambda: {"rgb": "L1Loss"})
    """Radiance loss function per modality"""
    geometry_losses: Dict[str, LossConfig] = field(default_factory=lambda: {"eikonal_loss": EikonalLossConfig})
    """Geometry loss functions"""
    additional_losses: Dict[str, LossConfig] = field(default_factory=lambda: {})
    """Additional loss functions"""

class LossManager:
    """
    Loss manager class. It is in charge of sequentially computing all the defined loss functions.
    """

    config: LossManagerConfig

    def __init__(
            self,
            config: LossManagerConfig,
            modalities: List[str],
            num_iterations: int,
            **kwargs,
    ):
        self.config = config
        self.modalities = modalities

        for element in self.modalities:
            loss_config = self.config.radiance_losses[element]
            if isinstance(loss_config, str):
                loss_class = globals()[loss_config]()
            else:
                loss_class = self.config.radiance_losses[element].setup(num_iterations=num_iterations, **kwargs)
            setattr(self, element, loss_class)

        for element in self.config.geometry_losses:
            loss_config = self.config.geometry_losses[element]
            setattr(self, element, loss_config.setup(num_iterations=num_iterations, **kwargs))

        for element in self.config.additional_losses:
            loss_config = self.config.additional_losses[element]
            setattr(self, element, loss_config.setup(num_iterations=num_iterations, **kwargs))

    def compute_loss(
            self,
            outputs: Dict[str, Dict[str, TensorType]],
            targets: Dict[str, TensorType],
            pixel_coords: Dict[str, TensorType],
            step: int,
            eval_step=False,
    ):
        """Computes all the defined losses"""
        losses = {}
        total_loss = 0.0
        for mod in self.modalities:
            output = outputs[mod][mod]
            target = targets[mod]
            loss_func = getattr(self, mod)
            loss, weight = loss_func(output, target, step, pixel_coords=pixel_coords, eval_step=eval_step)

            losses[mod] = loss
            if weight != 1:
                losses[mod + "_weight"] = weight
            total_loss += weight * loss

        if not eval_step:
            geometry_outputs = defaultdict(list)
            for mod in self.modalities:
                for element in outputs[mod]:
                    if element in ['gradients', 'hessians']:
                        if outputs[mod][element] is not None:
                            geometry_outputs[element].append(outputs[mod][element])
                        else:
                            geometry_outputs[element] = None

            for element in geometry_outputs:
                geometry_outputs[element] = torch.cat(geometry_outputs[element], dim=0) \
                    if geometry_outputs[element] \
                    else None

            for loss_name in self.config.geometry_losses:
                loss_fn = getattr(self, loss_name)
                if loss_name == "eikonal_loss":
                    loss, weight = loss_fn(geometry_outputs["gradients"], step)
                elif loss_name == "curvature_loss":
                    loss, weight = loss_fn(geometry_outputs["hessians"], step)
                else:
                    raise NotImplementedError
                losses[loss_name] = loss
                losses[loss_name + "_weight"] = weight
                total_loss += weight * loss

        for loss_name in self.config.additional_losses:
            pass

        return losses, total_loss
