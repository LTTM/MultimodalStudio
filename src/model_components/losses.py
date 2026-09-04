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
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Type, Dict, Union, List, Literal

from torchtyping import TensorType

import torch
from torch import nn

from configs.configs import InstantiateConfig
from data.datamanager import DataManager
from engine.schedulers import SchedulerConfig

LOSSES = {"L1": nn.L1Loss, "MSE": nn.MSELoss, "KL": nn.KLDivLoss}
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
    enabled: bool = True
    """Whether to enable the loss"""

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
class SaturationLossConfig(LossConfig):
    """Saturation loss configuration class"""

    _target: Type = field(default_factory=lambda: SaturationLoss)
    saturation_value: float = 1.0   # Pol: 0.9981
    """Saturation value"""
    saturation_threshold: float = 0.9999    # Pol: 0.9980
    """Saturation threshold"""

@dataclass
class SkipSaturationLossConfig(LossConfig):
    """Loss configuration class that skip the supervision of saturated pixels"""

    _target: Type = field(default_factory=lambda: SkipSaturationLoss)
    saturation_threshold: float = 0.9999
    """Saturation threshold: above it the pixel is cosidered as saturated"""

@dataclass
class LatentConsistencyLossConfig(LossConfig):
    """Latent loss configuration class"""

    _target: Type = field(default_factory=lambda: LatentConsistencyLoss)
    loss: str = "MSE"
    """Loss function"""
    first_latent_name: str = None
    """Name of the first latent vector"""
    second_latent_name: str = None
    """Name of the second latent vector"""
    latent_to_detach: Literal["first", "second", "None"] = None
    """Which latent not to supervise."""

@dataclass
class LumaConsistencyLossConfig(LatentConsistencyLossConfig):
    """Luma loss configuration class"""
    _target: Type = field(default_factory=lambda: LumaConsistencyLoss)

@dataclass
class LatentRegularizationLossConfig(LossConfig):

    _target: Type = field(default_factory=lambda: LatentRegularizationLoss)
    loss: str = "KL"
    """Loss function"""
    query_latent_name: str = None
    """Name of the first latent vector"""
    target_latent_name: str = None
    """Name of the second latent vector"""
    temperature: float = 1.0
    """Temperature for the softmax"""

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
        self.growth_factor_list = model_parameters["growth_factor_list"]

        if self.config.scheduler is not None:
            self.scheduler = self.config.scheduler.setup(
                num_iterations=num_iterations,
                growth_factor_list=self.growth_factor_list,
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

class SaturationLoss(Loss):
    """Saturation loss class. Supervise saturated pixels only util their prediction saturates itself."""

    def __init__(self, config: SaturationLossConfig, num_iterations: int, **kwargs):
        super().__init__(config, num_iterations=num_iterations)

    def forward(self, output, target, step, **kwargs):
        mask_rendering = output > 1.
        mask_gt = target > self.config.saturation_threshold
        mask = mask_rendering & mask_gt
        output = output.masked_fill(mask, self.config.saturation_value)
        return super().forward(output, target, step, **kwargs)

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

class LatentConsistencyLoss(Loss):
    """Latent loss class. Computes the L2 loss between the predicted and the target latent vectors."""

    def __init__(self, config: LatentConsistencyLossConfig, num_iterations: int, **kwargs):
        super().__init__(config, num_iterations=num_iterations)
        self.config = config

    def forward(self, first_latent, second_latent, step, **kwargs):
        """Computes the loss"""
        if self.config.latent_to_detach == "first":
            first_latent = first_latent.detach()
        elif self.config.latent_to_detach == "second":
            second_latent = second_latent.detach()
        elif self.config.latent_to_detach == "None" or self.config.latent_to_detach is None:
            pass
        else:
            raise ValueError(f"Invalid latent_to_detach value: {self.config.latent_to_detach}")
        return super().forward(first_latent, second_latent, step, **kwargs)

class LumaConsistencyLoss(LatentConsistencyLoss):
    """Latent loss class. Computes the L2 loss between the predicted and the target latent vectors."""

    def __init__(self, config: LumaConsistencyLossConfig, num_iterations: int, **kwargs):
        super().__init__(config, num_iterations=num_iterations)
        self.config = config

    def forward(self, rgb, luma, step, **kwargs):
        """Computes the loss"""
        target_luma = 0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]
        return super().forward(target_luma, luma, step, **kwargs)

class LatentRegularizationLoss(Loss):
    """Latent regularization loss class"""

    def __init__(self, config: LatentRegularizationLossConfig, num_iterations: int, **kwargs):
        super().__init__(config, num_iterations=num_iterations, reduction="batchmean")
        self.config = config

    def forward(self, output, target, step, **kwargs):
        """Computes the loss"""
        if output.shape[0] == 0 or target.shape[0] == 0:
            return torch.nan, self.config.weight
        with torch.no_grad():
            dist_target = torch.cdist(target, target, p=2) / math.sqrt(target.shape[-1])
            mask = torch.eye(dist_target.shape[0], device=dist_target.device, dtype=torch.bool)
            dist_target = dist_target[~mask].view(dist_target.shape[0], -1)
            p_target = torch.nn.functional.softmax(-dist_target / self.config.temperature, dim=-1)
        dist = torch.cdist(output, output, p=2) / math.sqrt(output.shape[-1])
        dist = dist[~mask].view(dist.shape[0], -1)
        log_p_output = torch.nn.functional.log_softmax(-dist / self.config.temperature, dim=-1)
        return super().forward(log_p_output, p_target, step, **kwargs)

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

@dataclass
class RawLossManagerConfig(LossManagerConfig):
    """
    Configuration for loss manager in the raw pipeline.
    """
    _target: Type = field(default_factory=lambda: RawLossManager)

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
            datamanager: DataManager,
            modalities_to_optimize: List[str] = None,
            **kwargs,
    ):
        self.config = config
        self.modalities = modalities
        self.datamanager = datamanager
        self.modalities_to_optimize = modalities_to_optimize if modalities_to_optimize is not None else modalities

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
            setattr(self, element, loss_config.setup(num_iterations=num_iterations, device=self.datamanager.device, **kwargs))

    def compute_loss(
            self,
            outputs: Dict[str, Union[Dict[str, TensorType], TensorType]],
            targets: Dict[str, TensorType],
            pixel_coords: Dict[str, TensorType],
            step: int,
            eval_step=False,
    ):
        """Computes all the defined losses"""
        losses = {}
        total_loss = 0.0

        radiance_losses, radiance_total_loss = self.compute_radiance_losses(outputs, targets, pixel_coords, step, eval_step)
        losses.update(radiance_losses)
        total_loss += radiance_total_loss

        if not eval_step:
            geometry_losses, geometry_total_loss = self.compute_geometry_losses(outputs, step)
            losses.update(geometry_losses)
            total_loss += geometry_total_loss

        additional_losses, additional_total_loss = self.compute_additional_losses(outputs, step)
        losses.update(additional_losses)
        total_loss += additional_total_loss

        return losses, total_loss

    def compute_radiance_losses(self, outputs, targets, pixel_coords, step, eval_step=False):
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
            if mod in self.modalities_to_optimize:
                total_loss += weight * loss
        return losses, total_loss

    def compute_geometry_losses(self, outputs, step):
        losses = {}
        total_loss = 0.0
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
            if loss_fn.config.enabled:
                if loss_name == "eikonal_loss":
                    loss, weight = loss_fn(geometry_outputs["gradients"], step)
                elif loss_name == "curvature_loss":
                    loss, weight = loss_fn(geometry_outputs["hessians"], step)
                else:
                    raise NotImplementedError
                losses[loss_name] = loss
                losses[loss_name + "_weight"] = weight
                total_loss += weight * loss
        return losses, total_loss

    def compute_additional_losses(self, outputs, step):
        losses = {}
        total_loss = 0.0
        for loss_name in self.config.additional_losses:
            loss_fn = getattr(self, loss_name)
            if loss_fn.config.enabled:
                if isinstance(loss_fn.config, LatentConsistencyLossConfig):
                    first_latent_name = loss_fn.config.first_latent_name
                    second_latent_name = loss_fn.config.second_latent_name
                    first_latents = []
                    second_latents = []
                    for mod in self.modalities:
                        mask = outputs[mod]["accumulation"].squeeze(-1)
                        if mask.sum() == 0:
                            continue
                        mask = mask > mask.max() * 0.75
                        if first_latent_name == "modalities":
                            first_latent = torch.cat([outputs[mod][m] for m in self.modalities], dim=-1)
                        else:
                            first_latent = outputs[mod][first_latent_name]
                        if second_latent_name == "modalities":
                            second_latent = torch.cat([outputs[mod][m] for m in self.modalities], dim=-1)
                        else:
                            second_latent = outputs[mod][second_latent_name]
                        first_latent = first_latent[mask]
                        second_latent = second_latent[mask]
                        first_latents.append(first_latent)
                        second_latents.append(second_latent)
                    if len(first_latents) == 0 or len(second_latents) == 0:
                        continue
                    first_latents = torch.cat(first_latents, dim=0)
                    second_latents = torch.cat(second_latents, dim=0)
                    loss, weight = loss_fn(
                        first_latents,
                        second_latents,
                        step
                    )
                elif isinstance(loss_fn.config, LatentRegularizationLossConfig):
                    query_latent_name = loss_fn.config.query_latent_name
                    target_latent_name = loss_fn.config.target_latent_name
                    query_latents = []
                    target_latents = []
                    for mod in self.modalities:
                        mask = outputs[mod]["accumulation"].squeeze(-1)
                        if mask.sum() == 0:
                            continue
                        mask = mask > mask.max() * 0.75
                        if query_latent_name == "modalities":
                            query_latent = torch.cat([outputs[mod][m] for m in self.modalities], dim=-1)
                        else:
                            query_latent = outputs[mod][query_latent_name]
                        query_latent = query_latent[mask]
                        # query_latent = query_latent[:query_latent.shape[0] // 2]
                        if query_latent.ndim == 1 or query_latent.shape[0] == 0:
                            continue
                        query_latents.append(query_latent)
                        if target_latent_name == "modalities":
                            target_latent = torch.cat([outputs[mod][m] for m in self.modalities], dim=-1)
                        else:
                            target_latent = outputs[mod][target_latent_name]
                        target_latent = target_latent[mask]
                        # target_latent = target_latent[:target_latent.shape[0] // 2]
                        if target_latent.ndim == 1 or target_latent.shape[0] == 0:
                            continue
                        target_latents.append(target_latent)
                    if len(query_latents) == 0 or len(target_latents) == 0:
                        continue
                    query_latents = torch.cat(query_latents, dim=0)
                    target_latents = torch.cat(target_latents, dim=0)
                    loss, weight = loss_fn(
                        query_latents,
                        target_latents,
                        step
                    )
                else:
                    raise NotImplementedError
                losses[loss_name] = loss
                losses[loss_name + "_weight"] = weight
                total_loss += weight * loss

        return losses, total_loss

class RawLossManager(LossManager):
    """
    Loss manager class for the raw pipeline. It is in charge of sequentially computing all the defined loss functions.
    """

    config: RawLossManagerConfig
    def __init__(
            self,
            config: RawLossManagerConfig,
            modalities: List[str],
            num_iterations: int,
            datamanager: DataManager,
            modalities_to_optimize: List[str] = None,
            **kwargs,
    ):
        super().__init__(
            config=config,
            modalities=modalities,
            num_iterations=num_iterations,
            datamanager=datamanager,
            modalities_to_optimize=modalities_to_optimize,
            **kwargs
        )
        self.config = config

    def compute_loss(
            self,
            outputs: Dict[str, Union[Dict[str, TensorType], TensorType]],
            targets: Dict[str, TensorType],
            pixel_coords: Dict[str, TensorType],
            step: int,
            eval_step=False,
    ):
        """Computes all the defined losses"""
        losses = {}
        total_loss = 0.0

        if not eval_step:
            geometry_losses, geometry_total_loss = self.compute_geometry_losses(outputs, step)
            losses.update(geometry_losses)
            total_loss += geometry_total_loss

        additional_losses, additional_total_loss = self.compute_additional_losses(outputs, step)
        losses.update(additional_losses)
        total_loss += additional_total_loss

        outputs = self.select_right_channel_per_pixel(pixel_coords, outputs)
        radiance_losses, radiance_total_loss = self.compute_radiance_losses(outputs, targets, pixel_coords, step, eval_step)
        losses.update(radiance_losses)
        total_loss += radiance_total_loss

        return losses, total_loss

    def select_right_channel_per_pixel(self, pixel_coords_per_modality, outputs):
        """Select only one channel per pixel for each modality to be supervised by the loss function."""
        mosaick_mask_per_modality = self.datamanager.train_dataset.mosaick_mask_per_modality
        for mod in self.modalities:
            mosaick_mask = mosaick_mask_per_modality[mod]
            pixel_coords = pixel_coords_per_modality[mod]
            rendered_pixels = outputs[mod][mod]
            band_mask = mosaick_mask[pixel_coords[:,1], pixel_coords[:,2]].unsqueeze(dim=1).type(torch.int64)
            outputs[mod][mod] = torch.gather(rendered_pixels, 1, band_mask)

        return outputs
