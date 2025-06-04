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

"""Scheduler Classes"""

from dataclasses import dataclass, field
from typing import Any, Optional, Type, List

import numpy as np
from torch.optim import Optimizer, lr_scheduler

from configs.configs import InstantiateConfig


@dataclass
class SchedulerConfig(InstantiateConfig):
    """Basic scheduler config with self-defined exponential decay schedule"""
    _target: Type = field(default_factory=lambda: ExponentialDecaySchedule)
    lr_final: float = 0.000005
    max_steps: int = 1000000

    # (Comment from NeRFStudio authors): Somehow make this more generic. I don't like the idea of overriding the setup
    # function but also not sure how to go about passing things into predefined torch objects.
    def setup(self, optimizer=None, lr_init=None, **kwargs) -> Any:
        """Returns the instantiated object using the config."""
        return self._target(optimizer, lr_init, self.lr_final, self.max_steps)

class ExponentialDecaySchedule(lr_scheduler.LambdaLR):
    """Exponential learning rate decay function.
    See https://github.com/google-research/google-research/blob/
    fd2cea8cdd86b3ed2c640cbe5561707639e682f3/jaxnerf/nerf/utils.py#L360
    for details.

    Args:
        optimizer: The optimizer to update.
        lr_init: The initial learning rate.
        lr_final: The final learning rate.
        max_steps: The maximum number of steps.
        lr_delay_steps: The number of steps to delay the learning rate.
        lr_delay_mult: The multiplier for the learning rate after the delay.
    """

    config: SchedulerConfig

    def __init__(self, optimizer, lr_init, lr_final, max_steps, lr_delay_steps=0, lr_delay_mult=1.0) -> None:
        def func(step):
            if lr_delay_steps > 0:
                delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                    0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
                )
            else:
                delay_rate = 1.0
            t = np.clip(step / max_steps, 0, 1)
            log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
            multiplier = (
                log_lerp / lr_init
            )  # divided by lr_init because the multiplier is with the initial learning rate
            return delay_rate * multiplier

        super().__init__(optimizer, lr_lambda=func)


class DelayerScheduler(lr_scheduler.LambdaLR):
    """Starts with a flat lr schedule until it reaches N epochs then applies a given scheduler"""

    def __init__(
        self,
        optimizer: Optimizer,
        lr_init,  # pylint: disable=unused-argument
        lr_final,  # pylint: disable=unused-argument
        max_steps,  # pylint: disable=unused-argument
        delay_epochs: int = 500,
        after_scheduler: Optional[lr_scheduler.LambdaLR] = None,
    ) -> None:
        def func(step):
            if step > delay_epochs:
                if after_scheduler is not None:
                    multiplier = after_scheduler.lr_lambdas[0](step - delay_epochs)  # type: ignore
                    return multiplier
                return 1.0
            return 0.0

        super().__init__(optimizer, lr_lambda=func)


class DelayedExponentialScheduler(DelayerScheduler):
    """Delayer Scheduler with an Exponential Scheduler initialized afterwards."""

    def __init__(
        self,
        optimizer: Optimizer,
        lr_init,
        lr_final,
        max_steps,
        delay_epochs: int = 200,
    ):
        after_scheduler = ExponentialDecaySchedule(
            optimizer,
            lr_init,
            lr_final,
            max_steps,
        )
        super().__init__(optimizer, lr_init, lr_final, max_steps, delay_epochs, after_scheduler=after_scheduler)


@dataclass
class MultiStepSchedulerConfig(InstantiateConfig):
    """Basic scheduler config with self-defined exponential decay schedule"""

    _target: Type = field(default_factory=lambda: lr_scheduler.MultiStepLR)
    max_steps: int = 1000000

    def setup(self, optimizer=None, lr_init=None, **kwargs) -> Any:
        """Returns the instantiated object using the config."""
        return self._target(
            optimizer,
            milestones=[self.max_steps // 2, self.max_steps * 3 // 4, self.max_steps * 9 // 10],
            gamma=0.33,
        )


@dataclass
class ExponentialSchedulerConfig(InstantiateConfig):
    """Basic scheduler config with self-defined exponential decay schedule"""

    _target: Type = field(default_factory=lambda: lr_scheduler.ExponentialLR)
    decay_rate: float = 0.1
    max_steps: int = 1000000

    def setup(self, optimizer=None, lr_init=None, **kwargs) -> Any:
        """Returns the instantiated object using the config."""
        return self._target(
            optimizer,
            self.decay_rate ** (1.0 / self.max_steps),
        )

@dataclass
class SchedulerConfig(InstantiateConfig):
    """Basic scheduler config"""

    _target: Type = field(default_factory=lambda: Scheduler)

@dataclass
class MaskedSchedulerConfig(SchedulerConfig):
    _target: Type = field(default_factory=lambda: MaskedScheduler)
    scheduler: Optional[SchedulerConfig] = None

    learning_factor: Optional[float] = 1.0

    mask_ratio: float = 0.0

@dataclass
class MultiStepWarmupSchedulerConfig(SchedulerConfig):
    """Basic scheduler config with self-defined exponential decay schedule"""

    _target: Type = field(default_factory=lambda: MultiStepWarmupScheduler)
    warm_up_ratio: float = 0.1
    """percentage of total steps to warm up"""
    milestones: List[float] = field(default_factory=lambda: [0.5, 0.75, 0.9])
    """milestones for learning rate decay"""
    gamma: float = 0.33

@dataclass
class NeuSSchedulerConfig(SchedulerConfig):
    """Basic scheduler config with self-defined cosine decay schedule"""

    _target: Type = field(default_factory=lambda: NeuSScheduler)
    warm_up_ratio: float = 0.1
    """percentage of total steps to warm up"""
    learning_rate_alpha: float = 0.01

@dataclass
class CosineRaiseSchedulerConfig(SchedulerConfig):
    """Basic scheduler config with self-defined cosine raise schedule"""

    _target: Type = field(default_factory=lambda: CosineRaiseScheduler)
    learning_rate_alpha: float = 0.01

    saturation_ratio: float = 0.5
    """percentage of total steps before value stays constant"""

@dataclass
class CurvatureLossWarmUpSchedulerConfig(SchedulerConfig):
    """Scheduler config for the curvature loss weight"""

    _target: Type = field(default_factory=lambda: CurvatureLossWarmUpScheduler)
    warm_up_ratio: float = 0.1
    """percentage of total steps to warm up"""

class OptimizerScheduler(lr_scheduler.LambdaLR):
    """General optimizer scheduler"""
    def __init__(self, optimizer, lr_lambda) -> None:
        super().__init__(optimizer, lr_lambda=lr_lambda)

class Scheduler:
    """General scheduler"""
    def __init__(self, config: SchedulerConfig, optimizer=None, func=None) -> None:
        self.config = config
        if optimizer is not None:
            self.optimizer_scheduler = OptimizerScheduler(optimizer, lr_lambda=func)

    def step(self):
        self.optimizer_scheduler.step()
    def get_last_lr(self):
        return self.optimizer_scheduler.get_last_lr()[0]
    def state_dict(self):
        return self.optimizer_scheduler.state_dict()
    def load_state_dict(self, state_dict):
        self.optimizer_scheduler.load_state_dict(state_dict)

class MaskedScheduler(Scheduler):

    def __init__(
            self,
            config: MaskedSchedulerConfig,
            num_iterations: int,
            optimizer=None,
    ) -> None:

        self.config = config
        self.scheduler = config.scheduler.setup(num_iterations=num_iterations, optimizer=optimizer) if config.scheduler is not None else None

        def func(step):
            if step < self.config.mask_ratio * num_iterations:
                learning_factor = 0.0
            else:
                learning_factor = self.scheduler.func(step) if self.scheduler is not None else self.config.learning_factor
            return learning_factor
        self.get_update_factor = func
        super().__init__(config, optimizer=optimizer, func=func)

class MultiStepWarmupScheduler(Scheduler):
    """Starts with a flat lr schedule until it reaches N epochs then applies a given scheduler"""

    def __init__(
            self,
            config: MultiStepWarmupSchedulerConfig,
            num_iterations: int,
            optimizer=None,
    ) -> None:

        self.config = config
        self.warm_up_end = int(num_iterations * self.config.warm_up_ratio)
        def func(step):
            if step < self.warm_up_end:
                learning_factor = step / self.warm_up_end
            else:
                index = np.searchsorted(self.config.milestones, step / num_iterations, side='left')
                learning_factor = self.config.gamma ** index
            return learning_factor

        self.get_update_factor = func
        super().__init__(config, optimizer=optimizer, func=func)


class NeuSScheduler(Scheduler):
    """Starts with a flat lr schedule until it reaches N epochs then applies a given scheduler"""

    def __init__(
            self,
            config: NeuSSchedulerConfig,
            num_iterations: int,
            optimizer=None,
    ) -> None:
        self.config = config
        self.warm_up_end = int(num_iterations * self.config.warm_up_ratio)
        def func(step):
            if step < self.warm_up_end:
                learning_factor = step / self.warm_up_end
            else:
                alpha = self.config.learning_rate_alpha
                progress = (step - self.warm_up_end) / (num_iterations - self.warm_up_end)
                learning_factor = (np.cos(np.pi * progress) + 1.0) * 0.5 * (1 - alpha) + alpha
            return learning_factor

        self.get_update_factor = func
        super().__init__(config, optimizer=optimizer, func=func)

class CosineRaiseScheduler(Scheduler):
    """Apply a cosine raise scheduler"""

    def __init__(
            self,
            config: CosineRaiseSchedulerConfig,
            num_iterations: int,
            optimizer=None,
    ) -> None:
        self.config = config
        self.saturation_start = int(num_iterations * self.config.saturation_ratio)

        def func(step):
            if step < self.saturation_start:
                alpha = self.config.learning_rate_alpha
                progress = step / self.saturation_start
                learning_factor = (-1 * np.cos(np.pi * progress) + 1.0) * 0.5 * (1 - alpha) + alpha
            else:
                learning_factor = 1.0
            return learning_factor

        self.get_update_factor = func
        super().__init__(config, optimizer=optimizer, func=func)

class CurvatureLossWarmUpScheduler(Scheduler):
    """Update curvature loss weight according to the eps value used in the coarse-to-fine feature grid."""

    def __init__(
            self,
            config: CurvatureLossWarmUpSchedulerConfig,
            num_iterations: int,
            grow_factor: float,
            level_init: int,
            num_levels: int,
            steps_per_level: int,
            optimizer=None,
    ) -> None:
        self.config = config
        self.warm_up_end = int(num_iterations * config.warm_up_ratio)

        def func(step):
            if step < self.warm_up_end:
                learning_factor = step / self.warm_up_end
            else:
                level = int(step / steps_per_level) + 1
                level = max(level, level_init)
                level = min(level, num_levels)
                learning_factor = np.reciprocal(grow_factor ** (level - 1))
            return learning_factor

        self.get_update_factor = func
        super().__init__(config, optimizer=optimizer, func=func)