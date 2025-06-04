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
Pipeline for training and evaluating the model with raw data.
"""

from dataclasses import dataclass, field
from typing import Type

import torch
from lightning import Fabric
from rich.console import Console

from pipelines.base_pipeline import BasePipelineConfig, BasePipeline
from utils import profiler
from utils.eval_utils import compute_metrics
from utils.misc import check_step

CONSOLE = Console(width=120)

@dataclass
class RawPipelineConfig(BasePipelineConfig):
    """Raw Pipeline Config"""
    _target: Type = field(default_factory=lambda: RawPipeline)

class RawPipeline(BasePipeline):
    """
    Raw Pipeline for training and evaluating the model with raw data.
    """

    def __init__(
            self,
            config: RawPipelineConfig,
            fabric: Fabric, trainer_config,
            output_dir: str,
            checkpoint_dir: str,
            mixed_precision: bool,
    ):
        super().__init__(
            config=config,
            fabric=fabric,
            trainer_config=trainer_config,
            output_dir=output_dir,
            checkpoint_dir=checkpoint_dir,
            mixed_precision=mixed_precision
        )

    @profiler.time_function
    def train_step(self, step: int):
        """Performs a training step and returns the losses and metrics."""
        (pixel_coords, pixels) = next(self.datamanager.iter_train_dataloader)
        ray_bundles = self.datamanager.train_ray_generator(pixel_coords)
        outputs = self.model(ray_bundles)
        outputs = self.select_right_channel_per_pixel(pixel_coords, outputs)
        losses, total_loss = self.loss_manager.compute_loss(outputs, pixels, pixel_coords, step)
        metrics = compute_metrics(outputs, pixels, modalities=self.datamanager.modalities)

        self.optimizers.zero_grad_all()
        self.fabric.backward(total_loss)
        self.clip_gradients(max_norm=2.0)
        self.optimizers.optimizer_step_all()
        self.optimizers.scheduler_step_all(step)

        return losses, total_loss, metrics

    def eval_step(self, step):
        """Performs an evaluation step and returns the losses and metrics."""
        self.set_eval()
        losses, total_loss, metrics = None, None, None
        if check_step(step, self.trainer_config.steps_per_eval_batch):
            (pixel_coords, pixels) = next(self.datamanager.iter_eval_dataloader)
            ray_bundles = self.datamanager.eval_ray_generator(pixel_coords)

            with torch.no_grad():
                outputs = self.model.module(ray_bundles)

            outputs = self.select_right_channel_per_pixel(pixel_coords, outputs, eval_step=True)
            losses, total_loss = self.loss_manager.compute_loss(outputs, pixels, pixel_coords, step, eval_step=True)
            metrics = compute_metrics(outputs, pixels, modalities=self.datamanager.modalities, eval_step=True)

        if check_step(step, self.trainer_config.steps_per_eval_image, skip_first=True) and self.global_rank == 0:
            self.evaluator.render_train_view(step)
            self.evaluator.render_eval_view(step)
        if check_step(step, self.trainer_config.steps_per_eval_all_images, skip_first=True) and self.global_rank == 0:
            self.evaluator.render_all_eval_views(step)
        if check_step(step, self.trainer_config.steps_per_export_mesh, skip_first=True) and self.global_rank == 0:
            self.evaluator.export_mesh(step)
        if check_step(step, self.trainer_config.steps_per_export_poses, skip_first=False) and self.global_rank == 0:
            self.evaluator.export_poses(step)

        self.set_train()
        return losses, total_loss, metrics

    def select_right_channel_per_pixel(self, pixel_coords_per_modality, outputs, eval_step=False):
        """Select only one channel per pixel for each modality to be supervised by the loss function."""
        mosaick_mask_per_modality = self.datamanager.train_dataset.mosaick_mask_per_modality
        for mod in self.datamanager.modalities:
            mosaick_mask = mosaick_mask_per_modality[mod]
            pixel_coords = pixel_coords_per_modality[mod]
            rendered_pixels = outputs[mod][mod]
            band_mask = mosaick_mask[pixel_coords[:,1], pixel_coords[:,2]].unsqueeze(dim=1).type(torch.int64)
            outputs[mod][mod] = torch.gather(rendered_pixels, 1, band_mask)

        return outputs
