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
Standard pipeline for training and evaluation. It handles the training with demosaicked frames.
It instantiates the datamanager, model, optimizers, loss manager, and evaluator.
"""

from dataclasses import dataclass, field
from typing import Type, Dict, Any

import torch
from lightning import Fabric
from rich.console import Console

from configs.configs import InstantiateConfig
from data.datamanager import DataManagerConfig
from engine.callbacks import TrainingCallbackAttributes
from engine.evaluator import EvaluatorConfig
from engine.optimizers import Optimizers, OptimizerConfig
from engine.schedulers import SchedulerConfig
from model_components.losses import LossManagerConfig
from models.base_model import BaseModelConfig
from utils import profiler
from utils.eval_utils import compute_metrics
from utils.misc import check_step

CONSOLE = Console(width=120)

@dataclass
class BasePipelineConfig(InstantiateConfig):
    """Base pipeline config"""
    _target: Type = field(default_factory=lambda: BasePipeline)
    datamanager: DataManagerConfig = field(default_factory=lambda: DataManagerConfig)
    """Datamanager configuration"""
    model: BaseModelConfig = field(default_factory=lambda: BaseModelConfig)
    """Model configuration"""
    optimizers: Dict[str, Any] = field(default_factory=lambda: dict(
        {
            "fields": {
                "optimizer": OptimizerConfig(),
                "scheduler": SchedulerConfig(),
            }
        }
    ))
    """Dictionary of optimizer groups and their schedulers"""
    loss_manager: LossManagerConfig = field(default_factory=lambda: LossManagerConfig)
    """Config for loss manager"""
    evaluator: EvaluatorConfig = field(default_factory=lambda: EvaluatorConfig)
    """Config for evaluation phase"""

class BasePipeline:
    """Standard pipeline for demosaicked frames"""

    def __init__(
            self,
            config: BasePipelineConfig,
            fabric: Fabric,
            trainer_config,
            output_dir: str,
            checkpoint_dir: str,
            mixed_precision: bool,
    ):
        self.config = config
        self.trainer_config = trainer_config
        self.fabric = fabric
        self.output_dir = output_dir
        self.checkpoint_dir = checkpoint_dir
        self.mixed_precision = mixed_precision
        self.global_rank = self.fabric.global_rank

    def setup(self):
        """Initialize the pipeline components"""
        # Initialize DataManager
        with self.fabric.init_module():
            self.datamanager = self.config.datamanager.setup(
                data_dir=self.trainer_config.data_dir,
                fabric=self.fabric,
                full_view_ids=self.trainer_config.view_ids,
            )

        # Initialize model
        scene_box = self.datamanager.train_dataset.scene_box
        with self.fabric.init_module():
            self.model = self.config.model.setup(
                scene_box=scene_box,
                modalities=self.datamanager.modalities,
            )

        # Initialize optimizers and schedulers
        optimizers, schedulers = self.initialize_optimizers()
        self.model, model_optimizers = self.fabric_setup_model(optimizers=optimizers)
        self.datamanager, datamanager_optimizer = self.fabric_setup_datamanager(optimizers=optimizers)
        model_optimizers.update(datamanager_optimizer)
        self.optimizers = Optimizers(optimizers=model_optimizers, schedulers=schedulers)

        # Initialize loss manager
        self.loss_manager = self.config.loss_manager.setup(
            modalities=self.datamanager.modalities,
            num_iterations=self.trainer_config.max_num_iterations,
            model=self.model,
            datamanager=self.datamanager,
        )

        # Setup evaluator
        self.evaluator = self.config.evaluator.setup(
            pipeline=self,
            scene_box=scene_box,
            w2gt=self.datamanager.train_dataset.w2gt,
            output_path=self.output_dir,
        )

        # callbacks
        self.callbacks = self.model.get_training_callbacks(
            TrainingCallbackAttributes(
                trainer=self.trainer_config,  # type: ignore
                model=self.model.config,
            )
        )

    @profiler.time_function
    def train_step(self, step: int):
        """Performs a training step and computes the losses and the metrics"""
        (pixel_coords, pixels) = next(self.datamanager.iter_train_dataloader)
        ray_bundles = self.datamanager.train_ray_generator(pixel_coords)
        outputs = self.model(ray_bundles)
        losses, total_loss = self.loss_manager.compute_loss(outputs, pixels, pixel_coords, step)
        metrics = compute_metrics(outputs, pixels, modalities=self.datamanager.modalities)

        self.optimizers.zero_grad_all()
        self.fabric.backward(total_loss)
        self.clip_gradients(max_norm=2.0)
        self.optimizers.optimizer_step_all()
        self.optimizers.scheduler_step_all(step)

        return losses, total_loss, metrics

    def eval_step(self, step):
        """Performs an evaluation step and computes the losses and the metrics"""
        self.set_eval()
        losses, total_loss, metrics = None, None, None
        if check_step(step, self.trainer_config.steps_per_eval_batch):
            (pixel_coords, pixels) = next(self.datamanager.iter_eval_dataloader)
            ray_bundles = self.datamanager.eval_ray_generator(pixel_coords)

            with torch.no_grad():
                outputs = self.model.module(ray_bundles)

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

    def initialize_optimizers(self):
        """Initialize the optimizers and schedulers for the model and datamanager."""
        optim = self.config.optimizers
        param_groups = self.model.get_param_groups()
        param_groups.update(self.datamanager.get_param_groups())
        optimizers = {}
        schedulers = {}
        fields_params = []
        for param_group_name, params in param_groups.items():
            if param_group_name not in optim.keys():
                fields_params = fields_params + params
            else:
                optimizers[param_group_name] = optim[param_group_name]["optimizer"].setup(params=params)
                if optim[param_group_name]["scheduler"]:
                    schedulers[param_group_name] = optim[param_group_name]["scheduler"].setup(
                        num_iterations=self.trainer_config.max_num_iterations,
                        optimizer=optimizers[param_group_name]
                    )
        optimizers["fields"] = optim["fields"]["optimizer"].setup(params=fields_params)
        if optim["fields"]["scheduler"]:
            schedulers["fields"] = optim["fields"]["scheduler"].setup(
                num_iterations=self.trainer_config.max_num_iterations,
                optimizer=optimizers["fields"]
            )
        return optimizers, schedulers

    def fabric_setup_model(self, optimizers: Dict[str, Any]):
        """Initialize the model and optimizers with fabric"""
        optims = [optimizers[x] for x in optimizers.keys() if x != "camera_poses"]
        optims_keys = [x for x in optimizers.keys() if x != "camera_poses"]
        model_and_optimizers = self.fabric.setup(
            self.model,
            *optims
        )
        optims = dict(zip(optims_keys, model_and_optimizers[1:]))
        return model_and_optimizers[0], optims

    def fabric_setup_datamanager(self, optimizers: Dict[str, Any]):
        """Initialize the datamanager and optimizers with fabric"""
        optims = [optimizers[x] for x in optimizers.keys() if x == "camera_poses"]
        optims_keys = [x for x in optimizers.keys() if x == "camera_poses"]
        if len(optims) != 0:
            model_and_optimizers = self.fabric.setup(
                self.datamanager,
                *optims
            )
            optims = dict(zip(optims_keys, model_and_optimizers[1:]))
            return model_and_optimizers[0], optims
        return self.datamanager, {}

    def clip_gradients(self, max_norm: float):
        """Clip the gradients of the model and datamanager"""
        for key in self.optimizers.optimizers.keys():
            if key != "camera_poses":
                self.fabric.clip_gradients(
                    self.model,
                    self.optimizers.optimizers[key],
                    max_norm=max_norm,
                    error_if_nonfinite=False
                )
            else:
                self.fabric.clip_gradients(
                    self.datamanager,
                    self.optimizers.optimizers[key],
                    max_norm=max_norm,
                    error_if_nonfinite=False
                )

    def set_eval(self):
        """Set the model and datamanager to evaluation mode"""
        self.datamanager.eval()
        self.model.eval()

    def set_train(self):
        """Set the model and datamanager to training mode"""
        self.datamanager.train()
        self.model.train()

    def state_dict(self, step: int) -> Dict[str, Any]:
        """Returns the state dict of the model, datamanager, optimizers and schedulers"""
        return {
            "step": step,
            "model": self.model.state_dict(),
            "datamanager": self.datamanager.state_dict(),
            "optimizers": {k: v.state_dict() for (k, v) in self.optimizers.optimizers.items()},
            "schedulers": {k: v.state_dict() for (k, v) in self.optimizers.schedulers.items()},
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Loads the state dict of the model, datamanager, optimizers and schedulers"""
        self.model.load_state_dict(state_dict["model"])
        self.datamanager.load_state_dict(state_dict["datamanager"])
        self.optimizers.load_optimizers(state_dict["optimizers"])
        self.optimizers.load_schedulers(state_dict["schedulers"])
