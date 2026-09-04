# Copyright 2025 Sony Group Corporation.
# All rights reserved.
#
# Licenced under the License reported at
#
#     https://github.com/LTTM/MultimodalStudio/LICENSE.txt (the "License").
#
# This code is inspired by the code available at
#
#     https://github.com/autonomousvision/factor-fields (commit 21ea155d70efce5f96399830cb424c444c977948)
#
# At the moment of this file creation, the original code is licensed under the MIT License,
# Copyright (c) 2023 autonomousvision; a copy of the MIT License, and the list of the files it
# applies to, is reported at
#
#     https://github.com/LTTM/MultimodalStudio/LICENSE_FACTOR_FIELDS.txt
#
# See the License for the specific language governing permissions and limitations under the License.
#
# Author: Federico Lincetto, Ph.D. Student at the University of Padova
from dataclasses import dataclass, field
from typing import Any

from lightning import Fabric
from rich.console import Console
from collections import defaultdict

from data.datamanager import MultiSceneDataManagerConfig
from engine.callbacks import TrainingCallbackAttributes, TrainingCallbackLocation
from engine.optimizers import Optimizers
from models.base_model import BaseModelConfig
from pipelines.base_pipeline import BasePipelineConfig, BasePipeline
from pipelines.raw_pipeline import RawPipeline
from utils.eval_utils import compute_metrics
from utils.misc import check_step

from typing import Type, Dict

CONSOLE = Console(width=120)


@dataclass
class MultiScenePipelineConfig(BasePipelineConfig):

    _target: Type = field(default_factory=lambda: MultiScenePipeline)
    steps_per_scene: int = 15
    """Number of training iterations to be performed over a selected scene"""
    specific_shared_optimization_ratio: float = 0.8
    """Ratio of optimization iterations received by the shared/specific parameters with respect to the number of steps per scene"""
    datamanager: MultiSceneDataManagerConfig = field(default_factory=lambda: MultiSceneDataManagerConfig)
    """Datamanager configuration"""
    model: BaseModelConfig = field(default_factory=lambda: BaseModelConfig)
    """Model configuration"""

@dataclass
class MultiSceneRawPipelineConfig(MultiScenePipelineConfig):
    """Raw Pipeline Config"""
    _target: Type = field(default_factory=lambda: MultiSceneRawPipeline)

class MultiScenePipeline(BasePipeline):

    def __init__(
            self,
            config: MultiScenePipelineConfig,
            fabric: Fabric,
            trainer_config,
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
        self.config = config

        self.optimizer_buffer = defaultdict(dict)
        self.scheduler_buffer = defaultdict(dict)
        self.datamanager_buffer = defaultdict(dict)
        self.specific_optimization_iters = int(self.config.steps_per_scene * self.config.specific_shared_optimization_ratio)

    def setup(self):

        self.multiscene_datamanager = self.config.datamanager.setup(
            data_dir=self.trainer_config.data_dir,
            steps_per_scene=self.config.steps_per_scene,
            fabric=self.fabric,
            full_view_ids=self.trainer_config.view_ids,
        )

        with self.fabric.init_module():
            self.multiscene_datamanager.load_datamanager(step=0)
        optimizers, self.scheduler_set = self.initialize_datamanager_optimizers()
        self.multiscene_datamanager.datamanager, self.optimizer_set = self.fabric_setup(self.multiscene_datamanager.datamanager, optimizers=optimizers)

        # INITIALIZE MODEL
        scene_box = self.multiscene_datamanager.current_train_dataset(0).scene_box
        modalities = self.multiscene_datamanager.current_train_dataset(0).get_channels_per_modality()
        n_scenes = len(self.multiscene_datamanager.data)

        with self.fabric.init_module():
            # scene_box.aabb = self.fabric.to_device(scene_box.aabb.clone().detach())
            self.model = self.config.model.setup(
                scene_box=scene_box,
                modalities=modalities,
                num_scenes=n_scenes
            )

        # INITIALIZE OPTIMIZERS AND SCHEDULERS
        optimizers, schedulers = self.initialize_model_optimizers()  # init everything
        self.model, optimizers = self.fabric_setup(self.model, optimizers=optimizers)
        self.optimizer_set.update(optimizers)
        self.scheduler_set.update(schedulers)
        self.optimizers = Optimizers(optimizers=self.optimizer_set, schedulers=self.scheduler_set)

        # INITIALIZE LOSS MANAGER
        self.loss_manager = self.config.loss_manager.setup(
            modalities=modalities,
            num_iterations=self.trainer_config.max_num_iterations,
            model=self.model,
            datamanager=self.datamanager,
        )

        # CALLBACKS
        self.callbacks = self.model.get_training_callbacks(
            TrainingCallbackAttributes(
                trainer=self.trainer_config,  # type: ignore
                model=self.model.config,
            )
        )

        # EVALUATOR
        with self.fabric.init_module():
            self.evaluator = self.config.evaluator.setup(
                pipeline=self,
                scene_box=self.multiscene_datamanager.current_train_dataset(0).scene_box,
                w2gt=self.multiscene_datamanager.current_train_dataset(0).w2gt,
                output_path=self.output_dir,
            )

        # SET OPTIMIZABLE PARAMETERS
        # TODO: check require_grads strange behavior
        #  if requires_grad of the surface model projection function is set to False at the beginning, before the first iteration,
        #  the model converges badly and the coeffs and bases cannot be optimized until the projection function is optimized
        #  for at least few iterations.

        for callback in self.callbacks:
            callback.run_callback_at_location(0, location=TrainingCallbackLocation.BEFORE_TRAIN_ITERATION)
        self.train_step(0)
        self.train_step(0)
        self.optimize_specific()

    def fabric_setup(self, module, optimizers: Dict[str, Any]):
        optims = list(optimizers.values())
        optims_keys = list(optimizers.keys())
        if len(optims) != 0:
            model_and_optimizers = self.fabric.setup(module, *optims)
            optims = {key: value for (key, value) in zip(optims_keys, model_and_optimizers[1:])}
            return model_and_optimizers[0], optims
        else:
            return module, {}

    def initialize_datamanager_optimizers(self, datamanager=None):
        optim = self.config.optimizers
        param_groups = self.multiscene_datamanager.get_param_groups() if datamanager is None else datamanager.get_param_groups()
        optimizers = {}
        schedulers = {}
        fields_params = []
        for param_group_name, params in param_groups.items():
            if param_group_name not in optim.keys():
                fields_params = fields_params + params
            else:
                optimizers[param_group_name] = optim[param_group_name]["optimizer"].setup(params=params)
                if optim[param_group_name]["scheduler"]:
                    schedulers[param_group_name] = optim[param_group_name]["scheduler"].setup(num_iterations=self.trainer_config.max_num_iterations, optimizer=optimizers[param_group_name])
        if len(fields_params) != 0:
            if optimizers.get("datamanager_fields", None) is None:
                optimizers["datamanager_fields"] = optim["datamanager_fields"]["optimizer"].setup(params=fields_params)
            else:
                optimizers["datamanager_fields"].add_param_gropus(fields_params)
            if optim["datamanager_fields"]["scheduler"] and schedulers.get("datamanager_fields", None) is None:
                schedulers["datamanager_fields"] = optim["datamanager_fields"]["scheduler"].setup(num_iterations=self.trainer_config.max_num_iterations, optimizer=optimizers["datamanager_fields"])
        return optimizers, schedulers

    def initialize_model_optimizers(self):
        optim = self.config.optimizers
        param_groups = self.model.get_param_groups()
        optimizers = {}
        schedulers = {}
        fields_params = []
        for param_group_name, params in param_groups.items():
            if param_group_name not in optim.keys():
                fields_params = fields_params + params
            else:
                optimizers[param_group_name] = optim[param_group_name]["optimizer"].setup(params=params)
                if optim[param_group_name]["scheduler"]:
                    schedulers[param_group_name] = optim[param_group_name]["scheduler"].setup(num_iterations=self.trainer_config.max_num_iterations, optimizer=optimizers[param_group_name])
        if len(fields_params) != 0:
            if optimizers.get("model_fields", None) is None:
                optimizers["model_fields"] = optim["model_fields"]["optimizer"].setup(params=fields_params)
            else:
                optimizers["model_fields"].add_param_gropus(fields_params)
            if optim["model_fields"]["scheduler"] and schedulers.get("model_fields", None) is None:
                schedulers["model_fields"] = optim["model_fields"]["scheduler"].setup(num_iterations=self.trainer_config.max_num_iterations, optimizer=optimizers["model_fields"])
        return optimizers, schedulers

    def load_module_state(self, scene_index):
        with self.fabric.init_module():
            self.multiscene_datamanager.load_datamanager(datamanager=self.datamanager_buffer[scene_index])
        for param_group in self.config.optimizers.keys():
            if self.config.optimizers[param_group]["optimization"] == "specific":
                param_optimizer = self.optimizer_set.get(param_group, None)
                if param_optimizer is not None:
                    param_optimizer.load_state_dict(self.optimizer_buffer[scene_index])
                    if self.config.optimizers[param_group]["scheduler"] is not None:
                        self.scheduler_set[param_group].load_state_dict(self.scheduler_buffer[scene_index])

    def save_module_state(self, scene_index):
        self.datamanager_buffer[scene_index] = self.multiscene_datamanager.current_datamanager()
        for param_group in self.config.optimizers.keys():
            if self.config.optimizers[param_group]["optimization"] == "specific":
                param_optimizer = self.optimizers.optimizers.get(param_group, None)
                if param_optimizer is not None:
                    self.optimizer_buffer[scene_index] = param_optimizer.state_dict()
                    if self.config.optimizers[param_group]["scheduler"] is not None:
                        self.scheduler_buffer[scene_index] = self.optimizers.schedulers[param_group].state_dict()

    def update_evaluator(self, step: int = 0, scene_index: int = None):
        scene_box = self.multiscene_datamanager.current_train_dataset(step).scene_box \
            if scene_index is None else self.multiscene_datamanager.get_train_dataset(scene_index).scene_box
        w2gt = self.multiscene_datamanager.current_train_dataset(step).w2gt \
            if scene_index is None else self.multiscene_datamanager.get_train_dataset(scene_index).w2gt
        self.evaluator.update_w2gt(w2gt)
        self.evaluator.update_scene_box(scene_box)

    def switch_optimizable(self, step: int):
        if check_step(step, self.config.steps_per_scene, skip_first=True):
            self.optimize_specific()
        elif check_step(step, self.config.steps_per_scene, shift=self.specific_optimization_iters, skip_first=False):
            self.optimize_shared()

    def optimize_shared(self):
        self.model.requires_grad_(requires_grad=True)
        self.model.surface_model.surface_field.field.requires_grad_(requires_grad=False)
        self.model.radiance_model.radiance_field.coefficient_field.requires_grad_(requires_grad=False)
        # self.model.radiance_model.radiance_field.basis_field.requires_grad_(requires_grad=True)
        # self.model.radiance_model.radiance_field.projection_function.requires_grad_(requires_grad=True)
        # self.model.radiance_model.radiance_field.radiance_predictor.requires_grad_(requires_grad=True)
        # self.model.radiance_model.head_field.requires_grad_(requires_grad=True)
        # self.model.surface_model.volume_rendering.density_fn.requires_grad_(requires_grad=True)
        self.model.background_model.requires_grad_(requires_grad=False)

    def optimize_specific(self):
        self.model.requires_grad_(requires_grad=False)
        self.model.surface_model.surface_field.field.requires_grad_(requires_grad=True)
        self.model.radiance_model.radiance_field.coefficient_field.requires_grad_(requires_grad=True)
        # self.model.radiance_model.radiance_field.basis_field.requires_grad_(requires_grad=False)
        # self.model.radiance_model.radiance_field.projection_function.requires_grad_(requires_grad=False)
        # self.model.radiance_model.radiance_field.radiance_predictor.requires_grad_(requires_grad=False)
        # self.model.radiance_model.head_field.requires_grad_(requires_grad=False)
        # self.model.surface_model.volume_rendering.density_fn.requires_grad_(requires_grad=False)
        self.model.background_model.requires_grad_(requires_grad=True)

    def switch_scene(self, step: int):
        # INDEX EXTRACTION
        scene_index = self.multiscene_datamanager.get_current_scene_idx(step=step-1)
        # SAVE MODULES STATE BEFORE SWAP
        self.save_module_state(scene_index)
        # CHECK SCENE RESHUFFLE
        if check_step(step, self.config.steps_per_scene * self.multiscene_datamanager.n_scenes, skip_first=True):
            self.multiscene_datamanager.shuffle_scenes()
        # SWAP MODULE
        scene_index = self.multiscene_datamanager.get_current_scene_idx(step=step)
        self.swap_modules(scene_index=scene_index)
        self.update_evaluator(step=step)

    def switch_scene_by_index(self, current_scene_index: int, next_scene_index: int, save_module_states: bool = True):
        # SAVE MODULES STATE BEFORE SWAP
        if save_module_states:
            self.save_module_state(current_scene_index)
        # SWAP MODULE
        self.swap_modules(scene_index=next_scene_index)
        self.update_evaluator(scene_index=next_scene_index)

    def swap_modules(self, step: int = None, scene_index: int = None):
        assert step is not None or scene_index is not None, "Either step or scene_index must be provided"
        if scene_index is None:
            scene_index = self.multiscene_datamanager.get_current_scene_idx(step=step)
        if scene_index not in self.datamanager_buffer:
            self.init_datamanager(scene_index=scene_index)
        else:
            self.load_module_state(scene_index)
        self.optimizers = Optimizers(optimizers=self.optimizer_set, schedulers=self.scheduler_set)

    def init_datamanager(self, scene_index: int):
        with self.fabric.init_module():
            self.multiscene_datamanager.load_datamanager(scene_index=scene_index)
        optimizers, schedulers = self.initialize_datamanager_optimizers()
        self.multiscene_datamanager.datamanager, optimizers = self.fabric_setup(self.multiscene_datamanager.datamanager, optimizers=optimizers)
        self.optimizer_set.update(optimizers)
        self.scheduler_set.update(schedulers)

    def init_detached_datamanager(self, scene_index: int):
        with self.fabric.init_module():
            datamanager = self.multiscene_datamanager.init_detached_datamanager(scene_idx=scene_index)
        optimizers, schedulers = self.initialize_datamanager_optimizers(datamanager=datamanager)
        datamanager, optimizers = self.fabric_setup(datamanager, optimizers=optimizers)
        return datamanager, optimizers, schedulers

    def train_step(self, step):
        # MULTI SCENE MANAGEMENT
        if check_step(step, self.config.steps_per_scene):
            self.switch_scene(step)
        self.switch_optimizable(step)
        # TRAINING STEP
        (pixel_coords, pixels) = next(self.multiscene_datamanager.current_datamanager().iter_train_dataloader)
        ray_bundles = self.multiscene_datamanager.current_datamanager().train_ray_generator(pixel_coords)
        scene_index = self.multiscene_datamanager.get_current_scene_idx(step=step)
        outputs = self.model(ray_bundles, scene_idx=scene_index)
        losses, total_loss = self.loss_manager.compute_loss(outputs, pixels, pixel_coords, step)
        metrics = compute_metrics(outputs, pixels, modalities=self.multiscene_datamanager.config.modalities)
        # OPTIMIZATION
        self.optimizers.zero_grad_all()
        self.fabric.backward(total_loss)
        self.clip_gradients(max_norm=2.0)
        self.optimizers.optimizer_step_all()
        self.optimizers.scheduler_step_all(step)
        return losses, total_loss, metrics

    def eval_step(self, step):
        self.set_eval()
        losses, total_loss, metrics = None, None, None
        if self.fabric.global_rank == 0:
            losses, total_loss, metrics = self.evaluator.evaluation_step(step)
        self.set_train()
        return losses, total_loss, metrics

    def set_eval(self):
        self.multiscene_datamanager.eval()
        self.model.eval()

    def set_train(self):
        self.multiscene_datamanager.train()
        self.model.train()

    @property
    def datamanager(self):
        return self.multiscene_datamanager.current_datamanager()

    def state_dict(self, step: int) -> Dict[str, Any]:
        scene_index = self.multiscene_datamanager.get_current_scene_idx(step=step)
        self.save_module_state(scene_index)
        state = {
            "step": step,
            "modalities": self.datamanager.modalities, #{mod: ch for mod, ch in zip(self.datamanager.modalities, self.datamanager.channels_per_modality)},
            "model": self.model.state_dict(),
            "multiscene_datamanager": self.multiscene_datamanager.state_dict(),
            "optimizers": {k: v.state_dict() for (k, v) in self.optimizers.optimizers.items()},
            "schedulers": {k: v.state_dict() for (k, v) in self.optimizers.schedulers.items()},
            "datamanager_buffer": {k: v.state_dict() for (k, v) in self.datamanager_buffer.items()},
            "pose_optimizer": {
                "shared_optimization": self.datamanager.train_camera_optimizer.config.shared_optimization,
                "reference_modality": self.datamanager.train_camera_optimizer.config.reference_modality,
                "relative_pose_opt": self.datamanager.train_camera_optimizer.config.relative_pose_opt,
            },
            "optimizer_buffer": {k: v for (k, v) in self.optimizer_buffer.items()},
            "scheduler_buffer": {k: v for (k, v) in self.scheduler_buffer.items()},
        }
        return state

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.model.load_state_dict(state_dict["model"])
        self.multiscene_datamanager.load_state_dict(state_dict["multiscene_datamanager"])
        for k, v in state_dict["datamanager_buffer"].items():
            datamanager, _, _ = self.init_detached_datamanager(scene_index=k)
            datamanager.load_state_dict(v)
            self.datamanager_buffer[k] = datamanager
        self.optimizer_buffer = state_dict["optimizer_buffer"]
        self.scheduler_buffer = state_dict["scheduler_buffer"]
        self.optimizers.load_optimizers(state_dict["optimizers"])
        self.optimizers.load_schedulers(state_dict["schedulers"])
        self.swap_modules(step=state_dict["step"])
        self.optimize_specific()
        self.update_evaluator(step=state_dict["step"])
        if self.datamanager.train_camera_optimizer.config.relative_pose_opt != state_dict["pose_optimizer"]["relative_pose_opt"] or \
           self.datamanager.train_camera_optimizer.config.reference_modality != state_dict["pose_optimizer"]["reference_modality"] or \
           self.datamanager.train_camera_optimizer.config.shared_optimization != state_dict["pose_optimizer"]["shared_optimization"]:
            raise ValueError("Loaded pose optimizer configuration does not match the current one.")

class MultiSceneRawPipeline(MultiScenePipeline,RawPipeline):
    def __init__(
            self,
            config: MultiScenePipelineConfig,
            fabric: Fabric,
            trainer_config,
            output_dir: str,
            checkpoint_dir: str,
            mixed_precision: bool,
    ):
        super().__init__(config, fabric, trainer_config, output_dir, checkpoint_dir, mixed_precision)
        self.config = config

    def train_step(self, step):
        # MULTI SCENE MANAGEMENT
        if check_step(step, self.config.steps_per_scene):
            self.switch_scene(step)
        self.switch_optimizable(step)
        # TRAINING STEP
        (pixel_coords, pixels) = next(self.multiscene_datamanager.current_datamanager().iter_train_dataloader)
        ray_bundles = self.multiscene_datamanager.current_datamanager().train_ray_generator(pixel_coords)
        scene_index = self.multiscene_datamanager.get_current_scene_idx(step=step)
        outputs = self.model(ray_bundles, scene_idx=scene_index)
        losses, total_loss = self.loss_manager.compute_loss(outputs, pixels, pixel_coords, step)
        metrics = compute_metrics(outputs, pixels, modalities=self.multiscene_datamanager.config.modalities)
        # OPTIMIZATION
        self.optimizers.zero_grad_all()
        self.fabric.backward(total_loss)
        self.clip_gradients(max_norm=2.0)
        self.optimizers.optimizer_step_all()
        self.optimizers.scheduler_step_all(step)
        return losses, total_loss, metrics

    def eval_step(self, step):
        self.set_eval()
        losses, total_loss, metrics = None, None, None
        if self.fabric.global_rank == 0:
            losses, total_loss, metrics = self.evaluator.evaluation_step(step)
        self.set_train()
        return losses, total_loss, metrics
