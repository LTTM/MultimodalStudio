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
import copy
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Type, Dict, Any, List

import torch
from lightning import Fabric
from rich.console import Console

from engine.callbacks import TrainingCallbackAttributes
from engine.optimizers import Optimizers
from pipelines.base_pipeline import BasePipelineConfig, BasePipeline
from pipelines.raw_pipeline import RawPipelineConfig, RawPipeline

CONSOLE = Console(width=120)

@dataclass
class FineTuningPipelineConfig(BasePipelineConfig):
    """Fine-tuning Pipeline Config"""
    _target: Type = field(default_factory=lambda: FineTuningPipeline)
    pretrained_model_path: str = ""
    """Path to the pretrained model checkpoint"""
    average_multi_modules: bool = False
    """Whether to load the averaged weights of multi modules in the single instance of the same module"""
    load_averaged_camera_poses: bool = False
    """Whether to load averaged camera poses from the pretrained model"""
    fine_tuning_modalities: List[str] = field(default_factory=lambda: ["rgb"])
    """List of modalities to fine-tune"""
    optimize_geometry: bool = True
    """Whether to optimize the geometry (surface model) during fine-tuning"""

@dataclass
class FineTuningRawPipelineConfig(FineTuningPipelineConfig, RawPipelineConfig):
    """Fine-tuning Raw Pipeline Config"""
    _target: Type = field(default_factory=lambda: FineTuningRawPipeline)

class FineTuningPipeline(BasePipeline):

    def __init__(
            self,
            config: FineTuningPipelineConfig,
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
        self.config = config

    def setup(self):
        # Initialize DataManager
        with self.fabric.init_module():
            self.datamanager = self.config.datamanager.setup(
                data_dir=self.trainer_config.data_dir,
                fabric=self.fabric,
                full_view_ids=self.trainer_config.view_ids,
            )

        pretraining_state = self.load_pretraining_state_dict()
        modalities = copy.deepcopy(self.datamanager.modalities)
        modalities.update(pretraining_state["modalities"])

        # Initialize model
        scene_box = self.datamanager.train_dataset.scene_box
        with self.fabric.init_module():
            self.model = self.config.model.setup(
                scene_box=scene_box,
                modalities=modalities,
            )

        # Initialize optimizers and schedulers
        optimizers, schedulers = self.initialize_optimizers()
        self.model, model_optimizers = self.fabric_setup_model(optimizers=optimizers)
        self.datamanager, datamanager_optimizer = self.fabric_setup_datamanager(optimizers=optimizers)
        model_optimizers.update(datamanager_optimizer)
        self.optimizers = Optimizers(optimizers=model_optimizers, schedulers=schedulers)

        # Initialize loss manager
        self.loss_manager = self.config.loss_manager.setup(
            modalities=modalities,
            num_iterations=self.trainer_config.max_num_iterations,
            model=self.model,
            datamanager=self.datamanager,
            modalities_to_optimize=self.config.fine_tuning_modalities,
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

        # Load pretraining state dict
        self.load_pretrained_model(pretraining_state)
        # Freeze modules
        self.freeze_modules()

    def load_pretraining_state_dict(self):
        print(f"Loading latest pre-training checkpoint from {self.config.pretrained_model_path}")
        load_dir = self.config.pretrained_model_path
        assert os.path.exists(load_dir), f"Checkpoint {load_dir} does not exist"
        if os.path.isdir(load_dir):
            load_step = sorted(int(x[x.find("-") + 1: x.find(".")]) for x in os.listdir(load_dir))[-1]
            load_path = os.path.join(load_dir, f"step-{load_step:09d}.ckpt")
        elif os.path.isfile(load_dir):
            load_path = load_dir
        else:
            raise ValueError(f"Invalid pretraining checkpoint path: {load_dir}")
        loaded_state = torch.load(load_path, map_location="cpu", weights_only=False)
        return loaded_state

    def load_pretrained_model(self, loaded_state: Dict[str, Any]):
        model_state = copy.deepcopy(loaded_state["model"])
        multi_states = defaultdict(list)
        for key in loaded_state["model"].keys():
            if "volume_rendering.density_fn" in key:
                model_state.pop(key)
            elif "background_model" in key:
                model_state.pop(key)
            elif self.config.average_multi_modules and 'module_list' in key:
                key_list = key.split(".")
                index = key_list.index("module_list")
                key_list.pop(index+1)
                key_list.pop(index)
                new_key = ".".join(key_list)
                multi_states[new_key].append(loaded_state["model"][key])
                model_state.pop(key)
        if self.config.average_multi_modules:
            for key in multi_states.keys():
                model_state[key] = torch.mean(torch.stack(multi_states[key]), dim=0)
        self.model.load_state_dict(model_state, strict=False)
        if self.config.load_averaged_camera_poses:
            # Load averaged camera poses but not optimized rig poses. Requires shared_optimization = True.
            assert self.datamanager.train_camera_optimizer.config.relative_pose_opt == loaded_state["pose_optimizer"]["relative_pose_opt"], \
                "Pretrained model uses relative pose optimization but current model does not."
            assert self.datamanager.train_camera_optimizer.config.shared_optimization == loaded_state["pose_optimizer"]["shared_optimization"], \
                "Pretrained model uses shared optimization but current model does not."
            datamanager_state = {}
            camera_adjustments = defaultdict(list)
            for datamanager in loaded_state["datamanager_buffer"].values():
                for key in datamanager.keys():
                    if "train_camera_optimizer.pose_adjustment" in key:
                        mod = key.split(".")[-1]
                        camera_adjustments[mod].append(datamanager[f"train_camera_optimizer.pose_adjustment.{mod}"])
            for mod in camera_adjustments.keys():
                camera_adjustments[mod] = torch.mean(torch.stack(camera_adjustments[mod]), dim=0)
            # Convert camera adjustments from GT reference system to current camera reference system
            camera_adjustments = self.datamanager.train_camera_optimizer.convert_adjustments_reference_system(camera_adjustments)
            if loaded_state["pose_optimizer"]["relative_pose_opt"]:
                pretraining_reference_mod = loaded_state["pose_optimizer"]["reference_modality"]
                if self.datamanager.train_camera_optimizer.config.reference_modality != pretraining_reference_mod:
                    camera_adjustments = self.datamanager.train_camera_optimizer.switch_reference_modality(
                        camera_adjustments,
                        pretraining_reference_mod,
                        self.datamanager.train_camera_optimizer.config.reference_modality
                    )
            for mod in camera_adjustments.keys():
                datamanager_state[f"train_camera_optimizer.pose_adjustment.{mod}"] = camera_adjustments[mod]
            self.datamanager.load_state_dict(datamanager_state, strict=False)

    def freeze_modules(self):
        self.model.requires_grad_(False)
        if self.config.optimize_geometry:
            self.model.surface_model.requires_grad_(True)
        self.model.radiance_model.radiance_field.coefficient_field.requires_grad_(True)
        self.model.background_model.requires_grad_(True)
        self.model.surface_model.volume_rendering.requires_grad_(True)

        for mod in self.datamanager.modalities:
            if mod not in self.config.fine_tuning_modalities:
                self.datamanager.train_camera_optimizer.pose_adjustment[mod].requires_grad_(False)

class FineTuningRawPipeline(FineTuningPipeline,RawPipeline):

    def __init__(
            self,
            config: FineTuningRawPipelineConfig,
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
        self.config = config
