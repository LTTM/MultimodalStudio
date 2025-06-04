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

"""DataManager"""

import copy
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Type, Optional, Tuple, List, Dict

import torch
import lightning as L
from torch.nn import Parameter

from cameras.camera_optimizers import CameraOptimizerConfig
from cameras.pixel_samplers import UniformPixelSamplerConfig, DensePixelSampler
from configs.configs import InstantiateConfig
from data.dataloaders import CacheDataloader, SingleViewDataloader
from data.datasets import BaseDatasetConfig
from model_components.ray_generators import RayGenerator

@dataclass
class DataManagerConfig(InstantiateConfig):
    """Configuration for data manager instantiation; DataManager is in charge of keeping the train/eval datasets and
    dataloaders, the pixel_sampler and the camera optimizers.
    """

    _target: Type = field(default_factory=lambda: DataManager)
    dataset_class: BaseDatasetConfig = field(default_factory=lambda: BaseDatasetConfig)
    """Dataset configuration"""
    eval_image_indices: Optional[Tuple[int, ...]] = None
    """Specifies the image indices to use during eval. Same indices for every modality."""
    eval_image_indices_per_modality: Optional[Dict[str, Tuple[int, ...]]] = None
    """Specifies the image indices to use during eval. Independent indices per modality."""
    eval_image_ratio: float = 0.0
    """Specifies the percentage of dataset image to be used for evaluation"""
    skip_image_indices: Optional[Tuple[int, ...]] = field(default_factory=lambda: [])
    """Specifies the image indices to ignore."""
    skip_image_indices_per_modality: Optional[Dict[str, Tuple[int, ...]]] = field(default_factory=lambda: defaultdict(list))
    """Specifies the image indices to ignore per modality."""
    pixel_sampler: UniformPixelSamplerConfig = field(default_factory=lambda: UniformPixelSamplerConfig)
    """Specifies the pixel sampler used during training."""
    camera_optimizer: CameraOptimizerConfig = field(default_factory=lambda: CameraOptimizerConfig)
    """Specifies the camera pose optimizer used during training. Helpful if poses are noisy."""
    modalities: List[str] = field(default_factory=list)
    """Modalities to load"""

class DataManager(torch.nn.Module):
    """
    Module in charge of managing the datasets, the dataloaders, the camera optimizers and the pixel samplers.

    Args:
        config: Configuration for the DataManager.
        data_dir: Directory where the data is stored.
        fabric: Lightning fabric object.
        full_view_ids: List of view ids to use for full view rendering. If None, all the views of train/eval datasets
                       are considered for full view rendering.
    """

    def __init__(
            self,
            config: DataManagerConfig,
            data_dir: str,
            fabric: L.Fabric,
            full_view_ids: Optional[List[int]] = None
    ):
        super().__init__()
        self.config = config
        self.fabric = fabric

        if self.config.eval_image_indices is not None:
            self.train_dataset = self.config.dataset_class.setup(
                modalities=config.modalities, data_dir=data_dir,
                indexes_to_exclude=self.config.eval_image_indices + self.config.skip_image_indices
            )
            self.eval_dataset = self.config.dataset_class.setup(
                modalities=config.modalities,
                data_dir=data_dir,
                indexes_to_choose=self.config.eval_image_indices
            )
        elif self.config.eval_image_indices_per_modality is not None:
            self.train_dataset = self.config.dataset_class.setup(
                modalities=config.modalities, data_dir=data_dir,
                indexes_to_exclude_per_modality={
                    mod: self.config.eval_image_indices_per_modality[mod] + self.config.skip_image_indices_per_modality[mod]
                    for mod in self.config.eval_image_indices_per_modality
                }
            )
            self.eval_dataset = self.config.dataset_class.setup(
                modalities=config.modalities, data_dir=data_dir,
                indexes_to_choose_per_modality=self.config.eval_image_indices_per_modality
            )
        elif self.config.eval_image_ratio > 0:
            self.train_dataset = self.config.dataset_class.setup(
                modalities=config.modalities,
                data_dir=data_dir,
                indexes_to_exclude_ratio=self.config.eval_image_ratio
            )
            self.eval_dataset = self.config.dataset_class.setup(
                modalities=config.modalities,
                data_dir=data_dir,
                indexes_to_exclude=self.train_dataset.indexes
            )
        else:
            self.train_dataset = self.config.dataset_class.setup(modalities=config.modalities, data_dir=data_dir)
            self.eval_dataset = self.config.dataset_class.setup(modalities=config.modalities, data_dir=data_dir)

        self.modalities = self.train_dataset.get_channels_per_modality()

        self.pixel_sampler = self.config.pixel_sampler.setup(device=self.fabric.device)

        self.train_camera_optimizer = self.config.camera_optimizer.setup(num_cameras=len(self.train_dataset))
        if self.config.camera_optimizer.shared_optimization:
            self.eval_camera_optimizer = copy.deepcopy(self.train_camera_optimizer)
            self.eval_camera_optimizer.pose_adjustment = self.train_camera_optimizer.pose_adjustment
            self.eval_camera_optimizer.set_num_cameras(len(self.eval_dataset))
        else:
            camera_optimizer_config = copy.deepcopy(self.config.camera_optimizer)
            camera_optimizer_config.mode = "off"
            self.eval_camera_optimizer = camera_optimizer_config.setup(num_cameras=len(self.eval_dataset))

        self.train_ray_generator = RayGenerator(
            self.train_dataset.data,
            self.train_camera_optimizer,
            self.train_dataset.metadata['pixel_offset']
        )
        self.eval_ray_generator = RayGenerator(
            self.eval_dataset.data,
            self.eval_camera_optimizer,
            self.eval_dataset.metadata['pixel_offset']
        )

        self.train_dataloader = CacheDataloader(
            self.train_dataset,
            self.pixel_sampler,
            num_workers=4,
            pin_memory=True
        )
        self.eval_dataloader = CacheDataloader(
            self.eval_dataset,
            self.pixel_sampler,
            num_workers=4,
            pin_memory=True
        )

        self.full_view_train_dataloader = SingleViewDataloader(
            self.train_dataset,
            pixel_sampler=DensePixelSampler(),
            num_workers=2,
            view_list=full_view_ids,
        )

        self.full_view_eval_dataloader = SingleViewDataloader(
            self.eval_dataset,
            pixel_sampler=DensePixelSampler(),
            num_workers=2,
            view_list=full_view_ids,
        )

        self.train_dataloader = self.fabric.setup_dataloaders(self.train_dataloader)
        self.eval_dataloader = self.fabric.setup_dataloaders(self.eval_dataloader)
        self.full_view_train_dataloader = self.fabric.setup_dataloaders(self.full_view_train_dataloader)
        self.full_view_eval_dataloader = self.fabric.setup_dataloaders(self.full_view_eval_dataloader)
        self.iter_train_dataloader = iter(self.train_dataloader)
        self.iter_eval_dataloader = iter(self.eval_dataloader)
        self.iter_full_view_train_dataloader = iter(self.full_view_train_dataloader)
        self.iter_full_view_eval_dataloader = iter(self.full_view_eval_dataloader)

    def forward(self):
        """Blank forward method

        This is a nn.Module, and so requires a forward() method normally, although in our case
        we do not need a forward() method"""
        raise NotImplementedError

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Return the parameters to pass to the optimizer"""
        camera_optimizer_parameters = list(self.train_camera_optimizer.parameters())
        param_groups = {}
        if len(camera_optimizer_parameters) > 0:
            param_groups["camera_poses"] = camera_optimizer_parameters
        return param_groups
