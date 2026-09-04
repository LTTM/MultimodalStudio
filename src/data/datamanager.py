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

import os
import copy
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Type, Optional, Tuple, Literal, List, Dict, Mapping, Any
from random import shuffle

import torch
import lightning as L
from torch.nn import Parameter
from tqdm import tqdm

from cameras.camera_optimizers import CameraOptimizerConfig
from cameras.pixel_samplers import UniformPixelSamplerConfig, DensePixelSampler
from configs.configs import InstantiateConfig
from data.dataloaders import CacheDataloader, SingleViewDataloader
from data.datasets import BaseDatasetConfig, BaseDataset
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

@dataclass
class MultiSceneDataManagerConfig(InstantiateConfig):
    """Configuration for data manager instantiation; MultiSceneDataManager is in charge of keeping the train/eval dataparsers;
    After instantiation, data manager holds both train/eval datasets and is in charge of returning unpacked
    train/eval data at each iteration
    """

    _target: Type = field(default_factory=lambda: MultiSceneDataManager)
    """Target class to instantiate."""
    dataset_class: BaseDatasetConfig = field(default_factory=lambda: BaseDatasetConfig)
    """Dataset configuration"""
    mode: Literal["pre-training", "fine-tuning"] = "pre-training"
    """Specifies the modality of the task."""
    eval_scene_indices: Tuple[int, ...] = None
    """Specifies the scene_indices to either exclude or use for test"""
    eval_image_indices: Optional[Tuple[int, ...]] = None
    """Specifies the image indices to use during eval. Same indices for every modality."""
    eval_image_indices_per_modality: Optional[Dict[str, Tuple[int, ...]]] = None
    """Specifies the image indices to use during eval. Independent indices per modality."""
    eval_image_ratio: float = 0.0
    """Specifies the percentage of dataset image to be used for evaluation"""
    skip_image_indices: Optional[Tuple[int, ...]] = field(default_factory=lambda: [])
    """Specifies the image indices to ignore."""
    pixel_sampler: UniformPixelSamplerConfig = field(default_factory=lambda: UniformPixelSamplerConfig)
    """Specifies the pixel sampler used during training."""
    camera_optimizer: CameraOptimizerConfig = field(default_factory=lambda: CameraOptimizerConfig)
    """Specifies the camera pose optimizer used during training. Helpful if poses are noisy, such as for data from Record3D."""
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
            fabric: L.Fabric,
            data_dir: str = None,
            full_view_ids: Optional[List[int]] = None,
            train_dataset: BaseDataset = None,
            eval_dataset: BaseDataset = None,
    ):
        super().__init__()
        self.config = config
        self.fabric = fabric

        if data_dir is not None and eval_dataset is not None and train_dataset is not None:
            raise ValueError("Either data_dir or train_dataset and eval_dataset must be provided.")

        if train_dataset is not None and eval_dataset is not None:
            self.train_dataset = train_dataset
            self.eval_dataset = eval_dataset
            # self.num_cameras = len(self.train_dataset) + len(self.eval_dataset)
        else:
            if self.config.eval_image_indices is not None:
                self.train_dataset = self.config.dataset_class.setup(
                    modalities=config.modalities,
                    data_dir=data_dir,
                    indexes_to_exclude=self.config.eval_image_indices + self.config.skip_image_indices
                )
                self.eval_dataset = self.config.dataset_class.setup(
                    modalities=config.modalities,
                    data_dir=data_dir,
                    indexes_to_choose=self.config.eval_image_indices
                )
            elif self.config.eval_image_indices_per_modality is not None:
                self.train_dataset = self.config.dataset_class.setup(
                    modalities=config.modalities,
                    data_dir=data_dir,
                    indexes_to_exclude_per_modality={
                        mod: self.config.eval_image_indices_per_modality[mod] + self.config.skip_image_indices_per_modality[mod]
                        for mod in self.config.eval_image_indices_per_modality
                    }
                )
                self.eval_dataset = self.config.dataset_class.setup(
                    modalities=config.modalities,
                    data_dir=data_dir,
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

        self.pixel_sampler = self.config.pixel_sampler.setup(device=self.fabric.device, modalities=self.modalities)

        self.train_camera_optimizer = self.config.camera_optimizer.setup(num_cameras=len(self.train_dataset), dataset=self.train_dataset)
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

    def load_state_dict(
        self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False
    ):
        self.train_camera_optimizer.load_state_dict({
            k[len("train_camera_optimizer") + 1:]: v
            for k, v in state_dict.items() if "train_camera_optimizer" in k
        }, strict=strict)
        self.eval_camera_optimizer.load_state_dict({
            k[len("eval_camera_optimizer") + 1:]: v
            for k, v in state_dict.items() if "eval_camera_optimizer" in k
        }, strict=strict)

class MultiSceneDataManager(torch.nn.Module):

    def __init__(
            self,
            config: MultiSceneDataManagerConfig,
            data_dir: str,
            steps_per_scene: int,
            fabric: L.Fabric,
            full_view_ids: Optional[List[int]] = None,
    ):
        super().__init__()
        self.config = config
        self.fabric = fabric
        self.steps_per_scene = steps_per_scene
        self.full_view_ids = full_view_ids

        # RETRIEVE SCENES PATHS BASED ON THE SELECTED TRAINING MODALITY
        if self.config.mode == "pre-training":
            self.scenes_path = self.scan_and_select_folders(
                data_dir, indices_to_exclude=self.config.eval_scene_indices
            )
        elif self.config.mode == "fine-tuning":
            self.scenes_path = self.scan_and_select_folders(
                data_dir, indices_to_choose=self.config.eval_scene_indices
            )
        else:
            raise ValueError("Mode must be either 'pre-training' or 'fine-tuning'.")

        # LOAD DATASETS
        print(f"Loading {len(self.scenes_path)} scenes from {data_dir}...")
        self.data = defaultdict(dict)
        if self.config.eval_image_indices is not None:
            for i in tqdm(range(len(self.scenes_path))):
                print(f"Processing scene: {self.scenes_path[i]}")
                self.data[i]["train"] = self.config.dataset_class.setup(
                    modalities=self.config.modalities,
                    data_dir=self.scenes_path[i],
                    indexes_to_exclude=self.config.eval_image_indices + self.config.skip_image_indices,
                )
                self.data[i]["eval"] = self.config.dataset_class.setup(
                    modalities=self.config.modalities,
                    data_dir=self.scenes_path[i],
                    indexes_to_choose=self.config.eval_image_indices,
                )
            # self.num_cameras = len(self.data[0]["train"]) + len(self.data[0]["eval"])
        elif self.config.eval_image_indices_per_modality is not None:
            for i in tqdm(range(len(self.scenes_path))):
                print(f"Processing scene: {self.scenes_path[i]}")
                self.data[i]["train"] = self.config.dataset_class.setup(
                    modalities=self.config.modalities,
                    data_dir=self.scenes_path[i],
                    indexes_to_exclude_per_modality=self.config.eval_image_indices_per_modality,
                )
                self.data[i]["eval"] = self.config.dataset_class.setup(
                    modalities=self.config.modalities,
                    data_dir=self.scenes_path[i],
                    indexes_to_choose_per_modality=self.config.eval_image_indices_per_modality,
                )
            # self.num_cameras = len(self.data[0]["train"]) + len(self.data[0]["eval"])
        elif self.config.eval_image_ratio > 0:
            for i in tqdm(range(len(self.scenes_path))):
                print(f"Processing scene: {self.scenes_path[i]}")
                self.data[i]["train"] = self.config.dataset_class.setup(
                    modalities=self.config.modalities,
                    data_dir=self.scenes_path[i],
                    indexes_to_exclude_ratio=self.config.eval_image_ratio,
                )
                self.data[i]["eval"] = self.config.dataset_class.setup(
                    modalities=self.config.modalities,
                    data_dir=self.scenes_path[i],
                    indexes_to_exclude=self.data[i]["train"].indexes,
                )
            # self.num_cameras = len(self.data[0]["train"]) + len(self.data[0]["eval"])
        else:
            for i in tqdm(range(len(self.scenes_path))):
                print(f"Processing scene: {self.scenes_path[i]}")
                self.data[i]["train"] = self.config.dataset_class.setup(
                    modalities=self.config.modalities,
                    data_dir=self.scenes_path[i],
                )
                self.data[i]["eval"] = self.config.dataset_class.setup(
                    modalities=self.config.modalities,
                    data_dir=self.scenes_path[i],
                )
            # self.num_cameras = len(self.data[0]["train"])

        self.modalities = self.data[0]["train"].get_channels_per_modality()
        self.datamanager = None
        self.n_scenes = len(self.data)
        self.scene_permutation = list(range(len(self.data)))
        self.shuffle_scenes()

    def load_datamanager(self, step: int = None, scene_index: int = None, datamanager: DataManager = None):
        if datamanager is not None:
            self.datamanager = datamanager
        else:
            assert step is not None or scene_index is not None, "Either step or scene_index must be provided."
            if scene_index is None:
                scene_index = self.get_current_scene_idx(step)
            # with self.fabric.init_module():
            self.datamanager = DataManagerConfig(
                pixel_sampler=self.config.pixel_sampler,
                camera_optimizer=self.config.camera_optimizer,
            ).setup(
                full_view_ids=self.full_view_ids,
                fabric=self.fabric,
                train_dataset=self.get_train_dataset(scene_idx=scene_index),
                eval_dataset=self.get_eval_dataset(scene_idx=scene_index),
            )
        if self.datamanager.train_dataset.metadata["raw"]:
            self.datamanager.train_dataset.mosaick_mask_per_modality = self.fabric.to_device(self.datamanager.train_dataset.mosaick_mask_per_modality)
            self.datamanager.eval_dataset.mosaick_mask_per_modality = self.fabric.to_device(self.datamanager.eval_dataset.mosaick_mask_per_modality)
            self.datamanager.train_dataset.mosaick_mask_across_modalities = self.fabric.to_device(self.datamanager.train_dataset.mosaick_mask_across_modalities)
            self.datamanager.eval_dataset.mosaick_mask_across_modalities = self.fabric.to_device(self.datamanager.eval_dataset.mosaick_mask_across_modalities)

    def init_detached_datamanager(self, step: int = None, scene_idx: int = None):
        datamanager = DataManagerConfig(
            pixel_sampler=self.config.pixel_sampler,
            camera_optimizer=self.config.camera_optimizer,
        ).setup(
            full_view_ids=self.full_view_ids,
            fabric=self.fabric,
            train_dataset=self.current_train_dataset(step=step) if step is not None else self.get_train_dataset(scene_idx),
            eval_dataset=self.current_eval_dataset(step=step) if step is not None else self.get_eval_dataset(scene_idx),
        )
        return datamanager

    def current_train_dataset(self, step):
        scene_idx = self.get_current_scene_idx(step)
        return self.data[scene_idx]["train"]

    def current_eval_dataset(self, step):
        scene_idx = self.get_current_scene_idx(step)
        return self.data[scene_idx]["eval"]

    def get_train_dataset(self, scene_idx):
        return self.data[scene_idx]["train"]

    def get_eval_dataset(self, scene_idx):
        return self.data[scene_idx]["eval"]

    def current_datamanager(self):
        return self.datamanager

    def shuffle_scenes(self):
        shuffle(self.scene_permutation)

    def get_current_scene_idx(self, step):
        return self.scene_permutation[step // self.steps_per_scene % len(self.data)]

    def scan_and_select_folders(
            self,
            root_folder: str,
            indices_to_choose: Tuple[int, ...] = None,
            indices_to_exclude: Tuple[int, ...] = None,
    ) -> Tuple[List[str]]:

        if not os.path.isdir(root_folder):
            raise ValueError(f"{root_folder} is not a valid directory.")

        folder_paths = [
            os.path.join(root_folder, name)
            for name in sorted(os.listdir(root_folder))
            if os.path.isdir(os.path.join(root_folder, name))
        ]

        if indices_to_exclude is not None:
            selected_paths = [folder_paths[i] for i in range(len(folder_paths)) if i not in indices_to_exclude]
            self.scene_indexes = [i for i in range(len(folder_paths)) if i not in indices_to_exclude]
        elif indices_to_choose is not None:
            selected_paths = [folder_paths[i] for i in range(len(folder_paths)) if i in indices_to_choose]
            self.scene_indexes = [i for i in range(len(folder_paths)) if i in indices_to_choose]
        else:
            raise ValueError("Either indices_to_choose or indices_to_exclude must be provided.")
        self.scene_names = [os.path.basename(path) for path in selected_paths]

        return selected_paths

    def forward(self):
        """Blank forward method

        This is an nn.Module, and so requires a forward() method normally, although in our case
        we do not need a forward() method"""
        raise NotImplementedError

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = self.datamanager.get_param_groups()
        return param_groups

    def state_dict(self):
        state_dict = {
            "scene_permutation": self.scene_permutation,
        }
        return state_dict

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False):
        self.scene_permutation = state_dict["scene_permutation"]