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
Dataset Classes
"""

import os
import sys
import math
import random
from dataclasses import dataclass, field
from typing import Tuple, Type, Dict
from torchtyping import TensorType

import torch
from torch.utils.data import Dataset

from configs.configs import InstantiateConfig
from data.scene_box import SceneBox
from cameras.cameras import Cameras, CAMERA_MODEL_TO_TYPE
from utils.io import load_from_json, read_frame
from utils.misc import normalize_frame

@dataclass
class BaseDatasetConfig(InstantiateConfig):
    """Base configuration for datasets."""
    _target: Type = field(default_factory=lambda: BaseDataset)

@dataclass
class RawDatasetConfig(BaseDatasetConfig):
    """Base configuration for raw datasets."""
    _target: Type = field(default_factory=lambda: RawDataset)

@dataclass
class BaseAlignedDatasetConfig(BaseDatasetConfig):
    """
    Dataset configuration for datasets containing aligned multimodal frames.
    All the modality frames of a specific view are considered to be either training or test frames.
    Class to be extended by more specific dataset classes, depending on whether the frames are raw or demosaicked.
    """
    _target: Type = field(default_factory=lambda: BaseAlignedDataset)

@dataclass
class BaseUnalignedDatasetConfig(BaseDatasetConfig):
    """
    Dataset configuration for datasets containing unaligned multimodal frames.
    It is possible to specify which modality frames of which views are considered to be training or test frames
    individually.
    Class to be extended by more specific dataset classes, according to whether the frames are raw or demosaicked.
    """
    _target: Type = field(default_factory=lambda: BaseUnalignedDataset)

@dataclass
class MultimodalAlignedDatasetConfig(BaseAlignedDatasetConfig):
    """Dataset configuration for datasets containing demosaicked aligned multimodal frames."""
    _target: Type = field(default_factory=lambda: MultimodalAlignedDataset)

@dataclass
class RawMultimodalAlignedDatasetConfig(BaseAlignedDatasetConfig):
    """Dataset configuration for datasets containing raw aligned multimodal frames."""
    _target: Type = field(default_factory=lambda: RawMultimodalAlignedDataset)

@dataclass
class MultimodalUnalignedDatasetConfig(BaseUnalignedDatasetConfig):
    """Dataset configuration for datasets containing demosaicked unaligned multimodal frames."""
    _target: Type = field(default_factory=lambda: MultimodalUnalignedDataset)

@dataclass
class RawMultimodalUnalignedDatasetConfig(MultimodalUnalignedDatasetConfig):
    """Dataset configuration for datasets containing raw unaligned multimodal frames."""
    _target: Type = field(default_factory=lambda: RawMultimodalUnalignedDataset)

class BaseDataset(Dataset):
    """
    Base class for datasets

    Args:
        config: Configuration for the dataset.
        modalities: List of modalities to load.
        data_dir: Directory where the data is stored.
    """

    def __init__(
            self,
            config: BaseDatasetConfig,
            modalities: Tuple[str, ...],
            data_dir: str,
    ):
        self.config = config
        self.data_dir = data_dir
        self.modalities = modalities

        self.metadata = load_from_json(os.path.join(self.data_dir, 'meta_data.json'))
        self.data = {}
        self.scene_box = {}

    def load_data(self):
        """
        Load data from data_dir.
        It loads frames, camera poses, camera intrinsics and all the metadata contained in meta_data.json
        """
        # Load world to GT
        self.w2gt = torch.tensor(self.metadata["worldtogt"])

        # Load modalities
        for mod in self.modalities:
            if mod in ['rgb', 'multispectral', 'infrared', 'mono', 'polarization']:
                self.load_generic(mod)
            else:
                print(f"modality {mod} not supported!")

        # Load bounding box
        self.load_bounding_box()

    def load_bounding_box(self):
        """
        Load the bounding box defining the Region of Interest and instantiate the correct SceneBox
        """
        if self.metadata['scene_box']['collider_type'] == 'sphere':
            self.scene_box = SceneBox(
                aabb=self.metadata['scene_box']['radius'] * torch.tensor([[-1, -1, -1], [1, 1, 1]]),
                collider_type=self.metadata['scene_box']['collider_type'],
                radius=self.metadata['scene_box']['radius'],
            )
        elif self.metadata['scene_box']['collider_type'] == 'near_far':
            self.scene_box = SceneBox(
                aabb=torch.tensor(self.metadata['scene_box']['aabb']),
                collider_type=self.metadata['scene_box']['collider_type'],
                near=self.metadata['scene_box']['near'],
                far=self.metadata['scene_box']['far'],
            )
        elif self.metadata['scene_box']['collider_type'] == 'box':
            self.scene_box = SceneBox(
                aabb=torch.tensor(self.metadata['scene_box']['aabb']),
                collider_type=self.metadata['scene_box']['collider_type'],
            )
        else:
            print(f"Collider {self.metadata['scene_box']['collider_type']} not supported.")
            sys.exit(1)

    def load_generic(self, modality: str):
        """
        Load generic modality
        """
        raise NotImplementedError

    def get_modality_list(self):
        """
        Return the list of modalities currently loaded.
        """
        return self.modalities

    def get_channels_per_modality(self):
        """Returns a dictionary with modality name - number of channels pairs."""
        channels = {}
        for mod, data in self.data.items():
            channels[mod] = data['images'].shape[-1]
        return channels

    def get_unique_views(self):
        """
        Returns the list of all the views in the dataset, even if some of them have only a subset of modality frames.
        """
        unique_views = set([])
        if isinstance(self.indexes, dict):
            for mod in self.modalities:
                unique_views = unique_views.union(set(self.indexes[mod]))
        else:
            unique_views = set(self.indexes)
        unique_views = list(unique_views)
        unique_views.sort()
        return unique_views

    def dynamic_get(self, item):
        """
        Returns the data of a specific view, even if some of the modalities are not available. If a modality is not
        available, the corresponding image and index are set to None.
        """
        return self.__getitem__(item)

class RawDataset(BaseDataset):
    """
    Base class for raw datasets.
    Consider all the modalities belonging to the same macro-view for either training or testing.
    To be extended.

    Args:
        config: Configuration for the dataset.
        modalities: List of modalities to load.
        data_dir: Directory where the data is stored.
    """

    def __init__(
            self,
            config: RawDatasetConfig,
            modalities: Tuple[str, ...],
            data_dir: str,
    ):
        super().__init__(config=config, modalities=modalities, data_dir=data_dir)
        self.config = config

        assert self.metadata["raw"], "Dataset frames are not raw."
        self.mosaick_pattern_per_modality = {mod: torch.tensor(self.metadata["modalities"][mod]["mosaick_pattern"])
                                            for mod in self.modalities}
        self.mosaick_mask_per_modality = self.build_mosaick_mask(self.mosaick_pattern_per_modality)
        self.mosaick_mask_across_modalities = self.build_mosaick_mask_across_modalities(
            mosaick_mask_per_modality=self.mosaick_mask_per_modality,
            mosaick_pattern_per_modality=self.mosaick_pattern_per_modality
        )

    def build_mosaick_mask(self, mosaick_pattern_per_modality) -> Dict[str, TensorType]:
        """
        Build the mosaick masks for each modality frame.

        Args:
            mosaick_pattern_per_modality: Dictionary with the mosaick patterns for each modality.

        Returns:
            mosaick_mask_per_modality: Dictionary with the mosaick masks for each modality.
                                       The dictionary is of the form:
                                       {
                                           mod: mosaick_mask
                                       }
        """
        mosaick_mask_per_modality = {}
        for mod, mosaick_pattern in mosaick_pattern_per_modality.items():
            w, h = self.metadata["modalities"][mod]["width"], self.metadata["modalities"][mod]["height"]
            n_width, n_height = math.ceil(w / mosaick_pattern.shape[1]), math.ceil(h / mosaick_pattern.shape[0])
            mosaick_pattern = mosaick_pattern.repeat((n_height, n_width))
            mosaick_pattern = mosaick_pattern[:h, :w]
            mosaick_mask_per_modality[mod] = mosaick_pattern.type(torch.int8)
        return mosaick_mask_per_modality

    def build_mosaick_mask_across_modalities(
            self,
            mosaick_mask_per_modality,
            mosaick_pattern_per_modality
    ) -> Dict[str, Dict[str, TensorType]]:
        """
        Build the mosaick masks for each modality for all the frame shapes of all the available modalities.

        Args:
            mosaick_mask_per_modality: Dictionary with the mosaick masks for each modality.
            mosaick_pattern_per_modality: Dictionary with the mosaick patterns for each modality.

        Returns:
            mosaick_mask_across_modalities: Dictionary with the mosaick masks for each modality for all the frame shapes

                                            E.g.: For a set of 2 modalities [mod1, mod2], the dictionary is of the form:
                                            {
                                                mod1: {
                                                    mod1: mosaick mask of mod1 with mod1.shape
                                                    mod2: mosaick mask of mod2 with mod1.shape
                                                }
                                                mod2: {
                                                    mod1: mosaick mask of mod1 with mod2.shape
                                                    mod2: mosaick mask of mod2 with mod2.shape
                                                }
                                            }
        """
        mosaick_mask_across_modalities = {}
        for mod_mask, current_modality_mosaick_mask in mosaick_mask_per_modality.items():
            masks = {}
            (h, w) = current_modality_mosaick_mask.shape
            for mod_pattern, current_modality_mosaick_pattern in mosaick_pattern_per_modality.items():
                x_times = w // current_modality_mosaick_pattern.shape[1] + 1
                y_times = h // current_modality_mosaick_pattern.shape[0] + 1
                if mod_mask != mod_pattern:
                    modality_mosaick_mask = current_modality_mosaick_pattern.repeat((y_times, x_times))
                    modality_mosaick_mask = modality_mosaick_mask[:h, :w]
                    modality_mosaick_mask = modality_mosaick_mask.type(torch.int8)
                    masks[mod_pattern] = modality_mosaick_mask
                else:
                    masks[mod_pattern] = current_modality_mosaick_mask.type(torch.int8)
            mosaick_mask_across_modalities[mod_mask] = masks
        return mosaick_mask_across_modalities

    def get_channels_per_modality(self) -> Dict[str, int]:
        """Return the number of channels per modality."""
        channels = {}
        for mod, mosaick_pattern in self.mosaick_pattern_per_modality.items():
            channels[mod] = len(torch.unique(mosaick_pattern))
        return channels

class BaseAlignedDataset(BaseDataset):
    """
    Base class for aligned datasets.
    Consider all the modalities belonging to the same macro-view for either training or testing.
    To be extended.

    Args:
        config: Configuration for the dataset.
        modalities: List of modalities to load.
        data_dir: Directory where the data is stored.
        indexes_to_choose: List of indexes to choose from the dataset.
        indexes_to_exclude: List of indexes to exclude from the dataset.
        indexes_to_exclude_ratio: Ratio of indexes to exclude from the dataset.
    """

    def __init__(
            self,
            config: BaseAlignedDatasetConfig,
            modalities: Tuple[str, ...],
            data_dir: str,
            indexes_to_choose: Tuple[int, ...] = None,
            indexes_to_exclude: Tuple[int, ...] = None,
            indexes_to_exclude_ratio: float = 0.0,
    ):
        super().__init__(config=config, modalities=modalities, data_dir=data_dir)

        # Defining indexes to load
        mod = self.modalities[0]
        if indexes_to_choose is not None:
            self.indexes = indexes_to_choose
        elif indexes_to_exclude is not None:
            self.indexes = [i for i in range(len(self.metadata["modalities"][mod]["frames"]))
                            if i not in indexes_to_exclude]
        elif indexes_to_exclude_ratio > 0:
            all_indexes = list(range(len(self.metadata["modalities"][mod]["frames"])))
            indexes_to_exclude = random.sample(
                population=all_indexes,
                k=int(len(self.metadata["modalities"][mod]["frames"]) * indexes_to_exclude_ratio)
            )
            self.indexes = [i for i in range(len(self.metadata["modalities"][mod]["frames"]))
                            if i not in indexes_to_exclude]
        else:
            self.indexes = list(range(len(self.metadata["modalities"][mod]["frames"])))
        self.indexes.sort()

        self.load_data()

    def __len__(self):
        """Return the number of loaded frames in the current dataset"""
        return self.data[self.modalities[0]]['images'].shape[0]

    def __getitem__(self, item):
        data = {}
        for mod in self.modalities:
            image = self.data[mod]['images'][item]
            data[mod] = {
                'index': torch.tensor(item, dtype=torch.int16),
                'images': image,
            }
        return data

class BaseUnalignedDataset(BaseDataset):
    """
    Base class for unaligned datasets.
    Consider each modality belonging the same macro-view for either training or testing, independently.
    To be extended.

    Args:
        config: Configuration for the dataset.
        modalities: List of modalities to load.
        data_dir: Directory where the data is stored.
        indexes_to_choose_per_modality: Dictionary with the indexes to choose for each modality.
                                        The dictionary is of the form:
                                        {
                                            mod1: [index1, index2, ...],
                                            mod2: [index3, index4, ...],
                                            ...
                                        }
        indexes_to_exclude_per_modality: Dictionary with the indexes to exclude for each modality.
                                          The dictionary is of the form:
                                          {
                                              mod1: [index1, index2, ...],
                                              mod2: [index3, index4, ...],
                                              ...
                                          }
    """

    def __init__(
            self,
            config: BaseUnalignedDatasetConfig,
            modalities: Tuple[str, ...],
            data_dir: str,
            indexes_to_choose_per_modality: Dict[str, Tuple[int, ...]] = None,
            indexes_to_exclude_per_modality: Dict[str, Tuple[int, ...]] = None,
    ):
        super().__init__(config=config, modalities=modalities, data_dir=data_dir)

        # Defining indexes to load
        self.indexes = {}
        for mod in self.modalities:
            if indexes_to_choose_per_modality is not None:
                self.indexes[mod] = indexes_to_choose_per_modality[mod]
            elif indexes_to_exclude_per_modality is not None:
                self.indexes[mod] = [i for i in range(len(self.metadata["modalities"][mod]["frames"]))
                                     if i not in indexes_to_exclude_per_modality[mod]]
            else:
                self.indexes[mod] = list(range(len(self.metadata["modalities"][mod]["frames"])))
            self.indexes[mod].sort()

        self.load_data()

    def __len__(self):
        return max([len(self.indexes[mod]) for mod in self.modalities])

    def __getitem__(self, item):
        data = {}
        for mod in self.modalities:
            image = self.data[mod]['images'][item] if item < len(self.indexes[mod]) else None
            data[mod] = {
                'index': torch.tensor(item, dtype=torch.int16),
                'images': image,
            }
        return data

    def dynamic_get(self, item):
        """
        Returns the data of a specific view, even if some of the modalities are not available. If a modality is not
        available, the corresponding image and index are set to None.
        """
        data = {}
        unique_views = self.get_unique_views()
        view_id = unique_views[item]
        for mod in self.modalities:
            idx = self.indexes[mod].index(view_id) if view_id in self.indexes[mod] else None
            image = self.data[mod]['images'][idx] if idx is not None else None
            data[mod] = {
                'index': torch.tensor(idx, dtype=torch.int16) if idx is not None else None,
                'images': image,
            }
        return data

class MultimodalAlignedDataset(BaseAlignedDataset):
    """
    Dataset class for aligned multimodal frames. With "aligned" we mean that we dispose of all the modalities belonging
    to the same view, even if not perfectly aligned.

    Args:
        config: Configuration for the dataset.
        modalities: List of modalities to load.
        data_dir: Directory where the data is stored.
        indexes_to_choose: List of indexes to choose from the dataset.
        indexes_to_exclude: List of indexes to exclude from the dataset.
        indexes_to_exclude_ratio: Ratio of indexes to exclude from the dataset.
    """

    def __init__(
            self,
            config: MultimodalAlignedDatasetConfig,
            modalities: Tuple[str, ...],
            data_dir: str,
            indexes_to_choose: Tuple[int, ...] = None,
            indexes_to_exclude: Tuple[int, ...] = None,
            indexes_to_exclude_ratio: float = 0.0,
    ):
        super().__init__(
            config=config,
            modalities=modalities,
            data_dir=data_dir,
            indexes_to_choose=indexes_to_choose,
            indexes_to_exclude=indexes_to_exclude,
            indexes_to_exclude_ratio=indexes_to_exclude_ratio,
        )
        self.load_data()

    def load_data(self):
        """
        Load data from data_dir. Swap BGR channels to RGB.
        """
        super().load_data()
        if "rgb" in self.modalities:
            self.data["rgb"]["images"] = self.data["rgb"]["images"][..., [2, 1, 0]]

    def load_generic(self, modality: str):
        """
        Load generic modality from data_dir.
        """
        images = []
        c2ws = []
        indexes = []

        for frame in self.metadata["modalities"][modality]["frames"]:
            idx = frame["frame_id"]
            if idx not in self.indexes:
                continue
            indexes.append(idx)
            frame_path = os.path.join(self.data_dir, 'modalities', modality, frame["file_name"])
            img = read_frame(frame_path)
            if img.max() > 1:
                img = normalize_frame(img)
            img = torch.tensor(img, dtype=torch.float32)
            images.append(img)
            camtoworld = frame["camtoworld"]
            c2ws.append(torch.tensor(camtoworld))
        indexes = sorted(range(len(indexes)), key=lambda k: indexes[k])
        images = [images[i] for i in indexes]
        c2ws = [c2ws[i] for i in indexes]
        c2ws = torch.stack(c2ws)

        num_cameras = len(c2ws)

        cameras = Cameras(
            fx=torch.tensor(self.metadata["modalities"][modality]["fx"]).expand((num_cameras, 1)),
            fy=torch.tensor(self.metadata["modalities"][modality]["fy"]).expand((num_cameras, 1)),
            cx=torch.tensor(self.metadata["modalities"][modality]["cx"]).expand((num_cameras, 1)),
            cy=torch.tensor(self.metadata["modalities"][modality]["cy"]).expand((num_cameras, 1)),
            height=torch.tensor(self.metadata["modalities"][modality]["height"]).expand((num_cameras, 1)),
            width=torch.tensor(self.metadata["modalities"][modality]["width"]).expand((num_cameras, 1)),
            camera_type=CAMERA_MODEL_TO_TYPE[self.metadata["modalities"][modality]["camera_model"]],
            distortion_params=torch.tensor(
                self.metadata["modalities"][modality]["distortion_params"]
            ).expand((num_cameras, 6)) if not self.metadata["undistorted"] else None,
            camera_to_worlds=c2ws,
        )

        self.data[modality] = {}
        self.data[modality]['images'] = torch.stack(images)
        self.data[modality]['cameras'] = cameras

class MultimodalUnalignedDataset(BaseUnalignedDataset):
    """
    Dataset class for unaligned multimodal frames. With "unaligned" we mean that we can choose which modality frames
    of which views are considered to be training or test frames independently.
    """

    def __init__(
            self,
            config: MultimodalUnalignedDatasetConfig,
            modalities: Tuple[str, ...],
            data_dir: str,
            indexes_to_choose_per_modality: Dict[str, Tuple[int, ...]] = None,
            indexes_to_exclude_per_modality: Dict[str, Tuple[int, ...]] = None,
    ):
        super().__init__(
            config=config,
            modalities=modalities,
            data_dir=data_dir,
            indexes_to_choose_per_modality=indexes_to_choose_per_modality,
            indexes_to_exclude_per_modality=indexes_to_exclude_per_modality,
        )
        self.load_data()

    def load_data(self):
        """
        Load data from data_dir. Swap BGR channels to RGB.
        """
        super().load_data()
        if "rgb" in self.modalities:
            self.data["rgb"]["images"] = self.data["rgb"]["images"][..., [2, 1, 0]] # BGR to RGB

    def load_generic(self, modality: str):
        """
        Load generic modality from data_dir.
        """
        images = []
        c2ws = []
        indexes = []

        for frame in self.metadata["modalities"][modality]["frames"]:
            idx = frame["frame_id"]
            if idx not in self.indexes[modality]:
                continue
            indexes.append(idx)
            frame_path = os.path.join(self.data_dir, 'modalities', modality, frame["file_name"])
            img = read_frame(frame_path)
            if img.max() > 1:
                img = normalize_frame(img)
            img = torch.tensor(img, dtype=torch.float32)
            images.append(img)
            camtoworld = frame["camtoworld"]
            c2ws.append(torch.tensor(camtoworld))
        indexes = sorted(range(len(indexes)), key=lambda k: indexes[k])
        images = [images[i] for i in indexes]
        c2ws = [c2ws[i] for i in indexes]
        c2ws = torch.stack(c2ws)

        num_cameras = len(c2ws)

        cameras = Cameras(
            fx=torch.tensor(self.metadata["modalities"][modality]["fx"]).expand((num_cameras, 1)),
            fy=torch.tensor(self.metadata["modalities"][modality]["fy"]).expand((num_cameras, 1)),
            cx=torch.tensor(self.metadata["modalities"][modality]["cx"]).expand((num_cameras, 1)),
            cy=torch.tensor(self.metadata["modalities"][modality]["cy"]).expand((num_cameras, 1)),
            height=torch.tensor(self.metadata["modalities"][modality]["height"]).expand((num_cameras, 1)),
            width=torch.tensor(self.metadata["modalities"][modality]["width"]).expand((num_cameras, 1)),
            camera_type=CAMERA_MODEL_TO_TYPE[self.metadata["modalities"][modality]["camera_model"]],
            distortion_params=torch.tensor(
                self.metadata["modalities"][modality]["distortion_params"]
            ).expand((num_cameras, 6)) if not self.metadata["undistorted"] else None,
            camera_to_worlds=c2ws,
        )

        self.data[modality] = {}
        self.data[modality]['images'] = torch.stack(images)
        self.data[modality]['cameras'] = cameras

class RawMultimodalAlignedDataset(MultimodalAlignedDataset, RawDataset):
    """
    Dataset class for aligned multimodal raw frames.
    """

    def __init__(
            self,
            config,
            modalities: Tuple[str, ...],
            data_dir: str,
            indexes_to_choose: Tuple[int, ...] = None,
            indexes_to_exclude: Tuple[int, ...] = None,
            indexes_to_exclude_ratio: float = 0.0,
    ):
        super().__init__(
            config=config,
            modalities=modalities,
            data_dir=data_dir,
            indexes_to_choose=indexes_to_choose,
            indexes_to_exclude=indexes_to_exclude,
            indexes_to_exclude_ratio=indexes_to_exclude_ratio
        )

    def load_data(self):
        """Load data from data_dir."""
        super(MultimodalAlignedDataset, self).load_data()

class RawMultimodalUnalignedDataset(MultimodalUnalignedDataset, RawDataset):
    """
    Dataset class for unaligned multimodal raw frames.
    """

    def __init__(
            self,
            config: RawMultimodalUnalignedDatasetConfig,
            modalities: Tuple[str, ...],
            data_dir: str,
            indexes_to_choose_per_modality: Dict[str, Tuple[int, ...]] = None,
            indexes_to_exclude_per_modality: Dict[str, Tuple[int, ...]] = None,
    ):
        super().__init__(
            config=config,
            modalities=modalities,
            data_dir=data_dir,
            indexes_to_choose_per_modality=indexes_to_choose_per_modality,
            indexes_to_exclude_per_modality=indexes_to_exclude_per_modality,
        )

    def load_data(self):
        """Load data from data_dir."""
        super(MultimodalUnalignedDataset, self).load_data()
