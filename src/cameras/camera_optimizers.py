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

"""Camera optimizers"""

import sys
from dataclasses import dataclass, field
from typing import Type, Literal, Dict

import torch
from torchtyping import TensorType

from cameras.lie_groups import exp_map_SE3, exp_map_SO3xR3
from configs.configs import InstantiateConfig
from engine.schedulers import SchedulerConfig

@dataclass
class CameraOptimizerConfig(InstantiateConfig):
    """Configuration class for camera pose optimizer."""

    _target: Type = field(default_factory=lambda: CameraOptimizer)
    mode: Literal["off", "SO3xR3", "SE3"] = "off"
    """Pose optimization strategy to use. If enabled, we recommend SO3xR3."""
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    """Learning rate scheduler for camera optimizer.."""
    modalities_to_optimize: dict[str, bool] = field(default_factory=dict)
    """List of modalities to optimize"""
    shared_optimization: bool = False
    """Whether to optimize relative poses by assuming that the same transformation applies to all views."""

class CameraOptimizer(torch.nn.Module):
    """Module that optimizes the camera poses. It accepts modes "off", "SO3xR3", "SE3"."""

    config: CameraOptimizerConfig

    def __init__(
            self,
            config: CameraOptimizerConfig,
            num_cameras: int,
            **kwargs,  # pylint: disable=unused-argument
    ) -> None:
        super().__init__()
        self.config = config
        self.num_cameras = num_cameras

        if self.config.mode == "SO3xR3":
            self.exp_map = exp_map_SO3xR3
        elif self.config.mode == "SE3":
            self.exp_map = exp_map_SE3
        elif self.config.mode == "off":
            pass
        else:
            raise ValueError(f"Camera optimization mode {self.config.mode} not supported.")

        # Initialize learnable parameters.
        self.pose_adjustment = torch.nn.ParameterDict([])
        if self.config.mode == "off":
            pass
        elif self.config.mode in ("SO3xR3", "SE3"):
            for mod in self.config.modalities_to_optimize.keys():
                if self.config.shared_optimization:
                    self.pose_adjustment[mod] = torch.nn.Parameter(torch.zeros((1, 6)))
                else:
                    self.pose_adjustment[mod] = torch.nn.Parameter(torch.zeros((self.num_cameras, 6)))
        else:
            print(f"Camera optimization mode {self.config.mode} not supported.")
            sys.exit(1)

    def forward(
            self,
            camera_indices: Dict[str, TensorType["num_rays", 3]],
    ) -> Dict[str, TensorType["num_cameras", 3, 4]]:
        """Indexing into camera adjustments.
        Args:
            camera_indices: indices of Cameras to optimize.
        Returns:
            Tranformation matrices from optimized camera coordinates
            to given camera coordinates.
        """

        outputs = {}

        for mod, indices in camera_indices.items():
            if indices is None:
                continue
            if self.config.mode == "off":
                # Note that using repeat() instead of tile() here would result in unnecessary copies.
                mat = torch.eye(4, device=indices.device)[None, :3, :4].tile(indices.shape[0], 1, 1)
            else:
                # Apply learned transformation delta.
                if self.config.shared_optimization:
                    parameters = self.pose_adjustment[mod].expand((self.num_cameras, 6))[indices[:,0]]
                else:
                    parameters = self.pose_adjustment[mod][indices[:,0]]
                mat = self.exp_map(parameters)

            if not self.config.modalities_to_optimize[mod]:
                mat = mat.detach()

            outputs[mod] = mat

        return outputs

    def forward_single_modality(
            self,
            camera_indices: Dict[str, TensorType["num_rays", 1]],
            modality: str,
    ) -> TensorType["num_cameras", 3, 4]:
        """Indexing into camera adjustments for a single modality."""
        indices = camera_indices[modality]
        output = self.forward({modality: indices.view(-1, 1).expand(indices.shape[0], 3)})
        return output[modality]

    def set_num_cameras(self, num_cameras: int):
        """Set the number of cameras."""
        self.num_cameras = num_cameras
