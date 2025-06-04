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
Ray generator module
"""

from typing import Dict
from torchtyping import TensorType

from torch import nn

from cameras.camera_optimizers import CameraOptimizer
from cameras.rays import RayBundle
from utils import profiler

class RayGenerator(nn.Module):
    """Module for generating rays.
    This class is the interface between the scene's cameras/camera optimizer and the ray sampler.

    Args:
        data:
        pose_optimizer: pose optimization module, for optimizing noisy camera intrisics/extrinsics.
    """

    def __init__(self, data: Dict, pose_optimizer: CameraOptimizer, pixel_offset: float) -> None:
        super().__init__()
        self.cameras = {mod: mod_data['cameras'] for mod, mod_data in data.items()}
        self.pose_optimizer = pose_optimizer
        self.pixel_offset = pixel_offset
        self.image_coords = {
            mod: camera.get_image_coords(pixel_offset=pixel_offset)
            for mod, camera in self.cameras.items()
        }

    @profiler.time_function
    def forward(self, ray_indices: Dict[str, TensorType["num_rays", 3]]) -> Dict[str, RayBundle]:
        """Index into the cameras to generate the rays.

        Args:
            ray_indices: Contains camera, row, and col indices for target rays.
        """
        ray_bundles = {}
        camera_opt_to_camera = self.pose_optimizer(ray_indices)

        for mod, indices in ray_indices.items():
            if indices is None:
                ray_bundles[mod] = None
                continue
            c = indices[:, 0]  # camera indices
            y = indices[:, 1]  # row indices
            x = indices[:, 2]  # col indices
            coords = self.image_coords[mod]
            coords = coords[y, x]

            cameras = self.cameras[mod].to(camera_opt_to_camera[mod].device)
            ray_bundle = cameras.generate_rays(
                camera_indices=c.unsqueeze(-1),
                coords=coords,
                camera_opt_to_camera=camera_opt_to_camera[mod],
            )
            ray_bundles[mod] = ray_bundle

        return ray_bundles
