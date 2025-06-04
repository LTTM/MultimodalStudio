# Copyright 2025 Sony Group Corporation.
# All rights reserved.
#
# Licenced under the License reported at
#
#     https://github.com/LTTM/MultimodalStudio/LICENSE.txt (the "License").
#
# See the License for the specific language governing permissions and limitations under the License.
#
# Author: Federico Lincetto, Ph.D. Student at the University of Padova

"""
Camera pose extractor class.
"""

import os.path
from dataclasses import dataclass, field
from typing import Type, Dict

import numpy as np
import torch
from torch.utils.data import Dataset
import trimesh

from cameras.camera_optimizers import CameraOptimizer
from configs.configs import InstantiateConfig
import utils.poses as pose_utils

COLORS = {
    "red": [255, 0, 0],
    "green": [0, 255, 0],
    "blue": [0, 0, 255],
    "black": [0, 0, 0],
    "white": [255, 255, 255],
    "yellow": [255, 255, 0],
    "cyan": [0, 255, 255],
    "magenta": [255, 0, 255],
}

@dataclass
class PoseExtractorConfig(InstantiateConfig):
    """
    Pose extractor config.
    """
    _target: Type = field(default_factory=lambda: PoseExtractor)
    """Target class to instantiate."""
    colors: Dict = field(default_factory=lambda: {})
    """Color dictionary for each modality"""
    gt_scale: bool = True
    """Export the camera poses in GT scale"""

class PoseExtractor:
    """
    Camera pose extractor class.

    Args:
        config: Configuration for the pose extractor.
        dataset: Dataset object containing the data.
        pose_optimizer: Camera optimizer object.
        w2gt: Transformation matrix from world to ground truth coordinates.
        output_path: Path to save the extracted camera poses.
    """

    def __init__(
            self,
            config: PoseExtractorConfig,
            dataset: Dataset,
            pose_optimizer: CameraOptimizer,
            w2gt,
            output_path: str,
    ):
        self.config = config
        self.dataset = dataset
        self.pose_optimizer = pose_optimizer
        self.w2gt = w2gt.cpu().numpy()
        self.output_path = output_path

    def extract(self, step):
        """Export the camera poses to disk as ply point cloud."""
        output_path = os.path.join(self.output_path, "camera_poses")
        os.makedirs(output_path, exist_ok=True)
        output_path = os.path.join(output_path, f"poses_{step:08}.ply")

        pointcloud = []
        colors = []

        for modality in self.dataset.modalities:
            cameras = self.dataset.data[modality]['cameras']
            n_cameras = cameras.shape[0]
            c2ws = cameras.camera_to_worlds.detach().cpu()
            camera_opt_to_camera = self.pose_optimizer.forward_single_modality(
                camera_indices={modality: torch.arange(n_cameras)},
                modality=modality
            ).detach().cpu()
            c2ws = pose_utils.multiply(c2ws, camera_opt_to_camera)
            poses = torch.zeros(n_cameras, 4)
            poses[:, 3] = 1
            poses = torch.einsum("bij,bj->bi", c2ws, poses)
            pointcloud.append(poses.numpy())
            color = np.zeros((1, 3))
            color[0] = COLORS[self.config.colors[modality]]
            color = np.broadcast_to(color, (n_cameras, 3))
            colors.append(color)

        pointcloud = np.concatenate(pointcloud, axis=0)
        colors = np.concatenate(colors, axis=0)
        mesh = trimesh.PointCloud(pointcloud, colors=colors)
        if self.config.gt_scale:
            mesh.apply_transform(self.w2gt)
        mesh.export(output_path)
