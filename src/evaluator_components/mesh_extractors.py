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
Mesh extractor class.
"""

import os.path
from dataclasses import dataclass, field
from typing import Type

from configs.configs import InstantiateConfig
from data.scene_box import SceneBox
from utils.marching_cubes import get_surface_sliding

@dataclass
class MeshExtractorConfig(InstantiateConfig):
    """
    Mesh extractor config.
    """

    _target: Type = field(default_factory=lambda: MeshExtractor)
    """Target class to instantiate."""
    resolution: int = 512
    """Mesh resolution"""
    marching_cube_threshold: float = 0.0
    """marching cube threshold"""
    gt_scale: bool = True
    """Export the mesh in GT scale"""

class MeshExtractor:
    """
    Mesh extractor class.

    Args:
        config: Configuration for the mesh extractor.
        scene_box: SceneBox object containing the bounding box.
        w2gt: Transformation matrix from world to ground truth coordinates.
        output_path: Path to save the extracted mesh.
    """

    def __init__(
            self,
            config: MeshExtractorConfig,
            scene_box: SceneBox,
            w2gt,
            output_path: str,
    ):
        self.config = config
        self.bounding_box_min = scene_box.aabb[0]
        self.bounding_box_max = scene_box.aabb[1]
        self.w2gt = w2gt.cpu().numpy()
        self.output_path = output_path

    def extract(self, sdf_fn, step):
        """Generate and export the mesh to disk."""
        output_path = os.path.join(self.output_path, "meshes")
        os.makedirs(output_path, exist_ok=True)
        output_path = os.path.join(output_path, f"{step:08}.ply")
        mesh = get_surface_sliding(
            sdf_fn=lambda x: sdf_fn(x).squeeze().contiguous(),
            resolution=self.config.resolution,
            bounding_box_min=self.bounding_box_min.cpu(),
            bounding_box_max=self.bounding_box_max.cpu(),
            coarse_mask=None,
            output_path=output_path,
            return_mesh=True,
        )
        if self.config.gt_scale:
            mesh.apply_transform(self.w2gt)
        mesh.export(output_path)
