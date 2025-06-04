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
Scene Colliders
"""

from __future__ import annotations
from typing import Tuple, Dict

import torch
from torch import nn
from torchtyping import TensorType

from cameras.rays import RayBundle
from data.scene_box import SceneBox

class SceneCollider(nn.Module):
    """Module for setting near and far values for rays."""

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        super().__init__()

    def forward(self, ray_bundle: RayBundle) -> RayBundle:
        """Sets the nears and fars of the ray bundle."""
        raise NotImplementedError

class SphereCollider(SceneCollider):
    """Sets the nears and fars with intersection with sphere.

    Args:
        radius: radius of sphere
        soft_intersection: default False, we clamp the value if not intersection found
        if set to True, the distance between near and far is always  2*radius,
    """

    def __init__(self, radius: float = 1.0, soft_intersection=False, **kwargs) -> None:
        self.radius = radius
        self.soft_intersection = soft_intersection
        super().__init__(**kwargs)

    def forward(self, ray_bundle: RayBundle) -> Tuple[RayBundle, TensorType]:
        """update the nears and fars of the ray bundle with the intersection with the sphere."""
        ray_cam_dot = (ray_bundle.directions * ray_bundle.origins).sum(dim=-1, keepdims=True)
        under_sqrt = ray_cam_dot**2 - (ray_bundle.origins.norm(p=2, dim=-1, keepdim=True) ** 2 - self.radius**2)

        mask = (under_sqrt > 0.01).squeeze(dim=-1)

        # sanity check
        under_sqrt = under_sqrt.clamp_min(0.01)

        if self.soft_intersection:
            under_sqrt = torch.ones_like(under_sqrt) * self.radius

        sphere_intersections = (
                torch.sqrt(under_sqrt) * torch.Tensor([-1, 1]).float().to(under_sqrt.device) - ray_cam_dot
        )
        sphere_intersections = sphere_intersections.clamp_min(0.01)

        ray_bundle.nears = sphere_intersections[:, 0:1]
        ray_bundle.fars = sphere_intersections[:, 1:2]
        return ray_bundle, mask

class ColliderInstancer:
    """
    Instancer for the colliders.
    """

    def __init__(
            self,
            scene_box: SceneBox,
    ):
        if scene_box.collider_type == 'sphere':
            self.collider = SphereCollider(scene_box.radius)
        else:
            raise ValueError(f"No collider of type {scene_box.collider_type}. Exiting.")

    def update_ray_bundles(self, ray_bundles: Dict[str, RayBundle]):
        """Update the nears and fars of the ray bundles with the intersection with the sphere."""
        masks = {}
        for mod, ray_bundle in ray_bundles.items():
            if ray_bundle is not None:
                ray_bundle, mask = self.collider(ray_bundle)
                masks[mod] = mask
            else:
                masks[mod] = None
        return masks

    def update_ray_bundles_for_background(self, ray_bundles: Dict[str, RayBundle]):
        """Update the nears and fars of the ray bundles for background estimation"""
        for ray_bundle in ray_bundles.values():
            if ray_bundle is not None:
                ray_bundle, mask = self.collider(ray_bundle)
                ray_bundle.nears[mask] = ray_bundle.fars[mask]
                ray_bundle.fars = torch.ones_like(ray_bundle.fars) * ray_bundle.fars + 3.
