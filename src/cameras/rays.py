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
Some ray datastructures.
"""
import random
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import torch
from torchtyping import TensorType

from utils.math import Gaussians, conical_frustum_to_gaussian
from utils.tensor_dataclass import TensorDataclass


@dataclass
class Frustums(TensorDataclass):
    """Describes region of space as a frustum."""

    origins: TensorType["bs":..., 3]
    """xyz coordinate for ray origin."""
    directions: TensorType["bs":..., 3]
    """Direction of ray."""
    starts: TensorType["bs":..., 1]
    """Where the frustum starts along a ray."""
    ends: TensorType["bs":..., 1]
    """Where the frustum ends along a ray."""
    pixel_area: TensorType["bs":..., 1]
    """Projected area of pixel a distance 1 away from origin."""
    up_directions: Optional[TensorType["bs":..., 3]] = None
    """Camera up direction for each ray."""
    offsets: Optional[TensorType["bs":..., 3]] = None
    """Offsets for each sample position"""

    def get_positions(self) -> TensorType[..., 3]:
        """Calulates "center" position of frustum. Not weighted by mass.

        Returns:
            xyz positions.
        """
        pos = self.origins + self.directions * (self.starts + self.ends) / 2
        if self.offsets is not None:
            pos = pos + self.offsets
        return pos

    def set_offsets(self, offsets):
        """Sets offsets for this frustum for computing positions"""
        self.offsets = offsets

    def get_start_positions(self) -> TensorType[..., 3]:
        """Calulates "start" position of frustum. We use start positions for MonoSDF
        because when we use error bounded sampling, we need to upsample many times.
        It's hard to merge two set of ray samples while keeping the mid points fixed.
        Every time we up sample the points the mid points will change and
        therefore we need to evaluate all points again which is 3 times slower.
        But we can skip the evaluation of sdf value if we use start position instead of mid position
        because after we merge the points, the starting point is the same and only the delta is changed.

        Returns:
            xyz positions.
        """
        return self.origins + self.directions * self.starts

    def get_gaussian_blob(self) -> Gaussians:
        """Calculates guassian approximation of conical frustum.

        Resturns:
            Conical frustums approximated by gaussian distribution.
        """
        # Cone radius is set such that the square pixel_area matches the cone area.
        cone_radius = torch.sqrt(self.pixel_area) / 1.7724538509055159  # r = sqrt(pixel_area / pi)
        if self.offsets is not None:
            raise NotImplementedError()
        return conical_frustum_to_gaussian(
            origins=self.origins,
            directions=self.directions,
            starts=self.starts,
            ends=self.ends,
            radius=cone_radius,
        )

    @classmethod
    def get_mock_frustum(cls, device="cpu") -> "Frustums":
        """Helper function to generate a placeholder frustum.

        Returns:
            A size 1 frustum with meaningless values.
        """
        return Frustums(
            origins=torch.ones((1, 3)).to(device),
            directions=torch.ones((1, 3)).to(device),
            starts=torch.ones((1, 1)).to(device),
            ends=torch.ones((1, 1)).to(device),
            pixel_area=torch.ones((1, 1)).to(device),
        )


@dataclass
class RaySamples(TensorDataclass):
    """Samples along a ray"""

    frustums: Frustums
    """Frustums along ray."""
    camera_indices: Optional[TensorType["bs":..., 1]] = None
    """Camera index."""
    deltas: Optional[TensorType["bs":..., 1]] = None
    """"width" of each sample."""
    spacing_starts: Optional[TensorType["bs":..., "num_samples", 1]] = None
    """Start of normalized bin edges along ray [0,1], before warping is applied, ie. linear in disparity sampling."""
    spacing_ends: Optional[TensorType["bs":..., "num_samples", 1]] = None
    """Start of normalized bin edges along ray [0,1], before warping is applied, ie. linear in disparity sampling."""
    spacing_to_euclidean_fn: Optional[Callable] = None
    """Function to convert bins to euclidean distance."""
    metadata: Optional[Dict[str, TensorType["bs":..., "latent_dims"]]] = None
    """addtional information relevant to generating ray samples"""
    times: Optional[TensorType[..., 1]] = None
    """Times at which rays are sampled"""

    def get_alphas(self, densities: TensorType[..., "num_samples", 1]) -> TensorType[..., "num_samples", 1]:
        """Return weights based on predicted densities

        Args:
            densities: Predicted densities for samples along ray

        Returns:
            Weights for each sample
        """

        delta_density = self.deltas * densities
        alphas = 1 - torch.exp(-delta_density)

        return alphas

    def get_weights(self, densities: TensorType[..., "num_samples", 1]) -> TensorType[..., "num_samples", 1]:
        """Return weights based on predicted densities

        Args:
            densities: Predicted densities for samples along ray

        Returns:
            Weights for each sample
        """

        delta_density = self.deltas * densities
        alphas = 1 - torch.exp(-delta_density)

        transmittance = torch.cumsum(delta_density[..., :-1, :], dim=-2)
        transmittance = torch.cat(
            [torch.zeros((*transmittance.shape[:1], 1, 1), device=densities.device), transmittance], dim=-2
        )
        transmittance = torch.exp(-transmittance)  # [..., "num_samples"]

        weights = alphas * transmittance  # [..., "num_samples"]

        return weights

    def get_weights_and_transmittance(
        self, densities: TensorType[..., "num_samples", 1]
    ) -> Tuple[TensorType[..., "num_samples", 1], TensorType[..., "num_samples", 1]]:
        """Return weights and transmittance based on predicted densities

        Args:
            densities: Predicted densities for samples along ray

        Returns:
            Weights and transmittance for each sample
        """

        delta_density = self.deltas * densities
        alphas = 1 - torch.exp(-delta_density)

        transmittance = torch.cumsum(delta_density[..., :-1, :], dim=-2)
        transmittance = torch.cat(
            [torch.zeros((*transmittance.shape[:1], 1, 1), device=densities.device), transmittance], dim=-2
        )
        transmittance = torch.exp(-transmittance)  # [..., "num_samples"]

        weights = alphas * transmittance  # [..., "num_samples"]

        return weights, transmittance

    def get_weights_from_alphas(self, alphas: TensorType[..., "num_samples", 1]) -> TensorType[..., "num_samples", 1]:
        """Return weights based on predicted alphas

        Args:
            alphas: Predicted alphas (maybe from sdf) for samples along ray

        Returns:
            Weights for each sample
        """

        transmittance = torch.cumprod(
            torch.cat([torch.ones((*alphas.shape[:1], 1, 1), device=alphas.device), 1.0 - alphas + 1e-7], 1), 1
        )  # [..., "num_samples"]

        weights = alphas * transmittance[:, :-1, :]  # [..., "num_samples"]

        return weights

    def get_weights_and_transmittance_from_alphas(
        self, alphas: TensorType[..., "num_samples", 1]
    ) -> TensorType[..., "num_samples", 1]:
        """Return weights based on predicted alphas

        Args:
            alphas: Predicted alphas (maybe from sdf) for samples along ray

        Returns:
            Weights for each sample
        """

        transmittance = torch.cumprod(
            torch.cat([torch.ones((*alphas.shape[:1], 1, 1), device=alphas.device), 1.0 - alphas + 1e-7], 1), 1
        )  # [..., "num_samples"]

        weights = alphas * transmittance[:, :-1, :]  # [..., "num_samples"]

        return weights, transmittance


@dataclass
class RayBundle(TensorDataclass):
    """A bundle of ray parameters."""

    camera_indices: TensorType[..., 1]
    """Camera indices"""
    origins: TensorType[..., 3]
    """Ray origins (XYZ)"""
    directions: TensorType[..., 3]
    """Unit ray direction vector"""
    up_directions: Optional[TensorType[..., 3]] = None
    """Camera up direction for each ray"""
    pixel_area: TensorType[..., 1] = None
    """Projected area of pixel a distance 1 away from origin"""
    directions_norm: Optional[TensorType[..., 1]] = None
    """Norm of ray direction vector before normalization"""
    nears: Optional[TensorType[..., 1]] = None
    """Distance along ray to start sampling"""
    fars: Optional[TensorType[..., 1]] = None
    """Rays Distance along ray to stop sampling"""
    metadata: Optional[Dict[str, TensorType["num_rays", "latent_dims"]]] = None
    """Additional metadata or data needed for interpolation, will mimic shape of rays"""
    times: Optional[TensorType[..., 1]] = None
    """Times at which rays are sampled"""


    def set_camera_indices(self, camera_index: int) -> None:
        """Sets all of the the camera indices to a specific camera index.

        Args:
            camera_index: Camera index.
        """
        self.camera_indices = torch.ones_like(self.origins[..., 0:1]).long() * camera_index

    def __len__(self):
        num_rays = torch.numel(self.origins) // self.origins.shape[-1]
        return num_rays

    def sample(self, num_rays: int) -> "RayBundle":
        """Returns a RayBundle as a subset of rays.

        Args:
            num_rays: Number of rays in output RayBundle

        Returns:
            RayBundle with subset of rays.
        """
        assert num_rays <= len(self)
        indices = random.sample(range(len(self)), k=num_rays)
        return self[indices]

    def get_row_major_sliced_ray_bundle(self, start_idx: int, end_idx: int) -> "RayBundle":
        """Flattens RayBundle and extracts chunk given start and end indicies.

        Args:
            start_idx: Start index of RayBundle chunk.
            end_idx: End index of RayBundle chunk.

        Returns:
            Flattened RayBundle with end_idx-start_idx rays.

        """
        return self.flatten()[start_idx:end_idx]

    def get_ray_samples(
        self,
        bin_starts: TensorType["bs":..., "num_samples", 1],
        bin_ends: TensorType["bs":..., "num_samples", 1],
        spacing_starts: Optional[TensorType["bs":..., "num_samples", 1]] = None,
        spacing_ends: Optional[TensorType["bs":..., "num_samples", 1]] = None,
        spacing_to_euclidean_fn: Optional[Callable] = None,
    ) -> RaySamples:
        """Produces samples for each ray by projection points along the ray direction. Currently samples uniformly.

        Args:
            bin_starts: Distance from origin to start of bin.
            bin_ends: Distance from origin to end of bin.

        Returns:
            Samples projected along ray.
        """
        deltas = bin_ends - bin_starts
        if self.camera_indices is not None:
            camera_indices = self.camera_indices[..., None]
        else:
            camera_indices = None

        shaped_raybundle_fields = self[..., None]

        frustums = Frustums(
            origins=shaped_raybundle_fields.origins,  # [..., 1, 3]
            directions=shaped_raybundle_fields.directions,  # [..., 1, 3]
            up_directions=shaped_raybundle_fields.up_directions,  # [..., 1, 3]
            starts=bin_starts,  # [..., num_samples, 1]
            ends=bin_ends,  # [..., num_samples, 1]
            pixel_area=shaped_raybundle_fields.pixel_area,  # [..., 1, 1]
        )

        ray_samples = RaySamples(
            frustums=frustums,
            camera_indices=camera_indices,  # [..., 1, 1]
            deltas=deltas,  # [..., num_samples, 1]
            spacing_starts=spacing_starts,  # [..., num_samples, 1]
            spacing_ends=spacing_ends,  # [..., num_samples, 1]
            spacing_to_euclidean_fn=spacing_to_euclidean_fn,
            metadata=shaped_raybundle_fields.metadata,
            times=None if self.times is None else self.times[..., None],  # [..., 1, 1]
        )

        return ray_samples
