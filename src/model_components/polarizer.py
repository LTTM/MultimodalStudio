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
Polarization helper functions.
"""

from typing import Tuple
from torchtyping import TensorType

import numpy as np
import torch
from torch.nn import functional as F

def mueller_linear_polarizer(theta: TensorType["batch", 1]) -> TensorType[3, 3]:
    """
    Mueller matrix for a linear polarizer.
    Args:
        theta: Angle of the polarizer.
    Returns:
        Mueller matrix for a linear polarizer.
    """
    c = torch.cos(2 * theta)
    s = torch.sin(2 * theta)
    linear_polarizer = 0.5 * torch.stack(
        [torch.ones_like(c), c, s, c, c ** 2, c * s, s, c * s, s ** 2],
        dim=-1
    ).view(-1, 3, 3)
    return linear_polarizer

def mueller_rotate(theta: TensorType["batch", 1]) -> TensorType[3, 3]:
    """
    Mueller matrix for a rotation.
    Args:
        theta: Angle of the rotation.
    Returns:
        Mueller matrix for a rotation.
    """
    c = torch.cos(2 * theta)
    s = torch.sin(2 * theta)
    one = torch.ones_like(c)
    zero = torch.zeros_like(c)
    rotate = torch.stack([one, zero, zero, zero, c, s, zero, -s, c], dim=-1).view(-1, 3, 3)
    return rotate

def align_polarization_filters(
        stokes_vectors: TensorType["batch", 3],
        directions: TensorType["batch", 3],
        camera_up_directions: TensorType["batch", 3]
) -> TensorType["batch", 3]:
    """
    Function that aligns the stokes vectors to the camera reference system.
    Args:
        stokes_vectors: Stokes vector of the light.
        directions: Direction of the ray.
        normals: Normal of the surface.
        camera_up_directions: Polarization direction of the polarizer.
    Returns:
        Aligned stokes vectors.
    """
    reflection_plane_normal = F.normalize(
        torch.linalg.cross(
            directions,
            torch.tensor(
                [0., 0., 1.], device=directions.device, dtype=directions.dtype
            )[None, ...].expand(directions.shape)
        ),
        dim=-1
    )
    cos_theta = torch.clamp(torch.sum(reflection_plane_normal * camera_up_directions, dim=-1),  min=-1+1e-4, max=1-1e-4)
    theta = torch.acos(cos_theta) - np.pi/2

    stokes_vectors = mueller_rotate(theta) @ stokes_vectors[..., None]
    return stokes_vectors.squeeze()

def stokes_to_intensity(
        stokes_vectors: TensorType["batch", 3]
) -> Tuple[TensorType["batch", 1], TensorType["batch", 1]]:
    """
    Convert stokes vectors to intensity.
    Args:
        stokes_vectors: Stokes vector of the light.
    Returns:
        Intensity of the polarized light and polarization coefficients.
    """
    polarized_channels = 0.5 * torch.tensor([[1., 1., 0.],
                                             [1., 0., 1.],
                                             [1., -1., 0.],
                                             [1., 0., -1.]], dtype=stokes_vectors.dtype, device=stokes_vectors.device)
    polarized_channels = (polarized_channels[None, ...] @ stokes_vectors[..., None]).squeeze()
    total_intensity = 0.5 * torch.sum(polarized_channels, dim=-1, keepdim=True)
    polarization_coefficients = polarized_channels / (total_intensity + 1e-10)
    return polarized_channels, polarization_coefficients

def to_dop(data: TensorType[..., 4] = None, stokes: TensorType[..., 3] = None):
    """Computes Degree of Linear Polarization given polarization data or stokes vector"""
    assert data is not None or stokes is not None, "Either data or stokes must be provided"
    if data is not None:
        shape = data.shape
        stokes = torch.tensor([[0.5, 0.5, 0.5, 0.5],
                               [1., 0., -1., 0.],
                               [0., 1., 0., -1.]], dtype=data.dtype, device=data.device)
        stokes = (stokes[None, ...] @ data.view(-1, 4, 1)).squeeze()
    else:
        shape = stokes.shape
    dop = torch.norm(stokes[..., 1:], dim=-1, keepdim=True) / stokes[..., :1]
    dop = dop.view(shape[:-1])
    return dop

def to_aop(data: TensorType[..., 4] = None, stokes: TensorType[..., 3] = None):
    """Computes Angle of Polarization given polarization data or stokes vector"""
    assert data is not None or stokes is not None, "Either data or stokes must be provided"
    if data is not None:
        shape = data.shape
        stokes = torch.tensor([[0.5, 0.5, 0.5, 0.5],
                               [1., 0., -1., 0.],
                               [0., 1., 0., -1.]], dtype=data.dtype, device=data.device)
        stokes = (stokes[None, ...] @ data.view(-1, 4, 1)).squeeze()
    else:
        shape = stokes.shape
    aop = 0.5 * torch.atan2(stokes[..., 2], stokes[..., 1] + 1e-7)
    mask = aop < 0
    aop[mask] = aop[mask] + np.pi
    aop = torch.clamp(aop, 0, np.pi)
    aop = aop.view(shape[:-1])
    return aop
