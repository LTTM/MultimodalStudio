# Copyright 2025 Sony Group Corporation.
# All rights reserved.
#
# Licenced under the License reported at
#
#     https://github.com/LTTM/MultimodalStudio/LICENSE.txt (the "License").
#
# This code is a modified version of the original code available at
#
#     https://github.com/nerfstudio-project/nerfstudio
#
# Copyright 2022 The Nerfstudio Team. All rights reserved.
# At the moment of this file creation, the original code is licensed under the Apache License,
# Version 2.0; You may obtain a copy of the Apache License, Version 2.0, at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# The discrete_cosine_transform_dict function is a modified version of the original code available at
#
#     https://github.com/autonomousvision/factor-fields (commit 21ea155d70efce5f96399830cb424c444c977948)
#
# At the moment of this file creation, the original code is licensed under the MIT License,
# Copyright (c) 2023 autonomousvision; a copy of the MIT License, and the list of the files it
# applies to, is reported at
#
#     https://github.com/LTTM/MultimodalStudio/LICENSE_FACTOR_FIELDS.txt
#
# See the License for the specific language governing permissions and limitations under the License.
#
# Author: Federico Lincetto, Ph.D. Student at the University of Padova

""" Math Helper Functions """
from dataclasses import dataclass
from torchtyping import TensorType

import math
import numpy as np

import torch

def components_from_spherical_harmonics(levels: int, directions: TensorType[..., 3]) -> TensorType[..., "components"]:
    """
    Returns value for each component of spherical harmonics.

    Args:
        levels: Number of spherical harmonic levels to compute.
        directions: Spherical hamonic coefficients
    """
    num_components = (levels + 1 ) ** 2
    components = torch.zeros((*directions.shape[:-1], num_components), device=directions.device)

    assert 0 <= levels <= 4, f"SH levels must be in [1,4], got {levels}"
    assert directions.shape[-1] == 3, f"Direction input should have three dimensions. Got {directions.shape[-1]}"

    x = directions[..., 0]
    y = directions[..., 1]
    z = directions[..., 2]

    xx = x**2
    yy = y**2
    zz = z**2

    # l0
    components[..., 0] = 0.28209479177387814

    # l1
    if levels > 0:
        components[..., 1] = -0.4886025119029199 * y
        components[..., 2] = 0.4886025119029199 * z
        components[..., 3] = -0.4886025119029199 * x

    # l2
    if levels > 1:
        components[..., 4] = 1.0925484305920792 * x * y
        components[..., 5] = -1.0925484305920792 * y * z
        components[..., 6] = 0.31539156525252005 * (2.0 * zz - xx - yy)
        components[..., 7] = -1.0925484305920792 * x * z
        components[..., 8] = 0.5462742152960396 * (xx - yy)

    # l3
    if levels > 2:
        components[..., 9] = -0.5900435899266435 * y * (3 * xx - yy)
        components[..., 10] = 2.890611442640554 * x * y * z
        components[..., 11] = -0.4570457994644658 * y * (4 * zz - xx - yy)
        components[..., 12] = 0.3731763325901154 * z * (2 * zz - 3 * (xx + yy))
        components[..., 13] = -0.4570457994644658 * x * (4 * zz - xx - yy)
        components[..., 14] = 1.445305721320277 * z * (xx - yy)
        components[..., 15] = -0.5900435899266435 * x * (xx - 3 * yy)

    # l4
    if levels > 3:
        components[..., 16] = 2.5033429417967046 * x * y * (xx - yy)
        components[..., 17] = -1.7701307697799304 * y * z * (3 * xx - yy)
        components[..., 18] = 0.9461746957575601 * x * y * (7 * zz - 1)
        components[..., 19] = -0.6690465435572892 * y * z * (7 * zz - 3)
        components[..., 20] = 0.10578554691520431 * (zz * (35 * zz - 30) + 3)
        components[..., 21] = -0.6690465435572892 * x * z * (7 * zz - 3)
        components[..., 22] = 0.47308734787878004 * (xx - yy) * (7 * zz - 1)
        components[..., 23] = -1.7701307697799304 * x * z * (xx - 3 * yy)
        components[..., 24] = 0.4425326924449826 * (xx * (xx - 3 * yy) - yy * (3 * xx - yy))

    return components


@dataclass
class Gaussians:
    """Stores Gaussians

    Args:
        mean: Mean of multivariate Gaussian
        cov: Covariance of multivariate Gaussian.
    """

    mean: TensorType[..., "dim"]
    cov: TensorType[..., "dim", "dim"]


def compute_3d_gaussian(
    directions: TensorType[..., 3],
    means: TensorType[..., 3],
    dir_variance: TensorType[..., 1],
    radius_variance: TensorType[..., 1],
) -> Gaussians:
    """Compute guassian along ray.

    Args:
        directions: Axis of Gaussian.
        means: Mean of Gaussian.
        dir_variance: Variance along direction axis.
        radius_variance: Variance tangent to direction axis.

    Returns:
        Gaussians: Oriented 3D gaussian.
    """

    dir_outer_product = directions[..., :, None] * directions[..., None, :]
    eye = torch.eye(directions.shape[-1], device=directions.device)
    dir_mag_sq = torch.clamp(torch.sum(directions**2, dim=-1, keepdim=True), min=1e-10)
    null_outer_product = eye - directions[..., :, None] * (directions / dir_mag_sq)[..., None, :]
    dir_cov_diag = dir_variance[..., None] * dir_outer_product[..., :, :]
    radius_cov_diag = radius_variance[..., None] * null_outer_product[..., :, :]
    cov = dir_cov_diag + radius_cov_diag
    return Gaussians(mean=means, cov=cov)


def cylinder_to_gaussian(
    origins: TensorType[..., 3],
    directions: TensorType[..., 3],
    starts: TensorType[..., 1],
    ends: TensorType[..., 1],
    radius: TensorType[..., 1],
) -> Gaussians:
    """Approximates cylinders with a Gaussian distributions.

    Args:
        origins: Origins of cylinders.
        directions: Direction (axis) of cylinders.
        starts: Start of cylinders.
        ends: End of cylinders.
        radius: Radii of cylinders.

    Returns:
        Gaussians: Approximation of cylinders
    """
    means = origins + directions * ((starts + ends) / 2.0)
    dir_variance = (ends - starts) ** 2 / 12
    radius_variance = radius**2 / 4.0
    return compute_3d_gaussian(directions, means, dir_variance, radius_variance)


def conical_frustum_to_gaussian(
    origins: TensorType[..., 3],
    directions: TensorType[..., 3],
    starts: TensorType[..., 1],
    ends: TensorType[..., 1],
    radius: TensorType[..., 1],
) -> Gaussians:
    """Approximates conical frustums with a Gaussian distributions.

    Uses stable parameterization described in mip-NeRF publication.

    Args:
        origins: Origins of cones.
        directions: Direction (axis) of frustums.
        starts: Start of conical frustums.
        ends: End of conical frustums.
        radius: Radii of cone a distance of 1 from the origin.

    Returns:
        Gaussians: Approximation of conical frustums
    """
    mu = (starts + ends) / 2.0
    hw = (ends - starts) / 2.0
    means = origins + directions * (mu + (2.0 * mu * hw**2.0) / (3.0 * mu**2.0 + hw**2.0))
    dir_variance = (hw**2) / 3 - (4 / 15) * ((hw**4 * (12 * mu**2 - hw**2)) / (3 * mu**2 + hw**2) ** 2)
    radius_variance = radius**2 * ((mu**2) / 4 + (5 / 12) * hw**2 - 4 / 15 * (hw**4) / (3 * mu**2 + hw**2))
    return compute_3d_gaussian(directions, means, dir_variance, radius_variance)


def expected_sin(x_means: torch.Tensor, x_vars: torch.Tensor) -> torch.Tensor:
    """Computes the expected value of sin(y) where y ~ N(x_means, x_vars)

    Args:
        x_means: Mean values.
        x_vars: Variance of values.

    Returns:
        torch.Tensor: The expected value of sin.
    """

    return torch.exp(-0.5 * x_vars) * torch.sin(x_means)


def discrete_cosine_transform_dict(n_atoms, size, max_n_basis, dimensions=1):
    """
    Create a dictionary using the Discrete Cosine Transform (DCT) basis.
    The returned dictionary will have min(n_atoms**dimensions, max_n_basis)
    atoms. The returned DCT bases are orthonormal.
    :param n_atoms:
        Number of atoms (basis) in dict for each dimension
    :param size:
        Size of first patch
    :param max_n_basis:
        Max number of returned bases
    :param dimensions:
        DCT basis dimensions
    :return:
        DCT dictionary, shape [min(n_atoms**dimensions, max_n_basis), size**dimensions]
    """
    p = n_atoms # index of the DCT basis
    dct = np.zeros((p, size))  # Shape [p, size], the p DCT basis of dimension size
    for k in range(p):
        basis = np.cos((np.arange(size) + 0.5) * k * math.pi / size)
        if k == 0:
            basis *= 1/np.sqrt(p)
        else:
            basis *= np.sqrt(2/p)
        # Not needed for the DCT basis, as they already have zero mean
        # if k > 0:
        #     basis = basis - np.mean(basis)
        dct[k] = basis # One-dimensional DTC set of bases
    dtc_basis = np.copy(dct)

    if dimensions > 1:
        dtc_basis = np.kron(dtc_basis, dct)  # Shape [p^2, size^2], two-dimensional DCT set of bases
    if dimensions > 2:
        dtc_basis = np.kron(dtc_basis, dct)  # Shape [p^3, size^3], three-dimensional DCT set of bases
    if dimensions > 3:
        raise ValueError(f"Dimensions > 3 not supported, got {dimensions}")

    if max_n_basis < dtc_basis.shape[0]:  # Select only a number equal to max_n_basis DTC basis elements
        idx = [x[0] for x in np.array_split(np.arange(dtc_basis.shape[0]), max_n_basis)]
        dtc_basis = dtc_basis[idx]  # Shape [max_n_basis, size^dimensions]

    # Normalize DTC basis
    for basis_elem in range(dtc_basis.shape[0]):
        norm = np.linalg.norm(dtc_basis[basis_elem]) or 1
        dtc_basis[basis_elem] /= norm

    # dtc_basis = torch.FloatTensor(dtc_basis)
    return dtc_basis

