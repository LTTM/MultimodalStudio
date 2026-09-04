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
Helper for Lie group operations. Currently only used for pose optimization.
"""
import torch
from torchtyping import TensorType

# We make an exception on snake case conventions because SO3 != so3.
def exp_map_SO3xR3(tangent_vector: TensorType["b", 6]) -> TensorType["b", 3, 4]:  # pylint: disable=invalid-name
    """Compute the exponential map of the direct product group `SO(3) x R^3`.

    This can be used for learning pose deltas on SE(3), and is generally faster than `exp_map_SE3`.

    Args:
        tangent_vector: Tangent vector; length-3 translations, followed by an `so(3)` tangent vector.
    Returns:
        [R|t] tranformation matrices.
    """
    # code for SO3 map grabbed from pytorch3d and stripped down to bare-bones
    log_rot = tangent_vector[:, 3:]
    nrms = (log_rot * log_rot).sum(1)
    rot_angles = torch.clamp(nrms, 1e-4).sqrt()
    rot_angles_inv = 1.0 / rot_angles
    fac1 = rot_angles_inv * rot_angles.sin()
    fac2 = rot_angles_inv * rot_angles_inv * (1.0 - rot_angles.cos())
    skews = torch.zeros((log_rot.shape[0], 3, 3), device=tangent_vector.device)
    skews[:, 0, 1] = -log_rot[:, 2]
    skews[:, 0, 2] = log_rot[:, 1]
    skews[:, 1, 0] = log_rot[:, 2]
    skews[:, 1, 2] = -log_rot[:, 0]
    skews[:, 2, 0] = -log_rot[:, 1]
    skews[:, 2, 1] = log_rot[:, 0]
    skews_square = torch.bmm(skews, skews)

    ret = torch.zeros((tangent_vector.shape[0], 3, 4), device=tangent_vector.device)
    ret[:, :3, :3] = (
        fac1[:, None, None] * skews
        + fac2[:, None, None] * skews_square
        + torch.eye(3, device=tangent_vector.device)[None]
    )

    # Compute the translation
    ret[:, :3, 3] = tangent_vector[:, :3]
    return ret


def exp_map_SE3(tangent_vector: TensorType["b", 6]) -> TensorType["b", 3, 4]:  # pylint: disable=invalid-name
    """Compute the exponential map `se(3) -> SE(3)`.

    This can be used for learning pose deltas on `SE(3)`.

    Args:
        tangent_vector: A tangent vector from `se(3)`.

    Returns:
        [R|t] tranformation matrices.
    """

    tangent_vector_lin = tangent_vector[:, :3].view(-1, 3, 1)
    tangent_vector_ang = tangent_vector[:, 3:].view(-1, 3, 1)

    theta = torch.linalg.norm(tangent_vector_ang, dim=1).unsqueeze(1)
    theta2 = theta**2
    theta3 = theta**3

    near_zero = theta < 1e-2
    non_zero = torch.ones(1, dtype=tangent_vector.dtype, device=tangent_vector.device)
    theta_nz = torch.where(near_zero, non_zero, theta)
    theta2_nz = torch.where(near_zero, non_zero, theta2)
    theta3_nz = torch.where(near_zero, non_zero, theta3)

    # Compute the rotation
    sine = theta.sin()
    cosine = torch.where(near_zero, 8 / (4 + theta2) - 1, theta.cos())
    sine_by_theta = torch.where(near_zero, 0.5 * cosine + 0.5, sine / theta_nz)
    one_minus_cosine_by_theta2 = torch.where(near_zero, 0.5 * sine_by_theta, (1 - cosine) / theta2_nz)
    ret = torch.zeros(tangent_vector.shape[0], 3, 4).to(dtype=tangent_vector.dtype, device=tangent_vector.device)
    ret[:, :3, :3] = one_minus_cosine_by_theta2 * tangent_vector_ang @ tangent_vector_ang.transpose(1, 2)

    ret[:, 0, 0] += cosine.view(-1)
    ret[:, 1, 1] += cosine.view(-1)
    ret[:, 2, 2] += cosine.view(-1)
    temp = sine_by_theta.view(-1, 1) * tangent_vector_ang.view(-1, 3)
    ret[:, 0, 1] -= temp[:, 2]
    ret[:, 1, 0] += temp[:, 2]
    ret[:, 0, 2] += temp[:, 1]
    ret[:, 2, 0] -= temp[:, 1]
    ret[:, 1, 2] -= temp[:, 0]
    ret[:, 2, 1] += temp[:, 0]

    # Compute the translation
    sine_by_theta = torch.where(near_zero, 1 - theta2 / 6, sine_by_theta)
    one_minus_cosine_by_theta2 = torch.where(near_zero, 0.5 - theta2 / 24, one_minus_cosine_by_theta2)
    theta_minus_sine_by_theta3_t = torch.where(near_zero, 1.0 / 6 - theta2 / 120, (theta - sine) / theta3_nz)

    ret[:, :, 3:] = sine_by_theta * tangent_vector_lin
    ret[:, :, 3:] += one_minus_cosine_by_theta2 * torch.cross(tangent_vector_ang, tangent_vector_lin, dim=1)
    ret[:, :, 3:] += theta_minus_sine_by_theta3_t * (
        tangent_vector_ang @ (tangent_vector_ang.transpose(1, 2) @ tangent_vector_lin)
    )
    return ret

def inverse_exp_map_SO3xR3(transform: TensorType["b", 3, 4]) -> TensorType["b", 6]:
    """Inverse of `exp_map_SO3xR3`.

    Given a batch of transformation matrices [R|t] (shape (b,3,4)), return the
    corresponding tangent vectors in so(3)+R^3 ordering used by
    `exp_map_SO3xR3`: first 3 entries are translation, last 3 are the so(3)
    tangent vector (axis-angle vector).

    The implementation is numerically stable for small angles.
    """

    if transform.ndim != 3 or transform.shape[1:] != (3, 4):
        raise ValueError("transform must have shape (b, 3, 4)")

    device = transform.device
    dtype = transform.dtype

    R = transform[:, :3, :3]
    t = transform[:, :3, 3]

    # trace and angle
    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    cos_theta = (trace - 1.0) / 2.0
    # clamp to valid domain to avoid NaNs from acos
    cos_theta = torch.clamp(cos_theta, -1.0 + 1e-7, 1.0 - 1e-7)
    theta = torch.acos(cos_theta)

    # vee of the skew-symmetric part: vee_skew = 0.5 * [R32-R23, R13-R31, R21-R12]
    r_minus_rt = R - R.transpose(1, 2)
    vee_skew = 0.5 * torch.stack(
        [r_minus_rt[:, 2, 1], r_minus_rt[:, 0, 2], r_minus_rt[:, 1, 0]],
        dim=1,
    )

    sin_theta = torch.sin(theta)

    # Avoid division by zero: when theta is very small, theta/sin(theta) -> 1
    near_zero = theta.abs() < 1e-4
    sin_theta_safe = torch.where(near_zero, torch.ones_like(sin_theta), sin_theta)

    factor = theta / sin_theta_safe

    # For small angles, use first-order approximation: so3 ~= vee_skew
    so3_vec = factor.unsqueeze(1) * vee_skew
    so3_vec = torch.where(near_zero.unsqueeze(1), vee_skew, so3_vec)

    tangent = torch.cat([t, so3_vec], dim=1).to(dtype=dtype, device=device)
    return tangent

def inverse_exp_map_SE3(transform: TensorType["b", 3, 4]) -> TensorType["b", 6]:
    """Inverse of `exp_map_SE3`.

    Given a batch of transformation matrices [R|t] (shape (b,3,4)), return the
    corresponding tangent vectors in se(3) ordering used by `exp_map_SE3`: the
    first 3 entries are the translational tangent v and the last 3 are the
    rotational tangent (axis-angle) w.

    Implementation notes:
    - Rotation extraction reuses the standard axis-angle extraction from a
      rotation matrix (robust to small angles via clamping).
    - The translation part is recovered by constructing the same "V" matrix
      used in the forward `exp_map_SE3` and solving V v = t using a batched
      linear solver. This is numerically stable and avoids deriving a closed
      form inverse.
    """

    if transform.ndim != 3 or transform.shape[1:] != (3, 4):
        raise ValueError("transform must have shape (b, 3, 4)")

    device = transform.device
    dtype = transform.dtype

    R = transform[:, :3, :3]
    t = transform[:, :3, 3].unsqueeze(2)  # shape (b, 3, 1)

    # --- Extract rotation axis-angle (so(3) vector) ---
    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    cos_theta = (trace - 1.0) / 2.0
    cos_theta = torch.clamp(cos_theta, -1.0 + 1e-7, 1.0 - 1e-7)
    theta = torch.acos(cos_theta)

    r_minus_rt = R - R.transpose(1, 2)
    vee_skew = 0.5 * torch.stack(
        [r_minus_rt[:, 2, 1], r_minus_rt[:, 0, 2], r_minus_rt[:, 1, 0]],
        dim=1,
    )

    sin_theta = torch.sin(theta)
    near_zero = theta.abs() < 1e-4
    sin_theta_safe = torch.where(near_zero, torch.ones_like(sin_theta), sin_theta)
    factor = theta / sin_theta_safe
    so3_vec = factor.unsqueeze(1) * vee_skew
    so3_vec = torch.where(near_zero.unsqueeze(1), vee_skew, so3_vec)

    # Use the axis-angle vector to build the same V matrix used in exp_map_SE3
    omega = so3_vec.view(-1, 3)  # (b,3)
    theta_from_omega = torch.linalg.norm(omega, dim=1, keepdim=True)  # (b,1)
    theta2 = theta_from_omega**2
    theta3 = theta_from_omega**3

    # safe non-zero substitutions
    non_zero = torch.ones(1, dtype=dtype, device=device)
    th_nz = torch.where(theta_from_omega < 1e-6, non_zero.to(device), theta_from_omega)
    th2_nz = th_nz**2
    th3_nz = th_nz**3

    sine = torch.sin(th_nz)
    cosine = torch.cos(th_nz)

    a = sine / th_nz  # sin(theta)/theta
    b = (1.0 - cosine) / th2_nz  # (1-cos)/theta^2
    c = (th_nz - sine) / th3_nz  # (theta - sin)/theta^3

    # small-angle series fallback consistent with exp_map_SE3
    a = torch.where(theta_from_omega < 1e-6, 1.0 - theta2 / 6.0, a)
    b = torch.where(theta_from_omega < 1e-6, 0.5 - theta2 / 24.0, b)
    c = torch.where(theta_from_omega < 1e-6, 1.0 / 6.0 - theta2 / 120.0, c)

    b = b.view(-1, 1, 1)
    a = a.view(-1, 1, 1)
    c = c.view(-1, 1, 1)

    # Build skew matrices W and outer products omega omega^T
    W = torch.zeros((omega.shape[0], 3, 3), dtype=dtype, device=device)
    W[:, 0, 1] = -omega[:, 2]
    W[:, 0, 2] = omega[:, 1]
    W[:, 1, 0] = omega[:, 2]
    W[:, 1, 2] = -omega[:, 0]
    W[:, 2, 0] = -omega[:, 1]
    W[:, 2, 1] = omega[:, 0]

    outer = omega.view(-1, 3, 1).matmul(omega.view(-1, 1, 3))

    I3 = torch.eye(3, device=device, dtype=dtype)[None]

    # V = a * I + b * W + c * (omega omega^T)
    V = a * I3 + b * W + c * outer

    # Solve for v: V v = t
    # Use batched solver; for tiny theta V is close to I so solve is stable
    v = torch.linalg.solve(V, t).view(-1, 3)

    tangent = torch.cat([v, so3_vec], dim=1).to(dtype=dtype, device=device)
    return tangent
