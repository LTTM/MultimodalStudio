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
from typing import Type, Literal, Dict, Mapping, Any

import torch
from torchtyping import TensorType

from cameras import lie_groups
from cameras.lie_groups import exp_map_SE3, exp_map_SO3xR3
from configs.configs import InstantiateConfig
from data.datasets import BaseDataset
from engine.schedulers import SchedulerConfig
import utils.poses as pose_utils
from utils.misc import get_dict_to_torch


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
    rig: bool = False
    """Whether to optimize cameras as a rig. The first camera in the rig is the reference. 
    Overrides shared_optimization flag: if this is set to True, the reference modality poses are optimized in non-shared 
    optimization mode while the other modality poses are optimized in shared optimization mode."""
    relative_pose_opt: bool = False
    """Whether to optimize modality poses relative to a reference modality."""
    reference_modality: str = None
    """Reference modality for relative pose optimization. If empty, the first modality in modalities_to_optimize is used."""
    gt_scale_state_dict: bool = False
    """Whether to load/save the optimized camera poses in GT scale."""

class CameraOptimizer(torch.nn.Module):
    """Module that optimizes the camera poses. It accepts modes "off", "SO3xR3", "SE3"."""

    config: CameraOptimizerConfig

    def __init__(
            self,
            config: CameraOptimizerConfig,
            num_cameras: int,
            dataset: BaseDataset = None,  # pylint: disable=unused-argument
            **kwargs,  # pylint: disable=unused-argument
    ) -> None:
        super().__init__()
        self.config = config
        self.num_cameras = num_cameras
        self.dataset = dataset

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

        if self.config.rig:
            assert self.config.shared_optimization, "Rig optimization requires shared_optimization to be True."
            reference_mod = next(iter(self.config.modalities_to_optimize.keys()))
            self.rig2camera = self.get_reference2camera_mats(reference_mod)
            self.rig_pose_opt = torch.nn.Parameter(torch.zeros((self.num_cameras, 6)))

        if self.config.relative_pose_opt:
            if self.config.reference_modality is None:
                self.reference_mod = next(iter(self.config.modalities_to_optimize.keys()))
            else:
                self.reference_mod = self.config.reference_modality
            self.reference2camera = self.get_reference2camera_mats(self.reference_mod)

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
                mat = self.get_mod_transformation_matrix(mod, indices[:,0])

                if self.config.relative_pose_opt:
                    if mod != self.reference_mod:
                        reference_mat = self.get_mod_transformation_matrix(self.reference_mod, indices[:,0]).detach()
                        reference2cam = self.reference2camera[mod].to(mat.device)
                        cam2reference = pose_utils.inverse(reference2cam)
                        reference_opt_transform = pose_utils.multiply(reference_mat, cam2reference)
                        reference_opt_transform = pose_utils.multiply(reference2cam, reference_opt_transform)
                        mat = pose_utils.multiply(reference_opt_transform, mat)

                if self.config.rig:
                    if self.num_cameras != self.rig_pose_opt.shape[0]:
                        rig_pose_opt = self.mean_rig_optimization()
                        rig_pose_opt = self.exp_map(rig_pose_opt[indices[:,0]])
                    else:
                        rig_pose_opt = self.exp_map(self.rig_pose_opt[indices[:,0]])
                    rig2cam = self.rig2camera[mod].to(mat.device)
                    cam2rig = pose_utils.inverse(rig2cam)
                    rig_pose_opt_transform = pose_utils.multiply(rig_pose_opt, cam2rig)
                    rig_pose_opt_transform = pose_utils.multiply(rig2cam, rig_pose_opt_transform)
                    mat = pose_utils.multiply(rig_pose_opt_transform, mat)

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

    def get_mod_transformation_matrix(self, modality: str, indices) -> TensorType["num_indices", 3, 4]:
        """Get the transformation matrix for a given modality."""
        if self.config.shared_optimization:
            parameters = self.pose_adjustment[modality].expand((self.num_cameras, 6))[indices]
        else:
            parameters = self.pose_adjustment[modality][indices]
        mat = self.exp_map(parameters)
        return mat

    def set_num_cameras(self, num_cameras: int):
        """Set the number of cameras."""
        self.num_cameras = num_cameras

    def get_reference2camera_mats(self, reference_mod: str) -> Dict[str, TensorType["num_cameras", 3, 4]]:
        """Get transformation matrices from reference camera to each camera in the rig.
        Args:
            reference_mod: reference modality.
        Returns:
            Transformation matrices from reference camera to each camera in the rig.
        """
        reference_c2w = self.dataset.data[reference_mod]["cameras"].get_c2w_matrices(0)
        reference2camera = {}
        for mod in self.config.modalities_to_optimize.keys():
            c2w = self.dataset.data[mod]["cameras"].get_c2w_matrices(0)
            reference2camera[mod] = pose_utils.multiply(pose_utils.inverse(c2w), reference_c2w)
        return reference2camera

    def mean_rig_optimization(self):
        """Return the mean optimized rig pose."""
        if not self.config.rig:
            raise ValueError("Rig optimization is not enabled.")
        mean_adjustment = torch.mean(self.rig_pose_opt, dim=0, keepdim=True).repeat(self.num_cameras, 1)
        return mean_adjustment

    def switch_reference_modality(self, camera_adjustments, current_reference_mod, new_reference_mod):
        """Switch the reference modality for relative pose optimization.
        Args:
            camera_adjustments: dictionary of camera adjustments for each modality in GT coordinates.
            current_reference_mod: current reference modality.
            new_reference_mod: new reference modality.
        Returns:
            Updated dictionary of camera adjustments for each modality.
        """
        assert current_reference_mod != new_reference_mod
        if self.config.mode == "SO3xR3":
            inv_exp_map = lie_groups.inverse_exp_map_SO3xR3
        elif self.config.mode == "SE3":
            inv_exp_map = lie_groups.inverse_exp_map_SE3
        else:
            raise ValueError("switch_reference_modality requires SO3xR3 or SE3 mode")
        updated_adjustments = {}
        device = self.reference2camera[current_reference_mod].device
        camera_adjustments = get_dict_to_torch(camera_adjustments, device=device)

        current_ref2cam_mats = self.get_reference2camera_mats(current_reference_mod)
        curr_ref2new_ref = current_ref2cam_mats[new_reference_mod]
        new_ref2curr_ref = pose_utils.inverse(curr_ref2new_ref)
        new_ref2cam_mats = self.get_reference2camera_mats(new_reference_mod)
        current_ref_mat = self.exp_map(camera_adjustments[current_reference_mod])
        new_ref_mat = pose_utils.multiply(current_ref_mat, new_ref2curr_ref)
        new_ref_mat = pose_utils.multiply(curr_ref2new_ref, new_ref_mat)
        new_ref_mat = pose_utils.multiply(new_ref_mat, self.exp_map(camera_adjustments[new_reference_mod]))
        updated_adjustments[new_reference_mod] = inv_exp_map(new_ref_mat)

        for mod in camera_adjustments.keys():
            if mod == new_reference_mod:
                continue
            curr_mod_mat = self.exp_map(camera_adjustments[mod])
            new_ref_mat_inv = pose_utils.inverse(new_ref_mat)
            curr_ref2cam = current_ref2cam_mats[mod]
            cam2curr_ref = pose_utils.inverse(curr_ref2cam)
            new_ref2cam = new_ref2cam_mats[mod]
            cam2new_ref = pose_utils.inverse(new_ref2cam)
            # For the modality that was the current reference, the previous final transform is
            # simply the optimized reference transform `current_ref_mat`. For other modalities
            # it is reference2cam * current_ref_mat * cam2reference * curr_mod_mat.
            if mod == current_reference_mod:
                curr_transform = current_ref_mat
            else:
                curr_transform = pose_utils.multiply(current_ref_mat, cam2curr_ref)
                curr_transform = pose_utils.multiply(curr_ref2cam, curr_transform)
                curr_transform = pose_utils.multiply(curr_transform, curr_mod_mat)
            new_mod_mat = pose_utils.multiply(new_ref_mat_inv, cam2new_ref)
            new_mod_mat = pose_utils.multiply(new_ref2cam, new_mod_mat)
            new_mod_mat = pose_utils.multiply(new_mod_mat, curr_transform)
            updated_adjustments[mod] = inv_exp_map(new_mod_mat)

        return updated_adjustments

    def state_dict(self, *args, destination=None, prefix="", keep_vars=False):
        """Returns the state dictionary of the module, excluding rig parameters if rig optimization is enabled."""
        state_dict = super().state_dict(*args, destination=destination, prefix=prefix, keep_vars=keep_vars)
        if self.config.mode != "off" and self.config.gt_scale_state_dict:
            state_dict_gt = {}
            for mod in self.config.modalities_to_optimize.keys():
                adjustment = state_dict[f'{prefix}pose_adjustment.{mod}']
                if self.config.shared_optimization:
                    c2w = self.dataset.data[mod]['cameras'].get_c2w_matrices(0)
                    w2gt = self.dataset.w2gt[:3]
                else:
                    c2w = self.dataset.data[mod]['cameras'].camera_to_worlds
                    w2gt = self.dataset.w2gt[:3].unsqueeze(0).expand(c2w.shape[0], 3, 4)
                c2gt = pose_utils.multiply(w2gt, c2w).to(adjustment.device)
                state_dict_gt[f'{prefix}pose_adjustment.{mod}'] = self.switch_transform_reference_system(adjustment, c2gt)
            state_dict = state_dict_gt
        return state_dict

    def load_state_dict(
        self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False
    ):
        """Loads the state dictionary of the module"""
        if self.config.mode != "off" and self.config.gt_scale_state_dict:
            state_dict_gt = {}
            for mod in self.config.modalities_to_optimize.keys():
                adjustment_gt = state_dict[f'pose_adjustment.{mod}']
                if self.config.shared_optimization:
                    c2w = self.dataset.data[mod]['cameras'].get_c2w_matrices(0)
                    w2gt = self.dataset.w2gt[:3]
                else:
                    c2w = self.dataset.data[mod]['cameras'].camera_to_worlds
                    w2gt = self.dataset.w2gt[:3].unsqueeze(0).expand(c2w.shape[0], 3, 4)
                c2gt = pose_utils.multiply(w2gt, c2w).to(adjustment_gt.device)
                state_dict_gt[f'pose_adjustment.{mod}'] = self.switch_transform_reference_system(adjustment_gt, b2a=c2gt)
            state_dict = state_dict_gt
        super().load_state_dict(state_dict, strict=strict, assign=assign)

    def switch_transform_reference_system(self, transform, a2b=None, b2a=None):
        """Convert linear transformation from reference system A to reference system B.
        Args:
            transform: linear transformation in reference system A.
            a2b: transformation matrix from reference system A to reference system B.
        Returns:
            Updated linear transformation in reference system B.
        """
        # TODO: can go into utils file and be static
        assert a2b is not None or b2a is not None, "Either a2b or b2a must be provided."
        if self.config.mode == "SO3xR3":
            inv_exp_map = lie_groups.inverse_exp_map_SO3xR3
        elif self.config.mode == "SE3":
            inv_exp_map = lie_groups.inverse_exp_map_SE3
        else:
            raise ValueError("switch_transform_reference_system requires SO3xR3 or SE3 mode")
        if a2b is not None:
            b2a = torch.eye(4, device=a2b.device)
            if a2b.ndim == 3:
                b2a = b2a.unsqueeze(0).expand(a2b.shape[0], 4, 4).clone()
            b2a[... ,:3 ,:4] = a2b[... ,:3, :4]
            b2a = torch.linalg.inv(b2a)[..., :3, :4]
        else:
            a2b = torch.eye(4, device=b2a.device)
            if b2a.ndim == 3:
                a2b = a2b.unsqueeze(0).expand(b2a.shape[0], 4, 4).clone()
            a2b[... ,:3 ,:4] = b2a[... ,:3, :4]
            a2b = torch.linalg.inv(a2b)[..., :3, :4]
        transform = self.exp_map(transform)
        transform = pose_utils.multiply(transform, b2a)
        transform = pose_utils.multiply(a2b, transform)
        return inv_exp_map(transform)

    def convert_adjustments_reference_system(self, camera_adjustments):
        """Convert camera adjustments from reference system A to reference system B.
        Args:
            a2b: transformation matrix from reference system A to reference system B.
            camera_adjustments: camera adjustments in reference system A.
        Returns:
            Updated camera adjustments in reference system B.
        """
        for mod in camera_adjustments.keys():
            adjustment = camera_adjustments[mod]
            if self.config.shared_optimization:
                c2w = self.dataset.data[mod]['cameras'].get_c2w_matrices(0).to(adjustment.device)
                w2gt = self.dataset.w2gt[:3].to(adjustment.device)
            else:
                c2w = self.dataset.data[mod]['cameras'].camera_to_worlds.to(adjustment.device)
                w2gt = self.dataset.w2gt[:3].unsqueeze(0).expand(c2w.shape[0], 3, 4).to(adjustment.device)
            c2gt = pose_utils.multiply(w2gt, c2w)
            adjustment = self.switch_transform_reference_system(adjustment, b2a=c2gt)
            camera_adjustments[mod] = adjustment
        return camera_adjustments
