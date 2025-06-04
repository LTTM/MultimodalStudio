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
Surface model module.
"""

from dataclasses import dataclass, field
from typing import Type, Union, Dict, List

import numpy as np
import torch
from torch.nn import functional as F, Parameter

from cameras.rays import RaySamples
from configs.configs import InstantiateConfig
from engine.callbacks import TrainingCallbackAttributes, TrainingCallback, TrainingCallbackLocation
from field_components.spatial_distortions import SpatialDistortionConfig
from fields.surface_field import SurfaceFieldConfig
from model_components.volume_rendering import NeuSVolumeRenderingConfig, VolumeRenderingConfig
from utils import profiler

@dataclass
class SurfaceModelConfig(InstantiateConfig):
    """Surface model config."""

    _target: Type = field(default_factory=lambda: SurfaceModel)
    surface_field: SurfaceFieldConfig = field(default_factory=lambda: SurfaceFieldConfig)
    """Surface field config. This is the base field that estimates the signed distance function of the scene."""
    volume_rendering: VolumeRenderingConfig = field(default_factory=lambda: NeuSVolumeRenderingConfig)
    """Volume rendering config. This is the module that estimates the density given the sdf values."""
    spatial_distortion: Union[None, SpatialDistortionConfig] = None
    """Spatial distortion module to use"""
    use_numerical_gradients: bool = False
    """whether to use numerical gradients computation"""
    numerical_gradient_taps: int = 4
    """number of taps for numerical gradients (whether to consider either 4 or 6 additional points)"""
    compute_hessian: bool = False
    """whether to compute hessian of the SDF"""

class SurfaceModel(torch.nn.Module):
    """Surface model in charge of estimating the geometry of the scene."""

    config: SurfaceModelConfig

    def __init__(
            self,
            config: SurfaceModelConfig,
    ):
        super().__init__()
        self.config = config
        self.surface_field = self.config.surface_field.setup()
        self.volume_rendering = self.config.volume_rendering.setup()
        self.spatial_distortion = self.config.spatial_distortion.setup() \
            if self.config.spatial_distortion is not None \
            else None

    @profiler.time_function
    def forward(self, ray_samples: RaySamples, return_weights=True, return_occupancy=False):
        """
        Forward pass of the surface model.
        Computes the SDF, the SDF gradient and hessian, the surface normals, and the geometric feature.
        """
        inputs = ray_samples.frustums.get_start_positions()
        inputs = inputs.view(-1, 3)

        if self.spatial_distortion is not None:
            inputs = self.spatial_distortion(inputs)
        inputs.requires_grad_(True)

        with torch.enable_grad():
            sdf, geo_feature =  self.surface_field(inputs)

        gradients, hessians, sampled_sdf = self.gradient(
            inputs,
            sdf,
            skip_spatial_distortion=True,
            return_sdf=True
        )

        if sampled_sdf is not None:
            sampled_sdf = sampled_sdf.view(
                -1,
                *ray_samples.frustums.directions.shape[:-1]
            ).permute(1, 2, 0).contiguous()

        sdf = sdf.view(*ray_samples.frustums.directions.shape[:-1], -1)
        gradients = gradients.view(*ray_samples.frustums.directions.shape[:-1], -1)
        hessians = hessians.view(*ray_samples.frustums.directions.shape[:-1], -1) if hessians is not None else None
        normals = F.normalize(gradients, p=2, dim=-1)

        outputs = {}
        outputs.update(
            {
                "sdf": sdf,
                "normals": normals,
                "gradients": gradients,
                "geo_feature": geo_feature,
                "hessians": hessians,
                "inputs": inputs,
                "sampled_sdf": sampled_sdf,

            }
        )

        if hasattr(self.volume_rendering.density_fn, "variance_network"):
            outputs.update({"inv_s": 1.0 / self.volume_rendering.density_fn.variance_network.get_inv_variance()})
        elif hasattr(self.volume_rendering.density_fn, "beta"):
            outputs.update({"beta": self.volume_rendering.density_fn.get_beta()})

        if return_weights:
            weights = self.volume_rendering(ray_samples, sdf, gradients=gradients)
            outputs.update({'weights': weights})

        if return_occupancy:
            occupancy = self.get_occupancy(sdf)
            outputs.update({"occupancy": occupancy})

        return outputs

    def gradient(self, x, y=None, skip_spatial_distortion=False, return_sdf=False):
        """compute the gradient of the SDF
        https://github.com/NVlabs/neuralangelo/blob/main/projects/neuralangelo/utils/modules.py"""
        if self.spatial_distortion is not None and not skip_spatial_distortion:
            x = self.spatial_distortion(x)

        # compute gradient in contracted space
        if self.config.use_numerical_gradients:
            if self.config.numerical_gradient_taps == 4:
                delta = self.numerical_gradients_delta / np.sqrt(3)
                k1 = torch.tensor([1, -1, -1], dtype=x.dtype, device=x.device)  # [3]
                k2 = torch.tensor([-1, -1, 1], dtype=x.dtype, device=x.device)  # [3]
                k3 = torch.tensor([-1, 1, -1], dtype=x.dtype, device=x.device)  # [3]
                k4 = torch.tensor([1, 1, 1], dtype=x.dtype, device=x.device)  # [3]
                sdf1 = self.surface_field.single_output(x + k1 * delta)  # [...,1]
                sdf2 = self.surface_field.single_output(x + k2 * delta)  # [...,1]
                sdf3 = self.surface_field.single_output(x + k3 * delta)  # [...,1]
                sdf4 = self.surface_field.single_output(x + k4 * delta)  # [...,1]
                gradients = (k1 * sdf1 + k2 * sdf2 + k3 * sdf3 + k4 * sdf4) / (4.0 * delta)
                points_sdf = torch.stack([sdf1, sdf2, sdf3, sdf4], dim=0)
                if self.training and self.config.compute_hessian:
                    hessian_xx = ((sdf1 + sdf2 + sdf3 + sdf4) / 2.0 - 2 * y) / delta ** 2  # [N,1]
                    hessians = torch.cat([hessian_xx, hessian_xx, hessian_xx], dim=-1) / 3.0
                else:
                    hessians = None
            elif self.config.numerical_gradient_taps == 6:
                # https://github.com/bennyguo/instant-nsr-pl/blob/main/models/geometry.py#L173
                delta = self.numerical_gradients_delta
                points = torch.stack(
                    [
                        x + torch.tensor([delta, 0.0, 0.0], dtype=x.dtype, device=x.device),
                        x + torch.tensor([-delta, 0.0, 0.0], dtype=x.dtype, device=x.device),
                        x + torch.tensor([0.0, delta, 0.0], dtype=x.dtype, device=x.device),
                        x + torch.tensor([0.0, -delta, 0.0], dtype=x.dtype, device=x.device),
                        x + torch.tensor([0.0, 0.0, delta], dtype=x.dtype, device=x.device),
                        x + torch.tensor([0.0, 0.0, -delta], dtype=x.dtype, device=x.device),
                    ],
                    dim=0,
                )

                points_sdf = self.surface_field.single_output(points.view(-1, 3)).view(6, *x.shape[:-1])
                gradients = torch.stack(
                    [
                        0.5 * (points_sdf[0] - points_sdf[1]) / delta,
                        0.5 * (points_sdf[2] - points_sdf[3]) / delta,
                        0.5 * (points_sdf[4] - points_sdf[5]) / delta,
                    ],
                    dim=-1,
                )
                if self.training and self.config.compute_hessian:
                    y = y.squeeze()
                    hessians = torch.stack(
                        [
                            (points_sdf[0] + points_sdf[1] - 2 * y) / (delta ** 2),
                            (points_sdf[2] + points_sdf[3] - 2 * y) / (delta ** 2),
                            (points_sdf[4] + points_sdf[5] - 2 * y) / (delta ** 2),
                        ],
                        dim=-1,
                    )
                else:
                    hessians = None
            else:
                raise ValueError("Invalid number of taps for numerical gradients. Must be 4 or 6.")
        else:
            x.requires_grad_(True)

            d_output = torch.ones_like(y, requires_grad=False, device=y.device)
            gradients = torch.autograd.grad(
                outputs=y, inputs=x, grad_outputs=d_output, create_graph=True, retain_graph=True, only_inputs=True
            )[0]
            points_sdf = None
            if self.training and self.config.compute_hessian:
                hessians = torch.autograd.grad(gradients.sum(), x, create_graph=True)[0]
            else:
                hessians = None
        if not return_sdf:
            return gradients, hessians
        return gradients, hessians, points_sdf

    def get_occupancy(self, sdf):
        """compute occupancy as in UniSurf"""
        occupancy = self.sigmoid(-10.0 * sdf)
        return occupancy

    def get_sdf(self, ray_samples: RaySamples):
        """Returns the signed distance function of the samples."""
        inputs = ray_samples.frustums.get_start_positions()
        inputs = inputs.view(-1, 3)

        if self.spatial_distortion is not None:
            inputs = self.spatial_distortion(inputs)
        inputs.requires_grad_(True)

        with torch.enable_grad():
            sdf, _ = self.surface_field(inputs)

        sdf = sdf.view(*ray_samples.frustums.directions.shape[:-1], -1)
        return sdf

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Returns the parameter groups for the optimizer."""
        param_groups = {
            "surface_field": list(self.surface_field.parameters()),
        }
        param_groups.update(self.volume_rendering.get_param_groups())
        return param_groups

    def set_numerical_gradients_delta(self, delta: float) -> None:
        """Set the delta value for numerical gradient."""
        self.numerical_gradients_delta = delta

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        """Returns the training callbacks from both the Dataloader and the Model."""
        volume_rendering_callbacks = self.volume_rendering.get_training_callbacks(training_callback_attributes)
        surface_field_callbacks = self.surface_field.get_training_callbacks(training_callback_attributes)
        callbacks = volume_rendering_callbacks + surface_field_callbacks

        if self.config.use_numerical_gradients:

            min_res = training_callback_attributes.model.surface_model.surface_field.field.feature_grid.encoding.min_res
            max_res = training_callback_attributes.model.surface_model.surface_field.field.feature_grid.encoding.max_res
            num_levels = training_callback_attributes.model.surface_model.surface_field.field.feature_grid.encoding.num_levels
            radius = training_callback_attributes.model.surface_model.surface_field.field.feature_grid.radius
            steps_per_level = int(
                training_callback_attributes.trainer.max_num_iterations * \
                training_callback_attributes.model.surface_model.surface_field.field.feature_grid.steps_per_level_ratio
            )
            steps_per_level = min(
                steps_per_level,
                int(
                    training_callback_attributes.trainer.max_num_iterations / \
                    training_callback_attributes.model.surface_model.surface_field.field.feature_grid.encoding.num_levels
                )
            )
            growth_factor = np.exp((np.log(max_res) - np.log(min_res)) / (num_levels - 1))
            def set_delta(step):
                delta = 1.0 / (min_res * growth_factor ** int(step / steps_per_level))
                delta = max(1.0 / max_res, delta)
                self.set_numerical_gradients_delta(
                    delta * (radius * 2.0)
                )

            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=set_delta,
                )
            )

        return callbacks

    def get_model_parameters(self):
        """Returns the model parameters."""
        return self.surface_field.get_model_parameters()
