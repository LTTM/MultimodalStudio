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
Multimodal model
"""

from dataclasses import dataclass, field
from typing import Type, List, Dict

import torch
from torch.nn import Parameter

from configs.configs import InstantiateConfig
from data.scene_box import SceneBox
from engine.callbacks import TrainingCallbackAttributes, TrainingCallback
from model_components.background_model import BackgroundModelConfig
from model_components.radiance_model import RadianceModelConfig
from model_components.ray_samplers import SamplerConfig
from model_components.renderers import RendererConfig
from model_components.scene_colliders import ColliderInstancer
from model_components.surface_model import SurfaceModelConfig
from utils import profiler


@dataclass
class BaseModelConfig(InstantiateConfig):
    """Base model Config"""

    _target: Type = field(default_factory=lambda: BaseModel)
    ray_sampler: SamplerConfig = field(default_factory=lambda: SamplerConfig)
    """Config for point sampler"""
    background_ray_sampler: SamplerConfig = field(default_factory=lambda: SamplerConfig)
    """Config for background point sampler"""
    surface_model: SurfaceModelConfig = field(default_factory=lambda: SurfaceModelConfig)
    """Config for surface model"""
    radiance_model: RadianceModelConfig = field(default_factory=lambda: RadianceModelConfig)
    """Config for radiance model"""
    background_model: BackgroundModelConfig = field(default_factory=lambda: BackgroundModelConfig)
    """Config for background model"""
    renderer: RendererConfig = field(default_factory=lambda: RendererConfig)
    """Config for renderer"""
    use_background_model: bool = True
    """Whether to instantiate a model for background estimation"""

class BaseModel(torch.nn.Module):
    """Standard multimodal model"""

    def __init__(
            self,
            config: BaseModelConfig,
            scene_box: SceneBox,
            modalities: Dict[str, int],
    ):
        super().__init__()
        self.config = config

        self.ray_sampler = self.config.ray_sampler.setup()
        self.collider = ColliderInstancer(scene_box)

        self.surface_model = self.config.surface_model.setup()
        self.radiance_model = self.config.radiance_model.setup(
            modalities=modalities,
        )

        if self.config.use_background_model:
            self.background_ray_sampler = self.config.background_ray_sampler.setup()
            self.background_model = self.config.background_model.setup(
                modalities=modalities,
            )

        self.renderer = self.config.renderer.setup()

    @profiler.time_function
    def forward(self, ray_bundles):
        """Estimates the radiance and geometry values for each ray in the batch"""
        # Sample points along rays
        colliding_rays_masks = self.collider.update_ray_bundles(ray_bundles)

        masked_ray_bundles = {
            mod: ray_bundle[colliding_rays_masks[mod]]
            if ray_bundle is not None
            else None
            for mod, ray_bundle in ray_bundles.items()
        }
        ray_sampler_output = self.ray_sampler(masked_ray_bundles, sdf_fn=self.surface_model.get_sdf)
        samples_per_modality = ray_sampler_output["ray_samples_per_modality"]
        background_samples_per_modality = {}
        if self.config.use_background_model:
            self.collider.update_ray_bundles_for_background(ray_bundles)
            background_samples_per_modality = self.background_ray_sampler(ray_bundles)

        outputs = {}
        for mod in samples_per_modality.keys():

            samples = samples_per_modality.get(mod, None)
            background_samples = background_samples_per_modality.get(mod, None)
            mask = colliding_rays_masks.get(mod, None)

            if samples is None:
                outputs[mod] = None
                continue

            background_outputs = None
            if self.config.use_background_model:
                background_outputs = self.background_model(background_samples)

            if samples.shape[0] == 0 and background_outputs is not None:
                background_outputs.update({
                    "normals": torch.zeros(background_samples.shape[0], 3, device=mask.device),
                    "depth": torch.zeros(background_samples.shape[0], 1, device=mask.device),
                    "accumulation": torch.zeros(background_samples.shape[0], 1, device=mask.device),
                })
                outputs[mod] = background_outputs
            else:
                # Get weights
                geometry_outputs = self.surface_model(samples)

                radiance_outputs = self.radiance_model(
                    ray_samples=samples,
                    normals=geometry_outputs["normals"].detach(),
                    geo_feature=geometry_outputs["geo_feature"],
                )

                renderer_input = {}
                renderer_input.update(radiance_outputs)
                renderer_input.update(
                        {
                            "normals": geometry_outputs["normals"],
                            "depth": samples,
                            "background": background_outputs
                        })

                modality_outputs = self.renderer.render(geometry_outputs['weights'], renderer_input, mask)

                if self.training:
                    modality_outputs.update({
                        "gradients": geometry_outputs["gradients"],
                        "hessians": geometry_outputs["hessians"],
                    })
                    if geometry_outputs.get("inv_s") is not None:
                        modality_outputs.update({"inv_s": geometry_outputs["inv_s"]})
                    elif geometry_outputs.get("beta") is not None:
                        modality_outputs.update({"beta": geometry_outputs["beta"]})

                modality_outputs.update({
                    key: value[mod]
                    for key, value in ray_sampler_output.items()
                    if key != "ray_samples_per_modality"
                })
                outputs[mod] = modality_outputs

        return outputs

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Returns the parameter groups of the model to be passed to the optimizer."""
        param_groups = {}
        surface_param_group = self.surface_model.get_param_groups()
        radiance_param_group = self.radiance_model.get_param_groups()
        ray_sampler_param_group = self.ray_sampler.get_param_groups()
        param_groups.update(surface_param_group)
        param_groups.update(radiance_param_group)
        param_groups.update(ray_sampler_param_group)
        if self.config.use_background_model:
            background_param_group = self.background_model.get_param_groups()
            param_groups.update(background_param_group)

        return param_groups

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        """Returns the training callbacks from both the Dataloader and the Model."""
        surface_model_callbacks = self.surface_model.get_training_callbacks(training_callback_attributes)
        radiance_model_callbacks = self.radiance_model.get_training_callbacks(training_callback_attributes)
        ray_sampler_callbacks = self.ray_sampler.get_training_callbacks(training_callback_attributes)
        callbacks = surface_model_callbacks + radiance_model_callbacks + ray_sampler_callbacks
        if self.config.use_background_model:
            background_model_callbacks = self.background_model.get_training_callbacks(training_callback_attributes)
            callbacks += background_model_callbacks
        return callbacks

    def get_model_parameters(self):
        """Return a set of model parameters useful for modules outside the model (e.g. optimizers, loss functions)."""
        # Surface and radiance model grids should have the same resolutions and number of levels
        parameters = {}
        parameters.update(self.surface_model.get_model_parameters())
        # parameters.update(self.radiance_model.get_model_parameters())
        if self.config.use_background_model:
            parameters.update(self.background_model.get_model_parameters())
        return parameters
