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

import copy
import random
from dataclasses import dataclass, field
from typing import Type, List, Dict, Optional

import torch
from torch.nn import Parameter

from data.scene_box import SceneBox
from engine.callbacks import TrainingCallback, TrainingCallbackLocation, TrainingCallbackAttributes
from field_components.mlp import MLPConfig
from model_components.latent_estimator import LatentEstimatorConfig
from models.base_model import BaseModelConfig, BaseModel
from utils import profiler


@dataclass
class LatentModelConfig(BaseModelConfig):
    """Base model Config"""

    _target: Type = field(default_factory=lambda: LatentModel)
    estimate_latents: bool = False
    """Whether to estimate a latent code from modality output."""
    latent_estimators: Optional[Dict[str, LatentEstimatorConfig]] = None
    """Modality heads to estimate the latent code from the modality output."""
    use_many2one_modality_estimator: bool = False
    """Whether to use a many-to-one modality estimator."""
    use_many2luma_estimator: bool = False
    """Whether to use a many-to-luma estimator."""
    many2one_modality_to_estimate: Optional[str] = None
    """Modality to estimate with the many-to-one modality estimator."""
    many2one_modality_estimator: Optional[MLPConfig] = None
    """Modality estimator to estimate one modality from all the others."""
    use_random_many2one_modality: bool =  False
    """Whether to randomly choose the modality to estimate with the many-to-one modality estimator at each forward pass."""
    detach_many2one_modalities: bool = False
    """Whether to detach the input modalities to the many-to-one modality estimator."""
    geo_feature_modality: Optional[str] = None
    """Modality for which the gradient is not detached when passed to the radiance model.
    If None, gradients are never detached. If "random", a random modality is chosen at each forward pass."""

class LatentModel(BaseModel):
    """Standard multimodal model"""

    def __init__(
            self,
            config: LatentModelConfig,
            scene_box: SceneBox,
            modalities: Dict[str, int],
            **kwargs
    ):
        super().__init__(config, scene_box, modalities, **kwargs)
        self.config = config
        self.modalities = modalities
        if self.config.estimate_latents:
            latent_estimators = {}
            for k, v in self.config.latent_estimators.items():
                latent_estimators[k] = v.setup(modalities=modalities, **kwargs)
            self.latent_estimators = torch.nn.ModuleDict(latent_estimators)

        if self.config.use_many2one_modality_estimator:
            self.many2one_modalities = [m for m in self.modalities if m != self.config.many2one_modality_to_estimate]
            if self.config.use_random_many2one_modality:
                self.many2one_modality_heads = torch.nn.ModuleDict({
                    mod: MLPConfig(num_layers=3, hidden_dim=256,  out_activation="ReLU").setup(
                        input_dim=modalities[mod],
                        output_dim=256
                    ) for mod in self.many2one_modalities
                })
                estimator_config = copy.deepcopy(self.config.many2one_modality_estimator)
                estimator_config.num_layers -= 3
                self.many2one_modality_estimator = estimator_config.setup(
                    input_dim=256,
                    output_dim=modalities[self.config.many2one_modality_to_estimate] if not self.config.use_many2luma_estimator else 1,
                    **kwargs
                )
            else:
                self.many2one_modality_estimator = self.config.many2one_modality_estimator.setup(
                    input_dim=sum([modalities[m] for m in self.many2one_modalities]),
                    output_dim=modalities[self.config.many2one_modality_to_estimate] if not self.config.use_many2luma_estimator else 1,
                    **kwargs
                )

    @profiler.time_function
    def forward(self, ray_bundles, **kwargs):
        """Estimates the radiance and geometry values for each ray in the batch"""
        # Sample points along rays
        colliding_rays_masks = self.collider.update_ray_bundles(ray_bundles)

        masked_ray_bundles = {
            mod: ray_bundle[colliding_rays_masks[mod]]
            if ray_bundle is not None
            else None
            for mod, ray_bundle in ray_bundles.items()
        }
        ray_sampler_output = self.ray_sampler(masked_ray_bundles, sdf_fn=self.surface_model.get_sdf, **kwargs)
        samples_per_modality = ray_sampler_output["ray_samples_per_modality"]
        background_samples_per_modality = {}
        if self.config.use_background_model:
            self.collider.update_ray_bundles_for_background(ray_bundles)
            background_samples_per_modality = self.background_ray_sampler(ray_bundles)

        outputs = {}
        geo_feature_modality = self.config.geo_feature_modality \
            if self.config.geo_feature_modality != "random" \
            else random.choice(list(self.modalities.keys()))
        if self.config.use_random_many2one_modality:
            random_many2one_modality = random.choice(self.many2one_modalities)
        for mod in samples_per_modality.keys():

            samples = samples_per_modality.get(mod, None)
            background_samples = background_samples_per_modality.get(mod, None)
            mask = colliding_rays_masks.get(mod, None)

            if samples is None:
                outputs[mod] = None
                continue

            background_outputs = None
            if self.config.use_background_model:
                background_outputs = self.background_model(background_samples, **kwargs)

            if samples.shape[0] == 0 and background_outputs is not None:
                background_outputs.update({
                    "normals": torch.zeros(background_samples.shape[0], 3, device=mask.device),
                    "depth": torch.zeros(background_samples.shape[0], 1, device=mask.device),
                    "accumulation": torch.zeros(background_samples.shape[0], 1, device=mask.device),
                })
                background_outputs.update(self.radiance_model.additional_background_output(background_samples))
                outputs[mod] = background_outputs
            else:
                # Get weights
                geometry_outputs = self.surface_model(samples, **kwargs)

                radiance_outputs = self.radiance_model(
                    ray_samples=samples,
                    normals=geometry_outputs["normals"].detach(),
                    geo_feature=geometry_outputs["geo_feature"] \
                        if mod == geo_feature_modality or geo_feature_modality is None \
                        else geometry_outputs["geo_feature"].detach(),
                    **kwargs
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

                if self.config.estimate_latents:
                    for estimator in self.latent_estimators.values():
                        input_latent_name = list(estimator.config.input_latent.keys())[0]
                        output_latent_name = list(estimator.config.output_latent.keys())[0]
                        if input_latent_name == "radiance_modalities":
                            estimator_input = torch.cat([modality_outputs[m] for m in self.modalities], dim=-1)
                            modality_outputs.update({
                                f"estimated_{output_latent_name}": estimator(estimator_input)
                            })
                        else:
                            modality_outputs.update({
                                f"estimated_{output_latent_name}": estimator(modality_outputs[input_latent_name])
                            })

                if self.config.use_many2one_modality_estimator:
                    if self.config.use_random_many2one_modality:
                        estimator_input = modality_outputs[random_many2one_modality].clamp_max(1.0)
                    else:
                        estimator_input = torch.cat([modality_outputs[m] for m in self.many2one_modalities], dim=-1).clamp_max(1.0)
                    if self.config.detach_many2one_modalities:
                        estimator_input = estimator_input.detach()
                    if self.config.use_random_many2one_modality:
                        estimator_input = self.many2one_modality_heads[random_many2one_modality](estimator_input)
                    if self.config.use_many2luma_estimator:
                        modality_outputs[f"many2luma_{self.config.many2one_modality_to_estimate}"] = self.many2one_modality_estimator(estimator_input)
                    else:
                        modality_outputs[f"many2one_{self.config.many2one_modality_to_estimate}"] = self.many2one_modality_estimator(estimator_input)

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
        param_groups = super().get_param_groups()
        if self.config.estimate_latents:
            latent_estimators_param_group = {"latent_estimators": list(self.latent_estimators.parameters())}
            param_groups.update(latent_estimators_param_group)
        if self.config.use_many2one_modality_estimator:
            many2one_parameters = list(self.many2one_modality_estimator.parameters())
            if self.config.use_random_many2one_modality:
                for head in self.many2one_modality_heads.values():
                    many2one_parameters += list(head.parameters())
            many2one_param_group = {"many2one_modality_estimator": many2one_parameters}
            param_groups.update(many2one_param_group)
        return param_groups

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        """Returns the training callbacks from both the Dataloader and the Model."""
        callbacks = super().get_training_callbacks(training_callback_attributes)

        def set_step(step):
            self.step = step

        callbacks.append(
            TrainingCallback(
                where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                update_every_num_iters=1,
                func=set_step,
            )
        )
        return callbacks
