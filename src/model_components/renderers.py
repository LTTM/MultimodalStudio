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
Collection of renderers
"""

from dataclasses import dataclass, field
from typing import Union, Type, Dict, Any

import torch
from torch import nn
from torchtyping import TensorType
from typing_extensions import Literal

from cameras.rays import RaySamples
from configs.configs import InstantiateConfig
from utils import profiler

@dataclass
class RendererConfig(InstantiateConfig):
    """Renderer config"""

    _target: Type = field(default_factory=lambda: Renderer)
    renderers: Dict[str, Any] = field(default_factory=lambda: {"rgb": "RadianceRenderer"})
    """Dictionary of renderers to use for each modality."""
    background_color: Union[Literal["random", "white", "black", "None"], TensorType[3]] = "None"
    """Background color to use for the background samples.
    If None, the background estimated by the background model is used."""

class Renderer:
    """
    Renderer module. This module is responsible for rendering the final pixel values.
    It renders radiance values, depth, normals, and other modalities.
    """

    config: RendererConfig

    def __init__(self, config):
        super().__init__()
        self.config = config
        for element in self.config.renderers:
            renderer_class = self.config.renderers[element]
            setattr(self, element, renderer_class())

    def prepare_background(self, background_samples, mask, n_channels):
        """Prepares background values for rendering."""
        if self.config.background_color == "None" and background_samples is not None:
            return background_samples
        elif self.config.background_color == "white":
            return torch.ones((mask.shape[0], n_channels), device=mask.device)
        elif self.config.background_color == "black":
            return torch.zeros((mask.shape[0], n_channels), device=mask.device)
        elif self.config.background_color == "random":
            return torch.rand((mask.shape[0], n_channels), device=mask.device)
        raise ValueError(f"Background color {self.config.background_color} not supported.")

    @profiler.time_function
    def render(
            self,
            weights: TensorType["bs", "num_samples", 1],
            data_fields: Dict[str, Union[RaySamples, TensorType["bs", "num_samples", 1]]],
            mask: TensorType,
    ) -> Dict[str, TensorType["bs", 1]]:
        """
        Render the final pixel values.
        Args:
            weights: Weights for each sample of each ray.
            data_fields: Dictionary of estimated values for each ray to render.
            mask: Mask of the rays intersecting the region of interest.

        Returns:
            outputs: Dictionary of rendered values for each modality.
        """
        outputs = {}

        for mod in data_fields:

            if mod == "background":
                continue

            if mod in self.config.renderers:
                n_channels = data_fields[mod].shape[-1]
                if data_fields["background"] is not None:
                    rendered_color = self.prepare_background(data_fields["background"][mod], mask, n_channels)
                else:
                    rendered_color = self.prepare_background(None, mask, n_channels)
                rendered_color[mask] = getattr(self, mod)(data_fields[mod], weights, rendered_color[mask])
                outputs[mod] = rendered_color
            elif mod == 'normals':
                rendered_normals = torch.zeros(
                    (mask.shape[0], 3),
                    device=data_fields[mod].device,
                    dtype=data_fields[mod].dtype
                )
                renderer = NormalsRenderer()
                rendered_normals[mask] = renderer(data_fields[mod], weights)
                outputs[mod] = rendered_normals
            elif mod == 'depth':
                steps = (data_fields[mod].frustums.starts + data_fields[mod].frustums.ends) / 2
                rendered_depth = torch.zeros(mask.shape[0], 1, device=steps.device, dtype=steps.dtype)
                renderer = DepthRenderer()
                rendered_depth[mask] = renderer(steps, weights)
                outputs[mod] = rendered_depth
            else:
                rendered_field = torch.zeros(
                    (mask.shape[0], data_fields[mod].shape[-1]),
                    device=data_fields[mod].device,
                    dtype=data_fields[mod].dtype
                )
                renderer = SemanticRenderer()
                rendered_field[mask] = renderer(data_fields[mod], weights)
                outputs[mod] = rendered_field

        rendered_accumulation = torch.zeros(mask.shape[0], 1, device=weights.device, dtype=weights.dtype)
        renderer = AccumulationRenderer()
        rendered_accumulation[mask] = renderer(weights)
        outputs["accumulation"] = rendered_accumulation
        return outputs

class BaseRenderer(nn.Module):
    """Base class for all renderers."""

    def forward(
        self,
        *args,
        **kwargs,
    ) -> TensorType:
        """Forward pass of the renderer."""
        return self.render(*args, **kwargs)

class RadianceRenderer(BaseRenderer):
    """Standard radiance volumetric rendering."""

    @classmethod
    def render(
        cls,
        radiance_values: TensorType["bs":..., "num_samples", "num_channels"],
        weights: TensorType["bs":..., "num_samples", 1],
        background_color: TensorType["bs":..., "num_channels"],
    ) -> TensorType["bs":..., 3]:
        """Composite samples along ray and render color image

        Args:
            radiance_values: Radiance value for each sample
            weights: Weights for each sample
            background_color: Color of the background pixel

        Returns:
            Outputs of radiance values.
        """
        comp_radiance = torch.sum(weights * radiance_values, dim=-2)
        accumulated_weight = torch.sum(weights, dim=-2)
        comp_radiance = comp_radiance + background_color * (1.0 - accumulated_weight)
        # if not self.training:     #TODO: check if this is necessary
        #     torch.clamp_(comp_radiance, min=0.0, max=1.0)
        return comp_radiance

class AccumulationRenderer(BaseRenderer):
    """Accumulated value along a ray."""

    @classmethod
    def render(
        cls,
        weights: TensorType["bs":..., "num_samples", 1],
    ) -> TensorType["bs":..., 1]:
        """Composite samples along ray and calculate accumulation.

        Args:
            weights: Weights for each sample

        Returns:
            Outputs of accumulated values.
        """

        accumulation = torch.sum(weights, dim=-2)
        return accumulation

class DepthRenderer(BaseRenderer):
    """Calculate depth along ray."""

    @classmethod
    def render(
        cls,
        steps: TensorType[..., "num_samples", 1],
        weights: TensorType[..., "num_samples", 1],
    ) -> TensorType[..., 1]:
        """Composite samples along ray and calculate depths.

        Args:
            steps: Set of steps along the rays.
            weights: Weights for each sample.

        Returns:
            Outputs of depth values.
        """
        depth = torch.sum(weights * steps, dim=-2)
        depth = torch.clip(depth, steps.min(), steps.max())
        return depth

class SemanticRenderer(BaseRenderer):
    """Calculate semantics along the ray."""

    @classmethod
    def render(
        cls,
        semantics: TensorType["bs":..., "num_samples", "num_classes"],
        weights: TensorType["bs":..., "num_samples", 1],
    ) -> TensorType["bs":..., "num_classes"]:
        """Calculate semantics along the ray."""
        sem = torch.sum(weights * semantics, dim=-2)
        return sem

class NormalsRenderer(BaseRenderer):
    """Calculate normals along the ray."""

    @classmethod
    def render(
        cls,
        normals: TensorType["bs":..., "num_samples", 3],
        weights: TensorType["bs":..., "num_samples", 1],
    ) -> TensorType["bs":..., 3]:
        """Calculate normals along the ray."""
        n = torch.sum(weights * normals, dim=-2)
        return n
