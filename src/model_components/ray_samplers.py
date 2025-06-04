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
Collection of sampling strategies
"""

from abc import abstractmethod
from typing import Callable, List, Optional, Union, Type, Dict
from dataclasses import dataclass, field
from torchtyping import TensorType

import torch
from torch import nn

from cameras.rays import RayBundle, RaySamples
from configs.configs import InstantiateConfig
from engine.callbacks import TrainingCallbackAttributes, TrainingCallback
from utils import profiler

def merge_ray_samples(ray_bundle: RayBundle, ray_samples_1: RaySamples, ray_samples_2: RaySamples):
    """Merge two set of ray samples and return sorted index which can be used to merge sdf values

    Args:
        ray_samples_1 : ray_samples to merge
        ray_samples_2 : ray_samples to merge
    """

    starts_1 = ray_samples_1.spacing_starts[..., 0]
    starts_2 = ray_samples_2.spacing_starts[..., 0]

    ends = torch.maximum(ray_samples_1.spacing_ends[..., -1:, 0], ray_samples_2.spacing_ends[..., -1:, 0])

    bins, sorted_index = torch.sort(torch.cat([starts_1, starts_2], -1), -1)

    bins = torch.cat([bins, ends], dim=-1)

    # Stop gradients
    bins = bins.detach()

    euclidean_bins = ray_samples_1.spacing_to_euclidean_fn(bins, ray_bundle)

    ray_samples = ray_bundle.get_ray_samples(
        bin_starts=euclidean_bins[..., :-1, None],
        bin_ends=euclidean_bins[..., 1:, None],
        spacing_starts=bins[..., :-1, None],
        spacing_ends=bins[..., 1:, None],
        spacing_to_euclidean_fn=ray_samples_1.spacing_to_euclidean_fn,
    )

    return ray_samples, sorted_index


@dataclass
class SamplerConfig(InstantiateConfig):
    """Base sampler config."""

    _target: Type = field(default_factory=lambda: Sampler)
    """Target class to instantiate."""
    num_samples: int = 32
    """Number of point to sample per ray"""
    train_stratified: bool = True
    """Whether to add gaussian noise to sampled points"""
    single_jitter: bool = False
    """Whether to use a same random jitter for all samples along a ray"""

@dataclass
class UniformSamplerConfig(SamplerConfig):
    """Uniform sampler config."""
    _target: Type = field(default_factory=lambda: UniformSampler)
    """Target class to instantiate."""

@dataclass
class LinearDisparitySamplerConfig(SamplerConfig):
    """Linear disparity sampler config."""
    _target: Type = field(default_factory=lambda: LinearDisparitySampler)
    """Target class to instantiate."""

@dataclass
class PDFSamplerConfig(SamplerConfig):
    """PDF sampler config."""
    _target: Type = field(default_factory=lambda: PDFSampler)
    """Target class to instantiate."""
    num_samples: int = 4
    """Number of point to sample per ray"""
    include_original: bool = True
    """Whether to include original samples in the PDF sampling output"""
    histogram_padding: float = 0.01
    """Amount of weights prior to computing PDF. This is used to prevent numerical issues when computing the PDF."""

@dataclass
class NeuSSamplerConfig(SamplerConfig):
    """NeuS sampler config."""
    _target: Type = field(default_factory=lambda: NeuSSampler)
    """Target class to instantiate."""
    num_samples_importance: int = 64
    """Number of point to sample per ray according to PDF"""
    num_upsample_steps: int = 4
    """Number of up-sample steps"""
    base_variance: float = 64
    """Base variance for the logistic function"""
    single_jitter: bool = True
    """Whether to use a same random jitter for all samples along a ray"""

class Sampler(nn.Module):
    """Base sampler class"""

    def __init__(
        self,
        config: SamplerConfig,
        train_stratified=None,
        single_jitter=None,
    ) -> None:
        super().__init__()
        self.config = config
        self.train_stratified = train_stratified if train_stratified is not None else self.config.train_stratified
        self.single_jitter = single_jitter if single_jitter is not None else self.config.single_jitter

    @abstractmethod
    def generate_ray_samples(self) -> RaySamples:
        """Generate Ray Samples"""

    def forward(self, *args, **kwargs) -> RaySamples:
        """Generate ray samples"""
        return self.generate_ray_samples(*args, **kwargs)

    def get_param_groups(self):
        """Returns the model parameters of the sampler."""
        params = list(self.parameters())
        return {"ray_sampler": params} if len(params) != 0 else {}

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        """Returns the training callbacks."""
        callbacks = []
        return callbacks

class SpacedSampler(Sampler):
    """Sample points according to a function.

    Args:
        spacing_fn: Function that dictates sample spacing (ie `lambda x : x` is uniform).
        spacing_fn_inv: The inverse of spacing_fn.
        train_stratified: Use stratified sampling during training. Defults to True
        single_jitter: Use a same random jitter for all samples along a ray. Defaults to False
    """

    def __init__(
        self,
        config: SamplerConfig,
        spacing_fn: Callable,
        spacing_fn_inv: Callable,
        train_stratified=None,
        single_jitter=None,
    ) -> None:
        super().__init__(config, train_stratified, single_jitter)
        self.spacing_fn = spacing_fn
        self.spacing_fn_inv = spacing_fn_inv

    def spacing_to_euclidean_fn(self, x, ray_bundle: RayBundle):
        """Estimate the euclidean position of a sample given the spacing function."""
        s_near, s_far = (self.spacing_fn(y) for y in (ray_bundle.nears.clone(), ray_bundle.fars.clone()))
        return self.spacing_fn_inv(s_far * x + s_near * (1 - x))

    def generate_ray_samples(
        self,
        ray_bundles: Dict[str, RayBundle] = None,
        num_samples: Optional[int] = None,
    ) -> Dict[str, RaySamples]:
        """Generates position samples according to spacing function.

        Args:
            ray_bundles: Rays to generate samples for
            num_samples: Number of samples per ray

        Returns:
            Positions and deltas for samples along a ray
        """
        ray_samples_per_modality = {}
        for mod, ray_bundle in ray_bundles.items():
            if ray_bundle is None:
                ray_samples_per_modality[mod] = None
                continue
            assert ray_bundle.nears is not None
            assert ray_bundle.fars is not None

            num_samples = num_samples or self.config.num_samples
            assert num_samples is not None
            num_rays = ray_bundle.origins.shape[0]

            bins = torch.linspace(0.0, 1.0, num_samples + 1).to(ray_bundle.origins.device)[None, ...]  # [1, num_samples+1]

            # (NeRFStudio comment) More complicated than it needs to be.
            if self.train_stratified and self.training:
                if self.single_jitter:
                    t_rand = torch.rand((num_rays, 1), dtype=bins.dtype, device=bins.device)
                else:
                    t_rand = torch.rand((num_rays, num_samples + 1), dtype=bins.dtype, device=bins.device)
                bin_centers = (bins[..., 1:] + bins[..., :-1]) / 2.0
                bin_upper = torch.cat([bin_centers, bins[..., -1:]], -1)
                bin_lower = torch.cat([bins[..., :1], bin_centers], -1)
                bins = bin_lower + (bin_upper - bin_lower) * t_rand

            euclidean_bins = self.spacing_to_euclidean_fn(bins, ray_bundle)  # [num_rays, num_samples+1]

            ray_samples = ray_bundle.get_ray_samples(
                bin_starts=euclidean_bins[..., :-1, None],
                bin_ends=euclidean_bins[..., 1:, None],
                spacing_starts=bins[..., :-1, None],
                spacing_ends=bins[..., 1:, None],
                spacing_to_euclidean_fn=self.spacing_to_euclidean_fn,
            )
            ray_samples_per_modality[mod] = ray_samples

        return ray_samples_per_modality


class UniformSampler(SpacedSampler):
    """Sample uniformly along a ray

    Args:
        train_stratified: Use stratified sampling during training. Defults to True
        single_jitter: Use a same random jitter for all samples along a ray. Defaults to False
    """

    def __init__(
        self,
        config: UniformSamplerConfig,
        train_stratified=None,
        single_jitter=None,
    ) -> None:
        super().__init__(
            config,
            spacing_fn=lambda x: x,
            spacing_fn_inv=lambda x: x,
            train_stratified=train_stratified,
            single_jitter=single_jitter,
        )

class LinearDisparitySampler(SpacedSampler):
    """Sample linearly in disparity along a ray

    Args:
        num_samples: Number of samples per ray
        train_stratified: Use stratified sampling during training. Defults to True
        single_jitter: Use a same random jitter for all samples along a ray. Defaults to False
    """

    def __init__(
        self,
        config: LinearDisparitySamplerConfig,
        train_stratified=None,
        single_jitter=None,
    ) -> None:
        super().__init__(
            config,
            spacing_fn=lambda x: 1 / x,
            spacing_fn_inv=lambda x: 1 / x,
            train_stratified=train_stratified,
            single_jitter=single_jitter,
        )

    def generate_ray_samples(
        self,
        ray_bundles: Optional[RayBundle] = None,
        num_samples: Optional[int] = None,
    ) -> List[RaySamples]:
        """Generates position samples according to spacing function.

        Args:
            ray_bundles: Rays to generate samples for
            num_samples: Number of samples per ray

        Returns:
            Positions and deltas for samples along a ray
        """
        ray_samples_per_modality = super().generate_ray_samples(ray_bundles, num_samples)
        return ray_samples_per_modality

class PDFSampler(Sampler):
    """Sample based on probability distribution

    Args:
        train_stratified: Randomize location within each bin during training.
        single_jitter: Use a same random jitter for all samples along a ray. Defaults to False
        include_original: Add original samples to ray.
        histogram_padding: Amount to weights prior to computing PDF.
    """

    def __init__(
        self,
        config: PDFSamplerConfig,
        train_stratified=None,
        single_jitter=None,
    ) -> None:
        super().__init__(config, train_stratified, single_jitter)

    def generate_ray_samples(
        self,
        ray_bundles: Optional[List[RayBundle]] = None,
        ray_samples_per_modality: Optional[List[RaySamples]] = None,
        weights_per_modality: List[TensorType[..., "num_samples", 1]] = None,
        num_samples: Optional[int] = None,
        eps: float = 1e-5,
    ) -> List[RaySamples]:
        """Generates position samples given a distribution.

        Args:
            ray_bundles: Rays per modality to generate samples for
            ray_samples_per_modality: Existing ray samples per modality
            weights_per_modality: Weights for each bin per modality
            num_samples: Number of samples per ray
            eps: Small value to prevent numerical issues.

        Returns:
            Positions and deltas for samples along a ray
        """

        num_samples = num_samples or self.config.num_samples
        assert num_samples is not None

        new_ray_samples_per_modality = []
        for ray_bundle, ray_samples, weights in zip(ray_bundles, ray_samples_per_modality, weights_per_modality):
            if ray_bundle is None:
                new_ray_samples_per_modality.append(None)
                continue
            assert ray_bundle.nears is not None
            assert ray_bundle.fars is not None


            if ray_samples is None:
                raise ValueError("ray_samples and ray_bundle must be provided")

            num_bins = num_samples + 1
            weights = weights[..., 0] + self.config.histogram_padding

            # Add small offset to rays with zero weight to prevent NaNs
            weights_sum = torch.sum(weights, dim=-1, keepdim=True)
            padding = torch.relu(eps - weights_sum)
            weights = weights + padding / weights.shape[-1]
            weights_sum += padding

            pdf = weights / weights_sum
            cdf = torch.min(torch.ones_like(pdf), torch.cumsum(pdf, dim=-1))
            cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)

            if self.config.train_stratified and self.training:
                # Stratified samples between 0 and 1
                u = torch.linspace(0.0, 1.0 - (1.0 / num_bins), steps=num_bins, device=cdf.device)
                u = u.expand(size=(*cdf.shape[:-1], num_bins))
                if self.config.single_jitter:
                    rand = torch.rand((*cdf.shape[:-1], 1), device=cdf.device) / num_bins
                else:
                    rand = torch.rand((*cdf.shape[:-1], num_samples + 1), device=cdf.device) / num_bins
                u = u + rand
            else:
                # Uniform samples between 0 and 1
                u = torch.linspace(0.0, 1.0 - (1.0 / num_bins), steps=num_bins, device=cdf.device)
                u = u + 1.0 / (2 * num_bins)
                u = u.expand(size=(*cdf.shape[:-1], num_bins))
            u = u.contiguous()

            assert (
                ray_samples.spacing_starts is not None and ray_samples.spacing_ends is not None
            ), "ray_sample spacing_starts and spacing_ends must be provided"
            assert ray_samples.spacing_to_euclidean_fn is not None, \
                "ray_samples.spacing_to_euclidean_fn must be provided"
            existing_bins = torch.cat(
                [
                    ray_samples.spacing_starts[..., 0],
                    ray_samples.spacing_ends[..., -1:, 0],
                ],
                dim=-1,
            )

            inds = torch.searchsorted(cdf, u, side="right")
            below = torch.clamp(inds - 1, 0, existing_bins.shape[-1] - 1)
            above = torch.clamp(inds, 0, existing_bins.shape[-1] - 1)
            cdf_g0 = torch.gather(cdf, -1, below)
            bins_g0 = torch.gather(existing_bins, -1, below)
            cdf_g1 = torch.gather(cdf, -1, above)
            bins_g1 = torch.gather(existing_bins, -1, above)

            t = torch.clip(torch.nan_to_num((u - cdf_g0) / (cdf_g1 - cdf_g0), 0), 0, 1)
            bins = bins_g0 + t * (bins_g1 - bins_g0)

            if self.config.include_original:
                bins, _ = torch.sort(torch.cat([existing_bins, bins], -1), -1)

            # Stop gradients
            bins = bins.detach()

            euclidean_bins = ray_samples.spacing_to_euclidean_fn(bins, ray_bundle)

            ray_samples = ray_bundle.get_ray_samples(
                bin_starts=euclidean_bins[..., :-1, None],
                bin_ends=euclidean_bins[..., 1:, None],
                spacing_starts=bins[..., :-1, None],
                spacing_ends=bins[..., 1:, None],
                spacing_to_euclidean_fn=ray_samples.spacing_to_euclidean_fn,
            )
            new_ray_samples_per_modality.append(ray_samples)

        return new_ray_samples_per_modality

class NeuSSampler(Sampler):
    """NeuS sampler that uses a sdf network to generate samples with fixed variance value in each iterations."""

    def __init__(
        self,
        config: NeuSSamplerConfig,
        train_stratified=None,
        single_jitter=None,
    ) -> None:
        super().__init__(config, train_stratified, single_jitter)
        self.config = config

        # samplers
        self.uniform_sampler = UniformSamplerConfig().setup(
            train_stratified=self.config.train_stratified,
            single_jitter=self.config.single_jitter
        )
        self.pdf_sampler = PDFSamplerConfig(
            include_original=False,
            single_jitter=self.config.single_jitter,
            histogram_padding=1e-5,
        ).setup()

    @profiler.time_function
    def generate_ray_samples(
        self,
        ray_bundles: Dict[str, RayBundle] = None,
        **kwargs,
    ) -> Dict[str, Dict[str, List[Union[RaySamples, TensorType]]]]:
        """Generates ray samples given a set of ray bundles for every modality."""
        sdf_fn = kwargs.get("sdf_fn", None)
        uniform_ray_samples_per_modality = kwargs.get("uniform_ray_samples_per_modality", None)
        assert ray_bundles is not None
        assert sdf_fn is not None

        # Start with uniform sampling
        if uniform_ray_samples_per_modality is None:
            uniform_ray_samples_per_modality = self.uniform_sampler(ray_bundles, num_samples=self.config.num_samples)

        ray_samples_per_modality = {}
        for mod, ray_bundle in ray_bundles.items():
            ray_samples = uniform_ray_samples_per_modality[mod]
            if ray_bundle is None:
                ray_samples_per_modality[mod] = None
                continue
            if ray_bundle.shape[0] == 0:
                ray_samples_per_modality[mod] = torch.tensor([])
                continue

            total_iters = 0
            sorted_index = None
            new_samples = ray_samples

            base_variance = self.config.base_variance


            while total_iters < self.config.num_upsample_steps:

                with torch.no_grad():
                    new_sdf = sdf_fn(new_samples)

                # merge sdf predictions
                if sorted_index is not None:
                    sdf_merge = torch.cat([sdf.squeeze(-1), new_sdf.squeeze(-1)], -1)
                    sdf = torch.gather(sdf_merge, 1, sorted_index).unsqueeze(-1)
                else:
                    sdf = new_sdf

                # compute with fix variances
                alphas = self.rendering_sdf_with_fixed_inv_s(
                    ray_samples, sdf.reshape(ray_samples.shape), inv_s=base_variance * 2**total_iters
                )

                weights = ray_samples.get_weights_from_alphas(alphas[..., None])
                weights = torch.cat((weights, torch.zeros_like(weights[:, :1])), dim=1)

                new_samples = self.pdf_sampler(
                    [ray_bundle],
                    [ray_samples],
                    [weights],
                    num_samples=self.config.num_samples_importance // self.config.num_upsample_steps,
                )[0]

                ray_samples, sorted_index = merge_ray_samples(
                    ray_bundle, ray_samples, new_samples
                )

                total_iters += 1
            ray_samples_per_modality[mod] = ray_samples

        return {"ray_samples_per_modality": ray_samples_per_modality}

    def rendering_sdf_with_fixed_inv_s(self, ray_samples: RaySamples, sdf: torch.Tensor, inv_s):
        """rendering given a fixed inv_s as NeuS"""
        batch_size = ray_samples.shape[0]
        prev_sdf, next_sdf = sdf[:, :-1], sdf[:, 1:]
        deltas = ray_samples.deltas[:, :-1, 0]
        mid_sdf = (prev_sdf + next_sdf) * 0.5
        cos_val = (next_sdf - prev_sdf) / (deltas + 1e-5)

        # ----------------------------------------------------------------------------------------------------------
        # Use min value of [ cos, prev_cos ]
        # Though it makes the sampling (not rendering) a little bit biased, this strategy can make the sampling more
        # robust when meeting situations like below:
        #
        # SDF
        # ^
        # |\          -----x----...
        # | \        /
        # |  x      x
        # |---\----/-------------> 0 level
        # |    \  /
        # |     \/
        # |
        # ----------------------------------------------------------------------------------------------------------
        prev_cos_val = torch.cat([torch.zeros([batch_size, 1], device=sdf.device), cos_val[:, :-1]], dim=-1)
        cos_val = torch.stack([prev_cos_val, cos_val], dim=-1)
        cos_val, _ = torch.min(cos_val, dim=-1, keepdim=False)
        cos_val = cos_val.clip(-1e3, 0.0)

        dist = deltas
        prev_esti_sdf = mid_sdf - cos_val * dist * 0.5
        next_esti_sdf = mid_sdf + cos_val * dist * 0.5
        prev_cdf = torch.sigmoid(prev_esti_sdf * inv_s)
        next_cdf = torch.sigmoid(next_esti_sdf * inv_s)
        alpha = (prev_cdf - next_cdf + 1e-5) / (prev_cdf + 1e-5)

        return alpha
