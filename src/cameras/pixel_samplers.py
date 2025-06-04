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
Pixel Samplers
"""
import random
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Type

import torch

from configs.configs import InstantiateConfig

@dataclass
class PixelSamplerConfig(InstantiateConfig):
    """General configuration for pixel samplers."""

    _target: Type = field(default_factory=lambda: PixelSampler)
    num_rays_per_modality: int = 32
    """Number of rays to sample for each modality per batch"""

@dataclass
class UniformPixelSamplerConfig(PixelSamplerConfig):
    """Configuration for uniform pixel sampling."""

    _target: Type = field(default_factory=lambda: UniformPixelSampler)

class PixelSampler:
    """Base class for pixel samplers."""

    config: PixelSamplerConfig

    def __init__(
            self,
            config: PixelSamplerConfig,
            device=None,
    ):
        self.config = config
        # Change seed for every device to sample different pixels
        if device is not None:
            self.generator = torch.Generator()
            self.generator.manual_seed(random.randint(0, 1000000) + device.index)

    @abstractmethod
    def sample(self, frames):
        """Abstract function to sample pixels from the given frames"""
        return NotImplementedError

class UniformPixelSampler(PixelSampler):
    """Uniform pixel sampler. Samples random pixels from all the training views."""

    config = UniformPixelSamplerConfig

    def __init__(
            self,
            config: UniformPixelSamplerConfig,
            device,
    ):
        super().__init__(config, device)

    def sample(self, frames):
        """Returns a set of random pixels for each modality and the respective radiance values."""
        coordinates = {}
        pixels = {}
        for mod in frames.keys():
            data = frames[mod]
            n_frames, height, width, _ = data['images'].shape
            random_indexes = torch.randint(low=0, high=n_frames, size=(self.config.num_rays_per_modality, 1),
                                           dtype=torch.int32, generator=self.generator)
            frame_indexes = data['indexes'][random_indexes]
            pixels_x = torch.randint(low=0, high=width, size=(self.config.num_rays_per_modality, 1),
                                     dtype=torch.int32, generator=self.generator)
            pixels_y = torch.randint(low=0, high=height, size=(self.config.num_rays_per_modality, 1),
                                     dtype=torch.int32, generator=self.generator)
            pixels_yx = torch.cat([frame_indexes, pixels_y, pixels_x], dim=-1)
            values = data['images'][random_indexes.squeeze(), pixels_yx[:, 1], pixels_yx[:, 2]]
            coordinates[mod] = pixels_yx
            pixels[mod] = values
        return coordinates, pixels

class DensePixelSampler(PixelSampler):
    """Dense pixel sampler. Samples all the pixels in the image."""

    def __init__(self):
        super().__init__(None)

    def sample(self, frames):
        """Returns an ordered set of pixels for each modality and the respective radiance values."""
        coordinates = {}
        pixels = {}

        for mod in frames.keys():
            data = frames[mod]
            if data['images'] is None:
                coordinates[mod] = None
                pixels[mod] = None
                continue
            _, height, width, _ = data['images'].shape
            frame_indexes = data['indexes'] * torch.ones((height*width,), dtype=torch.int32)
            pixels_x = torch.arange(width, dtype=torch.int32).expand((height,-1)).flatten()
            pixels_y = torch.arange(height, dtype=torch.int32).expand((width,-1)).T.flatten()
            pixels_yx = torch.stack([frame_indexes, pixels_y, pixels_x], dim=-1)
            coordinates[mod] = pixels_yx
            pixels[mod] = data['images'].squeeze(dim=0)
        return coordinates, pixels
