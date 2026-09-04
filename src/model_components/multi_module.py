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

import torch
from dataclasses import dataclass, field
from typing import Type, List

from configs.configs import InstantiateConfig
from engine.callbacks import TrainingCallbackAttributes, TrainingCallback
from field_components.base_field_component import FieldComponentConfig


@dataclass
class MultiModuleConfig(InstantiateConfig):

    _target: Type = field(default_factory=lambda: MultiModule)
    module: FieldComponentConfig = field(default_factory=lambda: FieldComponentConfig)
    """Module to replicate"""
    key_word_init: str = "n_scene"
    """Keyword to identify the parameters to consider during initialization"""
    key_word_forward: str = "scene_index"
    """Keyword to identify the parameters to consider during forward pass"""

class MultiModule(torch.nn.Module):

    config: MultiModuleConfig

    def __init__(
            self,
            config: MultiModuleConfig,
            *args,
            **kwargs
    ):
        super().__init__()
        self.config = config
        number_of_replicas = kwargs[config.key_word_init]
        self.module_list = torch.nn.ModuleList([self.config.module.setup(*args, **kwargs) for _ in range(number_of_replicas)])
        pass

    def forward(self, *args, **kwargs) -> torch.Tensor:
        index = kwargs[self.config.key_word_forward]
        return self.module_list[index](*args, **kwargs)

    def get_training_callbacks(
            self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        """Returns the training callbacks from both the Dataloader and the Model."""
        callbacks = []
        for module in self.module_list:
            callbacks += module.get_training_callbacks(training_callback_attributes)
        return callbacks

    def get_model_parameters(self):
        return self.module_list[0].get_model_parameters()

    def get_out_dim(self):
        """Get the output dimension of the module."""
        return self.module_list[0].get_out_dim()
