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
Single variance network proposed in NeuS.
"""

import torch
from torch import nn

class SingleVarianceNetwork(nn.Module):
    """Variance network in NeuS

    Args:
        init_val: init value in NeuS variance network
    """

    def __init__(self, init_val):
        super().__init__()
        self.register_parameter("s", nn.Parameter(init_val * torch.ones(1), requires_grad=True))

    def forward(self, x):
        """Returns current variance value"""
        return torch.ones([len(x), 1], device=x.device) * torch.exp(self.s * 10.0)

    def get_inv_variance(self):
        """return current variance value"""
        return torch.exp(self.s * 10.0).clip(1e-6, 1e6)
