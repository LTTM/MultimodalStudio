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
Framework launcher for MultimodalStudio.
"""

from configs.configs import Config
from engine.trainer import Trainer

if __name__ == '__main__':
    config = Config()
    config.save_config()

    trainer = Trainer(config)
    trainer.setup()

    if config.run_mode == "train":
        trainer.train()
    elif config.run_mode == "eval":
        step = trainer.step_start
        trainer.eval(step, view_ids=config.view_ids)

    print("Done!")
