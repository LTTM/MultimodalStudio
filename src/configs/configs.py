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
Code to manage general configurations, such as Trainer config or Run config
"""

import os
import subprocess
from datetime import datetime
from typing import Tuple, Optional, Any, Type, List, Literal, Union
from dataclasses import dataclass, field
import yaml
import tyro

from utils import writer


class PrintableConfig:  # pylint: disable=too-few-public-methods
    """Printable Config defining str function"""

    def __str__(self):
        lines = [self.__class__.__name__]
        for key, val in vars(self).items():
            if key == "_target":
                continue
            if isinstance(val, Tuple):
                flattened_val = "["
                for item in val:
                    flattened_val += str(item) + "\n"
                flattened_val = flattened_val.rstrip("\n")
                val = flattened_val + "]"
            lines += f"{key}: {str(val)}".split("\n")
        if lines[0] == "Config":
            lines[0] += ":"
        return "\n    ".join(lines)

@dataclass
class InstantiateConfig(PrintableConfig):  # pylint: disable=too-few-public-methods
    """Config class for instantiating an the class specified in the _target attribute."""

    _target: Type

    def setup(self, **kwargs) -> Any:
        """Returns the instantiated object using the config."""
        return self._target(self, **kwargs)

@dataclass
class LocalWriterConfig(InstantiateConfig):
    """Local Writer config"""

    _target: Type = field(default_factory=lambda: writer.LocalWriter)
    """target class to instantiate"""
    enable: bool = False
    """if True enables local logging, else disables"""
    stats_to_track: Tuple[writer.EventName, ...] = (
        writer.EventName.ITER_TRAIN_TIME,
        writer.EventName.TRAIN_RAYS_PER_SEC,
        writer.EventName.CURR_TEST_PSNR,
        writer.EventName.VIS_RAYS_PER_SEC,
        writer.EventName.TEST_RAYS_PER_SEC,
    )
    """specifies which stats will be logged/printed to terminal"""
    max_log_size: int = 10
    """maximum number of rows to print before wrapping. if 0, will print everything."""

    def setup(self, banner_messages: Optional[List[str]] = None, **kwargs) -> Any:
        """Instantiate local writer

        Args:
            banner_messages: List of strings that always print at the bottom of screen.
        """
        return self._target(self, banner_messages=banner_messages, **kwargs)

@dataclass
class LoggingConfig(PrintableConfig):
    """Configuration of loggers and profilers"""

    steps_per_log: int = 10
    """number of steps between logging stats"""
    max_buffer_size: int = 20
    """maximum history size to keep for computing running averages of stats.
     e.g. if 20, averages will be computed over past 20 occurances."""
    steps_per_flush_buffer: int = 100
    """number of steps between flushing buffer to disk"""
    local_writer: LocalWriterConfig = field(default_factory=lambda: LocalWriterConfig)
    """if provided, will print stats locally. if None, will disable printing"""
    enable_profiler: bool = False
    """whether to enable profiling code; prints speed of functions at the end of a program.
    profiler logs run times of functions and prints at end of training"""
    profiler: Literal["none", "basic", "pytorch"] = "basic"
    """how to profile the code;
        "basic" - prints speed of all decorated functions at the end of a program.
        "pytorch" - same as basic, but it also traces few training steps.
    """

from pipelines.base_pipeline import BasePipelineConfig

@dataclass
class TrainerConfig(PrintableConfig):
    """Configuration for training regimen"""

    output_dir: str = os.path.join("output")
    """relative or absolute output directory to save all checkpoints and logging"""
    method_name: Optional[str] = None
    """Method name. Required to set in python or via cli"""
    pipeline: BasePipelineConfig = field(default_factory=lambda: BasePipelineConfig)
    """Pipeline configuration"""
    steps_per_save: int = 5000
    """Number of steps between saves."""
    steps_per_eval_batch: int = 100
    """Number of steps between randomly sampled batches of rays."""
    steps_per_eval_image: int = 1000
    """Number of steps between single eval images."""
    steps_per_export_mesh: int = 5000
    """Number of steps between eval mesh."""
    steps_per_export_poses: int = 5000
    """Number of steps between eval camera poses."""
    steps_per_eval_all_images: int = 25000
    """Number of steps between eval all images."""
    max_num_iterations: int = 100000
    """Maximum number of iterations to run."""
    mixed_precision: bool = False
    """Whether or not to use mixed precision for training."""
    matmul_precision: Literal["highest", "high", "medium"] = "high"
    """Precision to use for matmuls."""
    save_only_latest_checkpoint: bool = True
    """Whether to only save the latest checkpoint or all checkpoints."""
    logging: LoggingConfig = field(default_factory=lambda: LoggingConfig)
    """Logging configuration"""
    vis: Literal["wandb", "tensorboard"] = "tensorboard"
    """Which visualizer to use."""
    data_dir: Optional[str] = None
    """Alias for --pipeline.datamanager.dataparser.data"""
    n_gpu: int = 1
    """Number of GPUs to use"""
    load_dir: Optional[str] = None
    """Optionally specify a pre-trained model directory to load from."""
    load_step: Optional[int] = None
    """Optionally specify model step to load from; if none, will find most recent model in load_dir."""

    def is_wandb_enabled(self) -> bool:
        """Checks if wandb is enabled."""
        return "wandb" == self.vis

    def is_tensorboard_enabled(self) -> bool:
        """Checks if tensorboard is enabled."""
        return "tensorboard" == self.vis


@dataclass
class RunConfig(PrintableConfig):
    """
    Current run configuration.
    """
    mode: str
    """run mode (train,test,inference)"""
    conf_path: str
    """configuration file"""
    scene: str
    """scene name"""
    version: str = ""
    """run version name"""
    view_ids: Union[None, tuple[int, ...]] = None
    """view ids to evaluate one-shot"""

    def __post_init__(self):
        with open(self.conf_path, 'r') as file:
            self.conf: dict = yaml.safe_load(file) # configuration dictionary

from configs.method_configs import method_configs

@dataclass
class Config(TrainerConfig):
    """
    Merges RunConfig with model Config
    """
    def __init__(self):
        command_line_config: RunConfig = tyro.cli(RunConfig)
        self.overwrite_vars(method_configs[command_line_config.conf['method']])
        self.conf_name = self.get_conf_name(command_line_config)
        self.run_mode = command_line_config.mode
        self.data_dir = command_line_config.scene
        scene_name = command_line_config.scene.strip().split('/')[-1]
        if command_line_config.version == "":
            command_line_config.version = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.output_dir = os.path.join('output', self.check_git_branch(), scene_name,
                               self.method_name, self.conf_name, command_line_config.version)
        self.update_config(command_line_config.conf)
        self.view_ids = command_line_config.view_ids if command_line_config.view_ids else None

    def overwrite_vars(self, src):
        """Overwrite class attributes given an object of the same class."""
        for attr in vars(src):
            setattr(self, attr, getattr(src, attr))

    def update_config(self, update: dict):
        """Update a Config object based on a given dict"""
        def set_attribute(target, update):
            """Recursive argument updater"""
            for key in update.keys():
                if isinstance(target, dict):
                    if hasattr(target.get(key), '__dict__'):
                        set_attribute(target[key], update[key])
                    elif isinstance(target.get(key), dict):
                        set_attribute(target[key], update[key])
                    else:
                        target[key] = update[key]
                else:
                    if hasattr(getattr(target, key), '__dict__'):
                        set_attribute(getattr(target, key), update[key])
                    elif isinstance(getattr(target, key), dict):
                        set_attribute(getattr(target, key), update[key])
                    else:
                        setattr(target, key, update[key])

        for key in update:
            if key in self.__dict__:
                target = getattr(self, key)
                if hasattr(target, '__dict__'):
                    set_attribute(target, update[key])
                elif isinstance(target, dict):
                    set_attribute(target, update[key])
                else:
                    setattr(self, key, update[key])

    def check_git_branch(self) -> str:
        """
        Get the git branch to be logged.
        """
        try:
            branch_list = subprocess.check_output(['git', 'branch'], cwd='.').splitlines()
        except(subprocess.CalledProcessError, OSError):
            return ""
        for branch_name in branch_list:
            if '*' in branch_name.decode():
                return branch_name.decode()[2:]
        return ""

    def get_conf_name(self, command_line_config) -> str:
        """Return configuration file name"""
        return os.path.split(command_line_config.conf_path)[-1][:-5]

    def get_output_dir(self) -> str:
        """Retrieve the base directory to set relative paths"""
        return self.output_dir


    def get_checkpoint_dir(self) -> str:
        """Retrieve the checkpoint directory"""
        return os.path.join(self.get_output_dir(), 'checkpoints')

    def save_config(self) -> None:
        """Save config to base directory"""
        output_dir = self.get_output_dir()
        assert output_dir is not None
        os.makedirs(output_dir, exist_ok=True)
        config_yaml_path = os.path.join(output_dir, "config.yaml")
        with open(config_yaml_path, 'w') as file:
            file.write(str(self))
