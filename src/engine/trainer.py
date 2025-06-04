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
Trainer class
"""

import os
from dataclasses import dataclass, asdict
from typing import List
from tqdm import tqdm

import torch
import lightning as L
from rich.console import Console

from engine.callbacks import TrainingCallbackLocation
from utils import writer, profiler
from utils.decorators import check_main_thread
from utils.misc import check_step
from utils.writer import TimeWriter, EventName

CONSOLE = Console(width=120)

@dataclass
class Trainer:
    """
    Class to train the model. It handles the training/evaluation loop, logging, and checkpointing.
    """

    def __init__(self, config):
        self.config = config
        self.n_gpu = config.n_gpu if self.config.run_mode == "train" else 1
        self.mixed_precision = "16-mixed" if config.mixed_precision else "32"
        self.output_dir = config.get_output_dir()
        self.checkpoint_dir = config.get_checkpoint_dir()

        torch.set_float32_matmul_precision(self.config.matmul_precision)

        self.fabric = L.Fabric(
            accelerator='cuda',
            devices=self.n_gpu,
            precision=self.mixed_precision,
            strategy='ddp' if self.n_gpu > 1 else 'auto',
        )
        self.fabric.launch()
        self.fabric.seed_everything(654824)
        self.fabric.print(config)

    def setup(self):
        """Initialize the pipeline, logger, and profiler."""

        # Initialize pipeline
        self.pipeline = self.config.pipeline.setup(
            fabric=self.fabric,
            trainer_config=self.config,
            output_dir=self.output_dir,
            checkpoint_dir=self.checkpoint_dir,
            mixed_precision=self.mixed_precision,
        )

        self.pipeline.setup()
        self.load_checkpoint(self.checkpoint_dir)

        # Initialize Logger and Profiler
        self.setup_logger()
        self.setup_profiler()

    def train(self):
        """Run the training loop."""

        self.pipeline.set_train()
        with TimeWriter(writer, EventName.TOTAL_TRAIN_TIME):
            for step in tqdm(
                    range(self.step_start, self.config.max_num_iterations + 1),
                    disable=self.fabric.global_rank != 0
            ):

                # training callbacks before the training iteration
                for callback in self.pipeline.callbacks:
                    callback.run_callback_at_location(step, location=TrainingCallbackLocation.BEFORE_TRAIN_ITERATION)

                with TimeWriter(writer, EventName.ITER_TRAIN_TIME, step=step) as train_t:
                    losses, total_loss, metrics = self.pipeline.train_step(step)

                # training callbacks after the training iteration
                for callback in self.pipeline.callbacks:
                    callback.run_callback_at_location(step, location=TrainingCallbackLocation.AFTER_TRAIN_ITERATION)

                if self.fabric.global_rank == 0:
                    writer.put_time(
                        name=EventName.TRAIN_RAYS_PER_SEC,
                        duration=self.config.pipeline.datamanager.pixel_sampler.num_rays_per_modality * \
                                 len(self.pipeline.datamanager.modalities) / train_t.duration,
                        step=step,
                        avg_over_steps=True,
                    )

                if check_step(step, self.config.logging.steps_per_log, skip_first=False):
                    total_loss = self.fabric.all_reduce(total_loss, reduce_op="mean")
                    losses = self.fabric.all_reduce(losses, reduce_op="mean")
                    metrics = self.fabric.all_reduce(metrics, reduce_op="mean")
                    writer.put_scalar(name="Train Loss", scalar=total_loss, step=step)
                    writer.put_dict(name="Train Loss Dict", scalar_dict=losses, step=step)
                    writer.put_dict(name="Train Metrics Dict", scalar_dict=metrics, step=step)

                losses, total_loss, metrics = self.pipeline.eval_step(step)
                if check_step(step, self.config.steps_per_eval_batch):
                    total_loss = self.fabric.all_reduce(total_loss, reduce_op="mean")
                    losses = self.fabric.all_reduce(losses, reduce_op="mean")
                    metrics = self.fabric.all_reduce(metrics, reduce_op="mean")
                    writer.put_scalar(name="Eval Loss", scalar=total_loss, step=step)
                    writer.put_dict(name="Eval Loss Dict", scalar_dict=losses, step=step)
                    writer.put_dict(name="Eval Metrics Dict", scalar_dict=metrics, step=step)

                self.fabric.barrier()
                if check_step(step, self.config.steps_per_save):
                    self.save_checkpoint(step)
                writer.write_out_storage(step, self.config.logging.steps_per_flush_buffer)

        profiler.flush_profiler(self.config.logging)

    def eval(self, step: int, view_ids: List[int] = None):
        """Run the evaluation loop."""
        self.pipeline.set_eval()
        # training callbacks before the training iteration
        for callback in self.pipeline.callbacks:
            callback.run_callback_at_location(step, location=TrainingCallbackLocation.BEFORE_TRAIN_ITERATION)
        if view_ids is None:
            self.pipeline.evaluator.render_all_eval_views(
                step,
                output_path=os.path.join(self.output_dir, 'evaluation')
            )
        else:
            self.pipeline.evaluator.render_specific_views(
                step,
                view_ids=view_ids,
                output_path=os.path.join(self.output_dir, 'evaluation')
            )
        self.pipeline.evaluator.export_mesh(step)
        self.pipeline.evaluator.export_poses(step)
        profiler.flush_profiler(self.config.logging)

    @check_main_thread
    def setup_logger(self):
        """Initialize the logger."""
        writer_log_path = os.path.join(self.output_dir, "logs")
        os.makedirs(writer_log_path, exist_ok=True)
        writer.setup_event_writer(self.config, log_dir=writer_log_path)
        writer.setup_local_writer(self.config.logging, max_iter=self.config.max_num_iterations)
        # writer.put_config(name="config", config_dict=asdict(self.config), step=0)
        writer.put_config(name="config", config_dict=str(self.config), step=0)

    @check_main_thread
    def setup_profiler(self):
        """Initialize the profiler."""
        if self.config.logging.enable_profiler:
            profiler.setup_profiler(self.config.logging, os.path.join(self.output_dir, "logs"))

    @check_main_thread
    def save_checkpoint(self, step: int) -> None:
        """Save the model and optimizers

        Args:
            step: current number of steps in training
        """
        # possibly make the checkpoint directory
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir, exist_ok=True)
        # save the checkpoint
        ckpt_name = f"step-{step:09d}.ckpt"
        ckpt_path = os.path.join(self.checkpoint_dir, ckpt_name)
        torch.save(self.pipeline.state_dict(step), ckpt_path)
        # possibly delete old checkpoints
        if self.config.save_only_latest_checkpoint:
            # delete everything else in the checkpoint folder
            for f in os.listdir(self.checkpoint_dir):
                if f != ckpt_name:
                    os.remove(os.path.join(self.checkpoint_dir, f))

    def load_checkpoint(self, checkpoint_dir = None) -> None:
        """Helper function to load pipeline and optimizer from specified checkpoint"""
        load_dir = checkpoint_dir if checkpoint_dir is not None else self.checkpoint_dir
        os.makedirs(load_dir, exist_ok=True)
        if load_dir is not None and len(os.listdir(load_dir)) != 0:
            load_step = self.config.load_step
            if load_step is None:
                print("Loading latest checkpoint from load_dir")
                # NOTE: this is specific for the checkpoint name format
                load_step = sorted(int(x[x.find("-") + 1 : x.find(".")]) for x in os.listdir(load_dir))[-1]
            load_path = os.path.join(load_dir, f"step-{load_step:09d}.ckpt")
            assert os.path.exists(load_path), f"Checkpoint {load_path} does not exist"
            loaded_state = torch.load(load_path, map_location="cpu")
            self.step_start = loaded_state["step"] + 1
            # load the checkpoints for pipeline, optimizers, and gradient scalar
            self.pipeline.load_state_dict(loaded_state)
            CONSOLE.print(f"done loading checkpoint from {load_path}")
        else:
            self.step_start = 0
            CONSOLE.print("No checkpoints to load, training from scratch")
