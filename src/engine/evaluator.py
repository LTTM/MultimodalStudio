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
Evaluator class for evaluating the model performance.
"""
import copy
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Type, Union, List, Tuple, Literal

import random
import numpy as np
import torch
from tqdm import tqdm

from configs.configs import InstantiateConfig
from data.datasets import BaseDataset, BaseAlignedDataset, BaseUnalignedDataset
from data.scene_box import SceneBox

from evaluator_components.mesh_extractors import MeshExtractorConfig
from evaluator_components.pose_extractor import PoseExtractorConfig
from model_components import polarizer
from model_components.ray_generators import RayGenerator
from utils import writer
from utils.writer import EventName, TimeWriter
from utils.eval_utils import export_renderings, compute_metrics, eval_model_query, render_outputs, combine_renderings, \
    export_metrics_to_file
from utils.misc import check_step, get_dict_to_cpu


@dataclass
class EvaluatorConfig(InstantiateConfig):
    """
    Evaluator config class.
    """
    _target: Type = field(default_factory=lambda: Evaluator)
    steps_per_eval_batch: int = 100
    """Number of steps between randomly sampled batches of rays."""
    steps_per_eval_image: int = 1000
    """Number of steps between single eval images."""
    steps_per_eval_all_images: int = 25000
    """Number of steps between eval all images."""
    steps_per_export_mesh: int = 5000
    """Number of steps between eval mesh."""
    steps_per_export_poses: int = 5000
    """Number of steps between eval camera poses."""
    eval_num_rays_per_chunk: int = 1024
    """Number of rays per chunk to use per eval iteration."""
    rendering_scale: float = 0.25
    """Scale factor to rescale renderings before save"""
    roi_only: bool = False
    """Whether to compute metrics only on the region of interest"""
    export_mesh: bool = True
    """Whether to export mesh"""
    mesh_extractor: MeshExtractorConfig = field(default_factory=lambda: MeshExtractorConfig)
    """Config for mesh extractor"""
    export_poses: bool = True
    """Whether to export poses"""
    pose_extractor: PoseExtractorConfig = field(default_factory=lambda: PoseExtractorConfig)
    """Config for pose extractor"""
    extra_modalities_to_exclude: List[Literal["angle_of_polarization", "degree_of_polarization"]] = field(default_factory=list)
    """List of modalities not to consider and export during evaluation"""
    export_aligned_renderings: bool = True
    """Whether to export the aligned renderings"""
    export_extra_renderings: bool = True
    """Whether to export the extra renderings such as the accumulation map"""

@dataclass
class MultisceneEvaluatorConfig(EvaluatorConfig):
    """
    Evaluator config class for multiscene evaluation.
    """
    _target: Type = field(default_factory=lambda: MultisceneEvaluator)
    steps_per_eval_all_scenes: int = 100000
    """Number of steps after which all scenes are evaluated"""
    number_of_scenes_to_eval: int = None
    """Number of scenes to evaluate at each evaluation step. If None, all scenes are evaluated."""
    scene_to_eval_indexes: List[int] = None
    """List of scene indexes to evaluate. If empty, all scenes are evaluated."""

@dataclass
class RawEvaluatorConfig(EvaluatorConfig):
    """Evaluator config class for raw data."""
    _target: Type = field(default_factory=lambda: RawEvaluator)
    export_demosaicked_renderings: bool = True
    """Whether to export the demosaicked renderings"""

@dataclass
class RawMultisceneEvaluatorConfig(MultisceneEvaluatorConfig, RawEvaluatorConfig):
    """Evaluator config class for raw data in multiscene evaluation."""
    _target: Type = field(default_factory=lambda: RawMultisceneEvaluator)

class Evaluator:
    """
    Evaluator class for evaluating the model performance.
    It hanldes the rendering of the model, the computation of the metrics, the generations odf the 3D mesh and the
    export of the optimized camera poses.

    Args:
        config (EvaluatorConfig): Evaluator config class.
        pipeline: Pipeline class.
        scene_box (SceneBox): Scene box class.
        w2gt: World to ground truth transformation matrix.
        output_path (str): Path to save the output.
    """

    def __init__(
            self,
            config: EvaluatorConfig,
            pipeline,
            scene_box: SceneBox,
            w2gt,
            output_path: str,
    ):
        self.config = config
        self.output_path = output_path
        self.pipeline = pipeline
        self.mesh_extractor = self.config.mesh_extractor.setup(
            scene_box=scene_box,
            w2gt=w2gt,
            output_path=output_path,
        )
        self.pose_extractor = self.config.pose_extractor.setup(
            dataset=self.pipeline.datamanager.train_dataset,
            pose_optimizer=self.pipeline.datamanager.train_camera_optimizer,
            w2gt=w2gt,
            output_path=output_path,
        )

    def evaluation_step(self, step: int):
        """Run the evaluation step."""
        losses, total_loss, metrics = None, None, None
        if check_step(step, self.config.steps_per_eval_batch):
            losses, total_loss, metrics = self.render_eval_batch(step)
        if check_step(step, self.config.steps_per_eval_image, skip_first=True):
            self.render_train_view(step)
            self.render_eval_view(step)
        if check_step(step, self.config.steps_per_eval_all_images, skip_first=True):
            self.render_all_eval_views(step)
        if check_step(step, self.config.steps_per_export_mesh, skip_first=True):
            self.export_mesh(step)
        if check_step(step, self.config.steps_per_export_poses, skip_first=False):
            self.export_poses(step)
        return losses, total_loss, metrics

    def single_evaluation_step(self, step: int, view_ids: Tuple[int] = None, output_path: str = None):
        if view_ids is None:
            self.pipeline.evaluator.render_all_eval_views(
                step,
                output_path=output_path
            )
        else:
            self.pipeline.evaluator.render_specific_views(
                step,
                view_ids=view_ids,
                output_path=output_path
            )
        self.export_mesh(step)
        self.export_poses(step)

    def render_eval_batch(self, step: int, **kwargs):
        """Render a batch of rays and compute the losses and metrics."""
        (pixel_coords, pixels) = next(self.pipeline.datamanager.iter_eval_dataloader)
        ray_bundles = self.pipeline.datamanager.eval_ray_generator(pixel_coords)

        with torch.no_grad():
            outputs = self.pipeline.model.module(ray_bundles)

        outputs, pixels, pixel_coords = self.mask_foreground(outputs, pixels, pixel_coords)
        losses, total_loss = self.pipeline.loss_manager.compute_loss(outputs, pixels, pixel_coords, step, eval_step=True)
        metrics = compute_metrics(outputs, pixels, modalities=self.pipeline.datamanager.modalities, eval_step=True)
        return losses, total_loss, metrics

    def render_view(
            self,
            step: int,
            dataset: BaseDataset,
            ray_generator: RayGenerator,
            iter_dataloader = None,
            relative_view_idx: Union[List[int], int] = None,
            view_idx: Union[List[int], int] = None,
            **kwargs,
    ):
        """
        Render a frame given the model and either the dataloader or the view id.

        Args:
            step: Current step.
            dataset: Dataset object.
            ray_generator: Ray generator object.
            iter_dataloader: Iterator over the Dataloader object.
            relative_view_idx: Index of the view id stored in the dataset.
            view_idx: Real view index.

        Returns:
            (
                renderings: Rendered frame per modality.
                side_by_side_renderings: Rendered frame per modality side by side in a single image.
                aligned_renderings: Rendered frame per modality aligned to the first modality.
                geometry_renderings: Normal and depth maps aligned to the first modality.
                extra_renderings: Extra renderings such as the accumulation map.
            )
            pixels: Ground truth pixels.
            view_idx: View index.
        """
        assert iter_dataloader is not None or (relative_view_idx is not None and view_idx is not None)
        if relative_view_idx is None and view_idx is None:
            (pixel_coords, pixels) = next(iter_dataloader)
            if isinstance(dataset, BaseAlignedDataset):
                first_mod = list(self.pipeline.datamanager.modalities.keys())[0]
                relative_view_idx = pixel_coords[first_mod][0, 0]
                view_idx = dataset.indexes[relative_view_idx]
            elif isinstance(dataset, BaseUnalignedDataset):
                relative_view_idx = [
                    pixel_coords[mod][0, 0]
                    if pixel_coords[mod] is not None
                    else None for mod in pixel_coords
                ]
                view_idx = [
                    dataset.indexes[dataset.modalities[i]][relative_view_idx[i]]
                    if relative_view_idx[i] is not None
                    else None
                    for i in range(len(relative_view_idx))
                ]

        ray_bundles = ray_generator(pixel_coords)
        first_valid_mod = next((mod for mod in ray_bundles if ray_bundles[mod] is not None), None)
        c2w = dataset.data[first_valid_mod]['cameras'].get_c2w_matrices(relative_view_idx) \
            if not isinstance(relative_view_idx, list) \
            else dataset.data[first_valid_mod]['cameras'].get_c2w_matrices(
                relative_view_idx[list(self.pipeline.datamanager.modalities.keys()).index(first_valid_mod)]
            )

        renderings, \
            side_by_side_renderings, \
            aligned_renderings, \
            geometry_renderings,\
            extra_renderings = self.generate_eval_renderings(
                                      ray_bundles=ray_bundles,
                                      gt_pixels=pixels,
                                      eval_num_rays_per_chunk=self.config.eval_num_rays_per_chunk,
                                      modalities=self.pipeline.datamanager.modalities,
                                      c2w=c2w,
                                      step=step,
                                      forward_fn=self.pipeline.model,
                                      **kwargs
                                  )
        return (
            renderings,
            side_by_side_renderings,
            aligned_renderings,
            geometry_renderings,
            extra_renderings
        ), pixels, view_idx

    def render_train_view(self, step, **kwargs):
        """Render, compute the metrics and save a frame of the training set."""
        renderings, gt_frames, view_idx = self.render_view(
                                      step=step,
                                      dataset=self.pipeline.datamanager.train_dataset,
                                      iter_dataloader=self.pipeline.datamanager.iter_full_view_train_dataloader,
                                      ray_generator=self.pipeline.datamanager.train_ray_generator,
                                      **kwargs
                                  )

        masks = {
            mod: renderings[-1][f"accumulation_{mod}"] > 0.9
            if renderings[-1][f"accumulation_{mod}"] is not None
            else None
            for mod in self.pipeline.datamanager.modalities
        } if self.config.roi_only else None
        metrics = self.compute_metrics(renderings[0], gt_frames, masks=masks, roi_only=self.config.roi_only)
        writer.put_dict(name="Train Full View Metrics Dict", scalar_dict=metrics, step=step)

        for mod in self.config.extra_modalities_to_exclude:
            renderings[-1].pop(mod, None)

        self.export_rendered_frames(
            *renderings[1:],
            step=step,
            output_path=os.path.join(self.output_path, 'train_renderings'),
            scale=self.config.rendering_scale,
            view_idx=view_idx,
        )

    def render_eval_view(self, step, **kwargs):
        """Render, compute the metrics and save a frame of the evaluation set."""
        renderings, gt_frames, view_idx = self.render_view(
            step=step,
            dataset=self.pipeline.datamanager.eval_dataset,
            iter_dataloader=self.pipeline.datamanager.iter_full_view_eval_dataloader,
            ray_generator=self.pipeline.datamanager.eval_ray_generator,
            **kwargs
        )

        masks = {
            mod: renderings[-1][f"accumulation_{mod}"] > 0.9
            if renderings[-1][f"accumulation_{mod}"] is not None
            else None
            for mod in self.pipeline.datamanager.modalities
        } if self.config.roi_only else None
        metrics = self.compute_metrics(renderings[0], gt_frames, masks=masks, roi_only=self.config.roi_only)
        writer.put_dict(name="Eval Full View Metrics Dict", scalar_dict=metrics, step=step)

        for mod in self.config.extra_modalities_to_exclude:
            renderings[-1].pop(mod, None)

        self.export_rendered_frames(
            *renderings[1:],
            step=step,
            output_path=os.path.join(self.output_path, 'eval_renderings'),
            scale=self.config.rendering_scale,
            view_idx=view_idx,
        )

    def render_specific_views(self, step, view_ids, output_path=None, export_metrics=True, **kwargs):
        """Render, compute the metrics and save all the frames specified in view_ids."""
        output_path = output_path if output_path is not None else os.path.join(self.output_path, 'validation')
        all_metrics = {mod: defaultdict(list) for mod in self.pipeline.datamanager.modalities}
        all_metrics['idx'] = []
        n_eval_views = len(self.pipeline.datamanager.full_view_eval_dataloader.selected_views)
        n_train_views = len(self.pipeline.datamanager.full_view_train_dataloader.selected_views)
        for i in range(n_eval_views + n_train_views):
            if i < n_eval_views:
                dataset = self.pipeline.datamanager.eval_dataset
                iter_dataloader = self.pipeline.datamanager.iter_full_view_eval_dataloader
                ray_generator = self.pipeline.datamanager.eval_ray_generator
            elif i - n_eval_views < n_train_views:
                dataset = self.pipeline.datamanager.train_dataset
                iter_dataloader = self.pipeline.datamanager.iter_full_view_train_dataloader
                ray_generator = self.pipeline.datamanager.train_ray_generator
            else:
                raise ValueError("View index out of range.")

            renderings, gt_frames, view_idx = self.render_view(
                step=step,
                dataset=dataset,
                iter_dataloader=iter_dataloader,
                ray_generator=ray_generator,
                **kwargs
            )

            masks = {
                mod: renderings[-1][f"accumulation_{mod}"] > 0.9
                if renderings[-1][f"accumulation_{mod}"] is not None
                else None
                for mod in self.pipeline.datamanager.modalities
            } if self.config.roi_only else None
            metrics = self.compute_metrics(renderings[0], gt_frames, masks=masks, roi_only=self.config.roi_only)
            all_metrics['idx'].append(view_idx)
            for mod, mod_metrics in metrics.items():
                for metric_name, metric in mod_metrics.items():
                    all_metrics[mod][metric_name].append(metric)

        for mod in self.config.extra_modalities_to_exclude:
            renderings[-1].pop(mod, None)

        self.export_rendered_frames(
            *renderings[1:],
            step=step,
            output_path=output_path,
            scale=1.0,
            view_idx=view_idx,
            single_channels=True,
        )

        if n_eval_views + n_train_views > len(view_ids):
            all_metrics = self.merge_metrics(all_metrics)
        if export_metrics:
            export_metrics_to_file(all_metrics, os.path.join(output_path, 'results.txt'), step)
        return all_metrics

    def render_all_eval_views(self, step, output_path=None, export_metrics=True, **kwargs):
        """Render, compute the metrics and save all the frames of the evaluation set."""
        output_path = output_path if output_path is not None else os.path.join(self.output_path, 'validation')
        all_metrics = {mod: defaultdict(list) for mod in self.pipeline.datamanager.modalities}
        all_metrics['idx'] = []
        for _ in range(len(self.pipeline.datamanager.eval_dataset.get_unique_views())):
            renderings, gt_frames, view_idx = self.render_view(
                step=step,
                dataset=self.pipeline.datamanager.eval_dataset,
                iter_dataloader=self.pipeline.datamanager.iter_full_view_eval_dataloader,
                ray_generator=self.pipeline.datamanager.eval_ray_generator,
                **kwargs
            )

            masks = {
                mod: renderings[-1][f"accumulation_{mod}"] > 0.9
                if renderings[-1][f"accumulation_{mod}"] is not None
                else None
                for mod in self.pipeline.datamanager.modalities
            } if self.config.roi_only else None
            metrics = self.compute_metrics(renderings[0], gt_frames, masks=masks, roi_only=self.config.roi_only)
            all_metrics['idx'].append(view_idx)
            for mod, mod_metrics in metrics.items():
                for metric_name, metric in mod_metrics.items():
                    all_metrics[mod][metric_name].append(metric)

            for mod in self.config.extra_modalities_to_exclude:
                renderings[-1].pop(mod, None)

            self.export_rendered_frames(
                *renderings[1:],
                step=step,
                output_path=output_path,
                scale=1.0,
                view_idx=view_idx,
            )

        if export_metrics:
            export_metrics_to_file(all_metrics, os.path.join(output_path, 'results.txt'), step)
        return all_metrics

    def generate_eval_renderings(
            self,
            ray_bundles,
            gt_pixels,
            eval_num_rays_per_chunk,
            modalities,
            c2w,
            step,
            forward_fn,
            process_fn=None,
            outputs=None,
            **kwargs
    ):
        """
        Generate the renderings for the given ray bundles.

        Args:
            ray_bundles: Ray bundles to render.
            gt_pixels: Ground truth pixels.
            eval_num_rays_per_chunk: Number of rays per chunk to use during the rendering.
            modalities: List of modalities.
            c2w: Camera to world transformation matrix.
            step: Current step.
            forward_fn: Forward function for the model.
            process_fn: Function to process the outputs.
            outputs: Outputs from the model. If provided, it will be used instead of calling the model again.

        Returns:
            renderings: Rendered frames per modality.
            side_by_side_renderings: Rendered frames per modality side by side in a single image.
            aligned_renderings: Rendered frames per modality aligned to the first modality.
            geometry_renderings: Normal and depth maps aligned to the first modality.
            extra_renderings: Extra renderings such as the accumulation map.
        """

        gt_pixels = get_dict_to_cpu(gt_pixels)

        if outputs is None:
            with TimeWriter(writer, EventName.TEST_RAYS_PER_SEC, write=False) as test_t:
                outputs = eval_model_query(
                    ray_bundles=ray_bundles,
                    num_rays_per_chunk=eval_num_rays_per_chunk,
                    model_fn=lambda x: forward_fn(x, **kwargs),
                    step=step,
                    key_to_exclude=self.config.extra_modalities_to_exclude,
                )
            writer.put_time(
                name=EventName.TEST_RAYS_PER_SEC,
                duration=sum([len(x) if x is not None else 0 for x in ray_bundles.values()]) / test_t.duration,
                step=step,
                avg_over_steps=True,
            )

        if process_fn is not None:
            outputs = process_fn(outputs)

        renderings, \
            aligned_renderings, \
            geometry_renderings,\
            extra_renderings = render_outputs(
            outputs=outputs,
            modalities=modalities,
            gt_frames=gt_pixels,
            c2w=c2w,
        )

        if "polarization" in self.pipeline.datamanager.modalities and ray_bundles["polarization"] is not None:
            extra_renderings["degree_of_polarization"] = polarizer.to_dop(data=aligned_renderings["polarization"])
            extra_renderings["angle_of_polarization"] = polarizer.to_aop(data=aligned_renderings["polarization"]) / np.pi

        side_by_side_renderings, aligned_renderings, geometry_renderings = combine_renderings(
            renderings=renderings,
            aligned_renderings=aligned_renderings,
            geometry_renderings=geometry_renderings,
            gt_frames=gt_pixels,
        )

        for mod in self.pipeline.datamanager.modalities:
            if len(extra_renderings[f"accumulation_{mod}"]) == 0:
                extra_renderings[f"accumulation_{mod}"] = None

        return renderings, side_by_side_renderings, aligned_renderings, geometry_renderings, extra_renderings

    def export_rendered_frames(
            self,
            side_by_side_renderings,
            aligned_renderings,
            geometry_renderings,
            extra_renderings,
            step,
            output_path,
            scale,
            view_idx,
            **kwargs,
    ):
        """Save to disk the rendered frames."""
        export_renderings(
            renderings=side_by_side_renderings,
            export_path=os.path.join(output_path, 'radiance_renderings', "validation"),
            step=step,
            view_idx=view_idx,
            scale=scale
        )
        export_renderings(
            renderings=geometry_renderings,
            export_path=os.path.join(output_path, 'geometry_renderings'),
            step=step,
            view_idx=view_idx,
            scale=scale
        )
        if self.config.export_aligned_renderings:
            export_renderings(
                renderings=aligned_renderings,
                export_path=os.path.join(output_path, 'radiance_renderings', "aligned"),
                step=step,
                view_idx=view_idx,
                scale=scale
            )
        if self.config.export_extra_renderings:
            export_renderings(
                renderings=extra_renderings,
                export_path=os.path.join(output_path, 'extra_renderings'),
                step=step,
                view_idx=view_idx,
                scale=scale
            )

    def compute_metrics(self, renderings, gt_frames, masks=None, roi_only=False):
        """Compute the quality metrics for the rendered frames with respect to the GT frames."""
        all_metrics = {}
        for mod in renderings:
            rendering = renderings[mod]
            gt = gt_frames[mod]
            mask = masks[mod] if masks is not None and roi_only else None
            metrics = compute_metrics(rendering, gt, mask=mask)
            all_metrics[mod] = metrics
        return all_metrics

    def merge_metrics(self, all_metrics):
        """Merge and order the metrics computed on views belonging to the training and evaluation sets."""
        unique_ids = list({
            next((x for i, x in enumerate(idx) if x is not None), None)
            for idx in all_metrics['idx']
        })
        unique_ids.sort()
        modalities = self.pipeline.datamanager.modalities
        merged_metrics = defaultdict(dict)
        for i, mod in enumerate(modalities):
            merged_metrics[mod] = defaultdict(list)
            for metr in all_metrics[mod].keys():
                sorting_list = []
                for k, idx in enumerate(all_metrics['idx']):
                    for j, x in enumerate(idx):
                        if x is None:
                            continue
                        if i == j:
                            merged_metrics[mod][metr].append(all_metrics[mod][metr][k])
                            sorting_list.append(unique_ids.index(x))
                sorting_list = sorted(range(len(sorting_list)), key=lambda k: sorting_list[k])
                merged_metrics[mod][metr] = [merged_metrics[mod][metr][l] for l in sorting_list]
        merged_metrics['idx'] = unique_ids
        return merged_metrics

    def export_mesh(self, step, **kwargs):
        """Generate and save the 3D mesh"""
        if self.config.export_mesh:
            with torch.no_grad():
                self.mesh_extractor.extract(
                    sdf_fn=lambda x: self.pipeline.model.surface_model.surface_field.single_output(x, **kwargs),
                    step=step,
                )

    def export_poses(self, step):
        """Export the optimized camera poses"""
        if self.config.export_poses:
            with torch.no_grad():
                self.pose_extractor.extract(
                    step=step,
                )

    def update_w2gt(self, w2gt):
        """Update the world to ground truth transformation matrix."""
        self.mesh_extractor.update_w2gt(w2gt)
        self.pose_extractor = self.config.pose_extractor.setup(
            dataset=self.pipeline.datamanager.train_dataset,
            pose_optimizer=self.pipeline.datamanager.train_camera_optimizer,
            w2gt=w2gt,
            output_path=self.output_path,
        )

    def update_scene_box(self, scene_box: SceneBox):
        """Update the scene box."""
        self.mesh_extractor.update_scene_box(scene_box)

    def mask_foreground(self, outputs, pixels, pixel_coords):
        """Mask the foreground pixels based on the accumulation map."""
        modalities = list(self.pipeline.datamanager.modalities.keys())
        for mod in outputs[modalities[0]]:
            if "latent" in mod:
                modalities.append(mod)
        for mod_view in self.pipeline.datamanager.modalities:
            mask = outputs[mod_view]["accumulation"].squeeze() > 0.9
            pixels[mod_view] = pixels[mod_view][mask]
            pixel_coords[mod_view] = pixel_coords[mod_view][mask]
            for mod_pixel in outputs[mod_view]:
                outputs[mod_view][mod_pixel] = outputs[mod_view][mod_pixel][mask]
        return outputs, pixels, pixel_coords

class RawEvaluator(Evaluator):
    """Evaluator for raw data"""

    def __init__(
            self,
            config: RawEvaluatorConfig,
            pipeline,
            scene_box: SceneBox,
            w2gt,
            output_path: str,
    ):
        super().__init__(config, pipeline, scene_box, w2gt, output_path)
        self.config = config
        self.mosaick_mask_per_modality = get_dict_to_cpu(
            copy.deepcopy(self.pipeline.datamanager.train_dataset.mosaick_mask_per_modality)
        )
        self.mosaick_mask_across_modalities = get_dict_to_cpu(
            copy.deepcopy(self.pipeline.datamanager.train_dataset.mosaick_mask_across_modalities)
        )

    def render_view(
            self,
            step: int,
            dataset: BaseDataset,
            ray_generator: RayGenerator,
            iter_dataloader = None,
            relative_view_idx: Union[List[int], int] = None,
            view_idx: Union[List[int], int] = None,
            **kwargs,
    ):
        """
        Render a frame given the model and either the dataloader or the view id.

        Args:
            step: Current step.
            dataset: Dataset object.
            ray_generator: Ray generator object.
            iter_dataloader: Iterator over the Dataloader object.
            relative_view_idx: Index of the view id stored in the dataset.
            view_idx: Real view index.

        Returns:
            (
                mosaicked_renderings: Rendered raw frame per modality.
                demosaicked_renderings: Rendered demosaicked frame per modality.
                side_by_side_renderings: Rendered frame per modality side by side in a single image.
                aligned_renderings: Rendered frame per modality aligned to the first modality.
                geometry_renderings: Normal and depth maps aligned to the first modality.
                extra_renderings: Extra renderings such as the accumulation map.
            )
            pixels: Ground truth pixels.
            view_idx: View index.
        """
        assert iter_dataloader is not None or (relative_view_idx is not None and view_idx is not None)
        if relative_view_idx is None and view_idx is None:
            (pixel_coords, pixels) = next(iter_dataloader)
            if isinstance(dataset, BaseAlignedDataset):
                first_mod = list(self.pipeline.datamanager.modalities.keys())[0]
                relative_view_idx = pixel_coords[first_mod][0, 0]
                view_idx = dataset.indexes[relative_view_idx]
            elif isinstance(dataset, BaseUnalignedDataset):
                relative_view_idx = [
                    pixel_coords[mod][0, 0]
                    if pixel_coords[mod] is not None
                    else None
                    for mod in pixel_coords
                ]
                view_idx = [
                    dataset.indexes[dataset.modalities[i]][relative_view_idx[i]]
                    if relative_view_idx[i] is not None
                    else None
                    for i in range(len(relative_view_idx))
                ]

        ray_bundles = ray_generator(pixel_coords)
        first_valid_mod = next((mod for mod in ray_bundles if ray_bundles[mod] is not None), None)
        c2w = dataset.data[first_valid_mod]['cameras'].get_c2w_matrices(relative_view_idx) \
            if not isinstance(relative_view_idx, list) \
            else dataset.data[first_valid_mod]['cameras'].get_c2w_matrices(
                relative_view_idx[list(self.pipeline.datamanager.modalities.keys()).index(first_valid_mod)]
            )

        demosaicked_renderings, \
            mosaicked_renderings, \
            side_by_side_renderings, \
            aligned_renderings, \
            geometry_renderings,\
            extra_renderings = self.generate_eval_renderings(
                                      ray_bundles=ray_bundles,
                                      pixel_coords_per_modality=pixel_coords,
                                      gt_pixels=pixels,
                                      eval_num_rays_per_chunk=self.config.eval_num_rays_per_chunk,
                                      modalities=self.pipeline.datamanager.modalities,
                                      c2w=c2w,
                                      step=step,
                                      forward_fn=lambda x: self.pipeline.model(x, **kwargs),
                                  )
        return (
            mosaicked_renderings,
            demosaicked_renderings,
            side_by_side_renderings,
            aligned_renderings,
            geometry_renderings,
            extra_renderings
        ), pixels, view_idx

    def generate_eval_renderings(
            self,
            ray_bundles,
            pixel_coords_per_modality,
            gt_pixels,
            eval_num_rays_per_chunk,
            modalities,
            c2w,
            step,
            forward_fn,
            process_fn=None,
            outputs=None,
    ):
        """
        Generate the renderings for the given ray bundles.

        Args:
            ray_bundles: Ray bundles to render.
            gt_pixels: Ground truth pixels.
            eval_num_rays_per_chunk: Number of rays per chunk to use during the rendering.
            modalities: List of modalities.
            c2w: Camera to world transformation matrix.
            step: Current step.
            forward_fn: Forward function for the model.
            process_fn: Function to process the outputs.
            outputs: Outputs from the model. If provided, it will be used instead of calling the model again.

        Returns:
            renderings: Rendered frames per modality, demosaicked.
            mosaicked_renderings: Rendered raw frames per modality.
            side_by_side_renderings: Rendered frames per modality side by side in a single image.
            aligned_renderings: Rendered frames per modality aligned to the first modality.
            geometry_renderings: Normal and depth maps aligned to the first modality.
            extra_renderings: Extra renderings such as the accumulation map.
        """

        pixel_coords_per_modality = get_dict_to_cpu(pixel_coords_per_modality)
        gt_pixels = get_dict_to_cpu(gt_pixels)

        if outputs is None:
            with TimeWriter(writer, EventName.TEST_RAYS_PER_SEC, write=False) as test_t:
                outputs = eval_model_query(
                    ray_bundles=ray_bundles,
                    num_rays_per_chunk=eval_num_rays_per_chunk,
                    model_fn=forward_fn,
                    step=step,
                    key_to_exclude=self.config.extra_modalities_to_exclude,
                )
            writer.put_time(
                name=EventName.TEST_RAYS_PER_SEC,
                duration=sum([len(x) if x is not None else 0 for x in ray_bundles.values()]) / test_t.duration,
                step=step,
                avg_over_steps=True,
            )

        if process_fn is not None:
            outputs = process_fn(outputs)

        renderings, \
            aligned_renderings, \
            geometry_renderings,\
            extra_renderings = render_outputs(
            outputs=outputs,
            modalities=modalities,
            gt_frames=gt_pixels,
            c2w=c2w,
        )

        mosaick_mask_per_modality = self.mosaick_mask_per_modality
        mosaicked_renderings = self.select_right_channel_per_rendered_pixel(
            pixel_coords_per_modality=pixel_coords_per_modality,
            renderings=renderings,
            mosaick_mask_per_modality=mosaick_mask_per_modality
        )
        first_valid_mod = next((mod for mod in ray_bundles if ray_bundles[mod] is not None), None)
        mosaick_mask_per_modality = {
            mod: self.mosaick_mask_across_modalities[first_valid_mod][mod]
            for mod in self.pipeline.datamanager.modalities
        }
        pixel_coords_per_aligned_modality = {
            mod: pixel_coords_per_modality[first_valid_mod]
            for mod in self.pipeline.datamanager.modalities
        }

        if "polarization" in self.pipeline.datamanager.modalities and ray_bundles["polarization"] is not None:
            if "polarization_coefficients" in extra_renderings:
                extra_renderings["polarization_coefficients"] = self.select_right_channel_per_rendered_pixel(
                    pixel_coords_per_modality[0],
                    extra_renderings["polarization_coefficients"],
                    mosaick_mask_per_modality["polarization"]
                )
            extra_renderings["degree_of_polarization"] = polarizer.to_dop(data=aligned_renderings["polarization"])
            extra_renderings["angle_of_polarization"] = polarizer.to_aop(data=aligned_renderings["polarization"]) / np.pi

        aligned_renderings = self.select_right_channel_per_rendered_pixel(
            pixel_coords_per_modality=pixel_coords_per_aligned_modality,
            renderings=aligned_renderings,
            mosaick_mask_per_modality=mosaick_mask_per_modality
        )

        side_by_side_renderings, aligned_renderings, geometry_renderings = combine_renderings(
            renderings=mosaicked_renderings,
            aligned_renderings=aligned_renderings,
            geometry_renderings=geometry_renderings,
            gt_frames=gt_pixels,
        )

        for mod in self.pipeline.datamanager.modalities:
            if len(extra_renderings[f"accumulation_{mod}"]) == 0:
                extra_renderings[f"accumulation_{mod}"] = None

        return renderings, mosaicked_renderings, side_by_side_renderings, aligned_renderings, geometry_renderings, extra_renderings

    def select_right_channel_per_rendered_pixel(self, pixel_coords_per_modality, renderings, mosaick_mask_per_modality):
        """
        Select the right channel per pixel for the rendered frames, according to the mosaick mask.
        """
        if isinstance(renderings, dict):
            mosaicked_renderings = {}
            for mod in self.pipeline.datamanager.modalities:
                if renderings[mod] is None:
                    mosaicked_renderings[mod] = None
                    continue
                frame = renderings[mod]
                mosaick_mask = mosaick_mask_per_modality[mod]
                pixel_coords = pixel_coords_per_modality[mod]
                band_mask = mosaick_mask[
                    pixel_coords[:, 1],
                    pixel_coords[:, 2]
                ].view((*frame.shape[:-1], 1)).type(torch.int64)
                frame = torch.gather(frame, 2, band_mask)
                mosaicked_renderings[mod] = frame
        else:
            band_mask = mosaick_mask_per_modality[
                pixel_coords_per_modality[:, 1],
                pixel_coords_per_modality[:, 2]
            ].view((*renderings.shape[:-1], 1)).type(torch.int64)
            mosaicked_renderings = torch.gather(renderings, 2, band_mask)
        return mosaicked_renderings

    def export_rendered_frames(
            self,
            demosaicked_renderings,
            side_by_side_renderings,
            aligned_renderings,
            geometry_renderings,
            extra_renderings,
            step,
            output_path,
            scale,
            view_idx,
            single_channels=False,
    ):
        """Save the rendered frames to disk."""
        export_renderings(
            renderings=side_by_side_renderings,
            export_path=os.path.join(output_path, 'radiance_renderings', "validation"),
            step=step,
            view_idx=view_idx,
            scale=scale
        )
        export_renderings(
            renderings=geometry_renderings,
            export_path=os.path.join(output_path, 'geometry_renderings'),
            step=step,
            view_idx=view_idx,
            scale=scale
        )
        if self.config.export_demosaicked_renderings:
            export_renderings(
                renderings=demosaicked_renderings,
                export_path=os.path.join(output_path, 'radiance_renderings', "demosaicked"),
                step=step,
                view_idx=view_idx,
                scale=scale,
                single_channels=single_channels,
            )
        if self.config.export_aligned_renderings:
            export_renderings(
                renderings=aligned_renderings,
                export_path=os.path.join(output_path, 'radiance_renderings', "aligned"),
                step=step,
                view_idx=view_idx,
                scale=scale
            )
        if self.config.export_extra_renderings:
            export_renderings(
                renderings=extra_renderings,
                export_path=os.path.join(output_path, 'extra_renderings'),
                step=step,
                view_idx=view_idx,
                scale=scale
            )

class MultisceneEvaluator(Evaluator):
    """Evaluator for multiscene data"""

    def __init__(
            self,
            config: MultisceneEvaluatorConfig,
            pipeline,
            scene_box: SceneBox,
            w2gt,
            output_path: str,
    ):
        """
        Initialize the evaluator.

        Args:
            config: Configuration for the evaluator.
            pipeline: Pipeline object.
            scene_box: SceneBox object.
            w2gt: World to ground truth transformation matrix.
            output_path: Path to save the evaluation results.
        """
        super().__init__(config, pipeline, scene_box, w2gt, output_path)
        self.config = config

    def evaluation_step(self, step: int):
        """Run the evaluation step."""
        losses, total_loss, metrics = None, None, None
        scene_index = self.pipeline.multiscene_datamanager.get_current_scene_idx(step=step)
        if check_step(step, self.config.steps_per_eval_batch):
            losses, total_loss, metrics = self.render_eval_batch(step, scene_idx=scene_index)
        if check_step(step, self.config.steps_per_eval_image, skip_first=True):
            self.render_train_view(step, scene_idx=scene_index)
            self.render_eval_view(step, scene_idx=scene_index)
        if check_step(step, self.config.steps_per_eval_all_scenes, skip_first=True):
            self.render_all_eval_scenes(step)
        if check_step(step, self.config.steps_per_export_mesh, skip_first=True):
            self.export_mesh(step, scene_idx=scene_index)
        if check_step(step, self.config.steps_per_export_poses, skip_first=False):
            self.export_poses(step)
        return losses, total_loss, metrics

    def single_evaluation_step(self, step: int, view_ids: Tuple[int] = None, output_path: str = None):
        self.render_all_eval_scenes(step, view_ids=view_ids, output_path=output_path)

    def render_eval_batch(self, step: int, scene_idx: int):
        """Render a batch of rays and compute the losses and metrics."""
        (pixel_coords, pixels) = next(self.pipeline.multiscene_datamanager.current_datamanager().iter_eval_dataloader)
        ray_bundles = self.pipeline.multiscene_datamanager.current_datamanager().eval_ray_generator(pixel_coords)
        with torch.no_grad():
            outputs = self.pipeline.model.module(ray_bundles, scene_idx=scene_idx)
        outputs, pixels, pixel_coords = self.mask_foreground(outputs, pixels, pixel_coords)
        losses, total_loss = self.pipeline.loss_manager.compute_loss(outputs, pixels, pixel_coords, step, eval_step=True)
        metrics = compute_metrics(outputs, pixels, modalities=self.pipeline.multiscene_datamanager.modalities, eval_step=True)
        return losses, total_loss, metrics

    def render_all_eval_scenes(self, step: int, view_ids: Tuple[int] = None, output_path: str = None):
        """
        Render, compute the metrics and save all the frames of the evaluation set for all scenes.
        """
        base_output_path = output_path if output_path is not None else os.path.join(self.output_path, 'validation')
        if self.config.scene_to_eval_indexes is None and self.config.number_of_scenes_to_eval is None:
            raise ValueError("Either scene_to_eval_indexes or number_of_scenes_to_eval must be set in the config.")

        current_scene_index = self.pipeline.multiscene_datamanager.get_current_scene_idx(step=step)
        if self.config.scene_to_eval_indexes is not None:
            scene_list = [current_scene_index] + [
                self.pipeline.multiscene_datamanager.scene_indexes.index(scene_index)
                for scene_index in self.config.scene_to_eval_indexes
            ]
        else:
            scene_list = [current_scene_index] + random.sample(
                range(len(self.pipeline.multiscene_datamanager.scene_indexes)),
                self.config.number_of_scenes_to_eval
            )

        all_scene_metrics = []
        for i in tqdm(range(len(scene_list) - 1), desc="Evaluating eval scenes"):
            current_scene_index = scene_list[i]
            eval_scene_index = scene_list[i + 1]
            eval_scene_name = self.pipeline.multiscene_datamanager.scene_names[eval_scene_index]
            self.pipeline.switch_scene_by_index(current_scene_index, eval_scene_index)
            if view_ids is None:
                scene_metrics = self.render_all_eval_views(
                    step=step,
                    output_path=os.path.join(base_output_path, f'{eval_scene_name}'),
                    scene_idx=eval_scene_index,
                    export_metrics=False
                )
            else:
                scene_metrics = self.render_specific_views(
                    step=step,
                    view_ids=view_ids,
                    output_path=os.path.join(base_output_path, f'{eval_scene_name}'),
                    scene_idx=eval_scene_index,
                    export_metrics=False
                )
            all_scene_metrics.append(scene_metrics)

        self.export_all_scene_metrics(
            step,
            all_scene_metrics,
            scene_list[1:],
            os.path.join(base_output_path, 'results.txt')
        )

        current_scene_index = self.pipeline.multiscene_datamanager.get_current_scene_idx(step=step)
        self.pipeline.switch_scene_by_index(eval_scene_index, current_scene_index)
        pass

    def export_all_scene_metrics(self, step, all_scene_metrics, scene_indexes, output_path):
        """
        Export metrics of all scenes to a txt file.
        """
        # Extract modalities and metric names
        modalities = [k for k in all_scene_metrics[0] if k != 'idx']
        metric_names = list(all_scene_metrics[0][modalities[0]].keys())
        eval_view_idxs = all_scene_metrics[0]['idx']
        n_views = len(eval_view_idxs)

        # Build per-view, per-modality, per-metric value lists for correct averaging
        avg_metrics = {mod: {metr: [] for metr in metric_names} for mod in modalities}
        avg_metrics_per_mod = {mod: {metr: None for metr in metric_names} for mod in modalities}
        for mod in modalities:
            for metr in metric_names:
                vals_per_view = []
                for v in range(n_views):
                    vals = []
                    for scene in all_scene_metrics:
                        vals.append(scene[mod][metr][v])
                    avg = sum(val for val in vals if not isinstance(val, str)) / \
                          sum(1 for val in vals if not isinstance(val, str))
                    vals_per_view.append(avg)
                avg_metrics[mod][metr] = vals_per_view
                # Average over all views
                avg_metrics_per_mod[mod][metr] = sum(avg_metrics[mod][metr]) / n_views if n_views > 0 else None

        # Formatting for aligned columns (left-aligned)
        max_mod_len = max(len(mod) for mod in modalities)
        mod_col_width = max_mod_len + 2
        view_col_width = 10
        avg_col_title = "AVG"
        avg_col_width = 10

        def write_header(f, view_idxs, metric_names):
            # First header row: empty for mod, then view idx under first metric, empty for others, then AVG
            header = ["".ljust(mod_col_width)]
            for idx in view_idxs:
                for i, metr in enumerate(metric_names):
                    if i == 0:
                        header.append(str(idx).ljust(view_col_width))
                    else:
                        header.append("".ljust(view_col_width))
            for i, metr in enumerate(metric_names):
                if i == 0:
                    header.append(avg_col_title.ljust(avg_col_width))
                else:
                    header.append("".ljust(avg_col_width))
            f.write("".join(header) + "\n")
            # Second header row: metric names under each view, then metric names under AVG
            header = ["".ljust(mod_col_width)]
            for _ in view_idxs:
                for metr in metric_names:
                    header.append(metr.ljust(view_col_width))
            for metr in metric_names:
                header.append(metr.ljust(avg_col_width))
            f.write("".join(header) + "\n")

        def write_metrics_row(f, mod, values_dict, avg_dict, is_first):
            row = []
            if is_first:
                row.append(mod.upper().ljust(mod_col_width))
            else:
                row.append("".ljust(mod_col_width))
            for v in range(n_views):
                for metr in metric_names:
                    val = values_dict[metr][v]
                    if isinstance(val, str):
                        row.append("-".ljust(view_col_width))
                    else:
                        row.append(f"{val:.3f}".ljust(view_col_width))
            for metr in metric_names:
                val = avg_dict[metr]
                if isinstance(val, str) or val is None:
                    row.append("-".ljust(avg_col_width))
                else:
                    row.append(f"{val:.3f}".ljust(avg_col_width))
            f.write("".join(row) + "\n")

        open(output_path, 'a').close()
        with open(output_path, 'r+') as f:
            content = f.read()
            f.seek(0, 0)

            f.write(f"Step: {step}\n")
            f.write("Average metrics per eval view across all scenes\n")
            write_header(f, eval_view_idxs, metric_names)
            for mod in modalities:
                write_metrics_row(f, mod, avg_metrics[mod], avg_metrics_per_mod[mod], is_first=True)

            # Per-scene metrics
            f.write("\nMetrics per eval view for each scene\n")
            for i, scene_dict in enumerate(all_scene_metrics):
                scene_name = self.pipeline.multiscene_datamanager.scene_names[scene_indexes[i]]
                f.write(f"{scene_name.upper()}\n")
                write_header(f, scene_dict['idx'], metric_names)
                for mod in modalities:
                    values_dict = {metr: [] for metr in metric_names}
                    avg_dict = {}
                    for metr in metric_names:
                        values = []
                        for v in range(n_views):
                            values.append(scene_dict[mod][metr][v])
                        values_dict[metr] = values
                        # Average for this metric in this scene
                        avg_dict[metr] = sum(val for val in values if not isinstance(val, str)) / \
                                         sum(1 for val in values if not isinstance(val, str)) if n_views > 0 else None
                    write_metrics_row(f, mod, values_dict, avg_dict, is_first=True)
                f.write("\n")
            f.write(content + "\n")

class RawMultisceneEvaluator(MultisceneEvaluator, RawEvaluator):
    """Evaluator for raw multiscene data"""

    def __init__(
            self,
            config: RawMultisceneEvaluatorConfig,
            pipeline,
            scene_box: SceneBox,
            w2gt,
            output_path: str,
    ):
        """
        Initialize the evaluator.

        Args:
            config: Configuration for the evaluator.
            pipeline: Pipeline object.
            scene_box: SceneBox object.
            w2gt: World to ground truth transformation matrix.
            output_path: Path to save the evaluation results.
        """
        super().__init__(config, pipeline, scene_box, w2gt, output_path)
        self.config = config
