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

import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Type, Union, List

import numpy as np
import torch

from configs.configs import InstantiateConfig
from data.datasets import BaseDataset, BaseAlignedDataset, BaseUnalignedDataset
from data.scene_box import SceneBox

from evaluator_components.mesh_extractors import MeshExtractorConfig
from evaluator_components.pose_extractor import PoseExtractorConfig
from model_components import polarizer
from model_components.ray_generators import RayGenerator
from utils import writer
from utils.eval_utils import export_renderings, compute_metrics, eval_model_query, render_outputs, combine_renderings

@dataclass
class EvaluatorConfig(InstantiateConfig):
    """
    Evaluator config class.
    """
    _target: Type = field(default_factory=lambda: Evaluator)
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

@dataclass
class RawEvaluatorConfig(EvaluatorConfig):
    """Evaluator config class for raw data."""
    _target: Type = field(default_factory=lambda: RawEvaluator)

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
        self.model = self.pipeline.model.module
        self.datamanager = self.pipeline.datamanager
        self.mesh_extractor = self.config.mesh_extractor.setup(
            scene_box=scene_box,
            w2gt=w2gt,
            output_path=output_path,
        )
        self.pose_extractor = self.config.pose_extractor.setup(
            dataset=self.datamanager.train_dataset,
            pose_optimizer=self.datamanager.train_camera_optimizer,
            w2gt=w2gt,
            output_path=output_path,
        )

    def render_view(
            self,
            step: int,
            dataset: BaseDataset,
            ray_generator: RayGenerator,
            iter_dataloader = None,
            relative_view_idx: Union[List[int], int] = None,
            view_idx: Union[List[int], int] = None,
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
                first_mod = list(self.datamanager.modalities.keys())[0]
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
                relative_view_idx[self.datamanager.modalities.index(first_valid_mod)]
            )

        renderings, \
            side_by_side_renderings, \
            aligned_renderings, \
            geometry_renderings,\
            extra_renderings = self.generate_eval_renderings(
                                      ray_bundles=ray_bundles,
                                      gt_pixels=pixels,
                                      eval_num_rays_per_chunk=self.config.eval_num_rays_per_chunk,
                                      modalities=self.datamanager.modalities,
                                      c2w=c2w,
                                      step=step,
                                      forward_fn=self.model,
                                  )
        return (
            renderings,
            side_by_side_renderings,
            aligned_renderings,
            geometry_renderings,
            extra_renderings
        ), pixels, view_idx

    def render_train_view(self, step):
        """Render, compute the metrics and save a frame of the training set."""
        renderings, gt_frames, view_idx = self.render_view(
                                      step=step,
                                      dataset=self.datamanager.train_dataset,
                                      iter_dataloader=self.datamanager.iter_full_view_train_dataloader,
                                      ray_generator=self.datamanager.train_ray_generator,
                                  )

        masks = {
            mod: renderings[-1][f"accumulation_{mod}"] > 0.9
            if renderings[-1][f"accumulation_{mod}"] is not None
            else None
            for mod in self.datamanager.modalities
        } if self.config.roi_only else None
        metrics = self.compute_metrics(renderings[0], gt_frames, masks=masks, roi_only=self.config.roi_only)
        writer.put_dict(name="Train Full View Metrics Dict", scalar_dict=metrics, step=step)

        self.export_rendered_frames(
            *renderings[1:],
            step=step,
            output_path=os.path.join(self.output_path, 'train_renderings'),
            scale=self.config.rendering_scale,
            view_idx=view_idx,
        )

    def render_eval_view(self, step):
        """Render, compute the metrics and save a frame of the evaluation set."""
        renderings, gt_frames, view_idx = self.render_view(
            step=step,
            dataset=self.datamanager.eval_dataset,
            iter_dataloader=self.datamanager.iter_full_view_eval_dataloader,
            ray_generator=self.datamanager.eval_ray_generator,
        )

        masks = {
            mod: renderings[-1][f"accumulation_{mod}"] > 0.9
            if renderings[-1][f"accumulation_{mod}"] is not None
            else None
            for mod in self.datamanager.modalities
        } if self.config.roi_only else None
        metrics = self.compute_metrics(renderings[0], gt_frames, masks=masks, roi_only=self.config.roi_only)
        writer.put_dict(name="Eval Full View Metrics Dict", scalar_dict=metrics, step=step)

        self.export_rendered_frames(
            *renderings[1:],
            step=step,
            output_path=os.path.join(self.output_path, 'eval_renderings'),
            scale=self.config.rendering_scale,
            view_idx=view_idx,
        )

    def render_specific_views(self, step, view_ids, output_path=None):
        """Render, compute the metrics and save all the frames specified in view_ids."""
        output_path = output_path if output_path is not None else os.path.join(self.output_path, 'validation')
        all_metrics = {mod: defaultdict(list) for mod in self.datamanager.modalities}
        all_metrics['idx'] = []
        n_eval_views = len(self.datamanager.full_view_eval_dataloader.selected_views)
        n_train_views = len(self.datamanager.full_view_train_dataloader.selected_views)
        for i in range(n_eval_views + n_train_views):
            if i < n_eval_views:
                dataset = self.datamanager.eval_dataset
                iter_dataloader = self.datamanager.iter_full_view_eval_dataloader
                ray_generator = self.datamanager.eval_ray_generator
            elif i - n_eval_views < n_train_views:
                dataset = self.datamanager.train_dataset
                iter_dataloader = self.datamanager.iter_full_view_train_dataloader
                ray_generator = self.datamanager.train_ray_generator
            else:
                raise ValueError("View index out of range.")

            renderings, gt_frames, view_idx = self.render_view(
                step=step,
                dataset=dataset,
                iter_dataloader=iter_dataloader,
                ray_generator=ray_generator,
            )

            self.export_rendered_frames(
                *renderings[1:],
                step=step,
                output_path=output_path,
                scale=1.0,
                view_idx=view_idx,
                single_channels=True,
            )

            masks = {
                mod: renderings[-1][f"accumulation_{mod}"] > 0.9
                if renderings[-1][f"accumulation_{mod}"] is not None
                else None
                for mod in self.datamanager.modalities
            } if self.config.roi_only else None
            metrics = self.compute_metrics(renderings[0], gt_frames, masks=masks, roi_only=self.config.roi_only)
            all_metrics['idx'].append(view_idx)
            for mod, mod_metrics in metrics.items():
                for metric_name, metric in mod_metrics.items():
                    all_metrics[mod][metric_name].append(metric)

        if n_eval_views + n_train_views > len(view_ids):
            all_metrics = self.merge_metrics(all_metrics)
        self.export_metrics(all_metrics, os.path.join(output_path, 'results.txt'), step)

    def render_all_eval_views(self, step, output_path=None):
        """Render, compute the metrics and save all the frames of the evaluation set."""
        output_path = output_path if output_path is not None else os.path.join(self.output_path, 'validation')
        all_metrics = {mod: defaultdict(list) for mod in self.datamanager.modalities}
        all_metrics['idx'] = []
        for _ in range(len(self.datamanager.eval_dataset.get_unique_views())):
            renderings, gt_frames, view_idx = self.render_view(
                step=step,
                dataset=self.datamanager.eval_dataset,
                iter_dataloader=self.datamanager.iter_full_view_eval_dataloader,
                ray_generator=self.datamanager.eval_ray_generator,
            )

            self.export_rendered_frames(
                *renderings[1:],
                step=step,
                output_path=output_path,
                scale=1.0,
                view_idx=view_idx,
            )

            masks = {
                mod: renderings[-1][f"accumulation_{mod}"] > 0.9
                if renderings[-1][f"accumulation_{mod}"] is not None
                else None
                for mod in self.datamanager.modalities
            } if self.config.roi_only else None
            metrics = self.compute_metrics(renderings[0], gt_frames, masks=masks, roi_only=self.config.roi_only)
            all_metrics['idx'].append(view_idx)
            for mod, mod_metrics in metrics.items():
                for metric_name, metric in mod_metrics.items():
                    all_metrics[mod][metric_name].append(metric)

        self.export_metrics(all_metrics, os.path.join(output_path, 'results.txt'), step)

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
        if outputs is None:
            outputs = eval_model_query(
                ray_bundles=ray_bundles,
                num_rays_per_chunk=eval_num_rays_per_chunk,
                model_fn=forward_fn,
                step=step
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

        if "polarization" in self.datamanager.modalities and ray_bundles["polarization"] is not None:
            extra_renderings["degree_of_polarization"] = polarizer.to_dop(data=aligned_renderings["polarization"])
            extra_renderings["angle_of_polarization"] = polarizer.to_aop(data=aligned_renderings["polarization"]) / np.pi

        side_by_side_renderings, aligned_renderings, geometry_renderings = combine_renderings(
            renderings=renderings,
            aligned_renderings=aligned_renderings,
            geometry_renderings=geometry_renderings,
            gt_frames=gt_pixels,
        )
        extra_renderings = {f"accumulation_{mod}": extra_renderings[f"accumulation_{mod}"]
                                if len(extra_renderings[f"accumulation_{mod}"]) != 0
                                else None
                                for mod in self.datamanager.modalities
                            }

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
            renderings=aligned_renderings,
            export_path=os.path.join(output_path, 'radiance_renderings', "aligned"),
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

    def export_metrics(self, metrics, path, step):
        """Export the metrics to a txt file."""
        open(path, 'a').close()
        with open(path, 'r+') as f:
            content = f.read()
            f.seek(0, 0)
            indexes = metrics['idx']
            if not isinstance(indexes[0], list):
                sorting_idx = sorted(range(len(indexes)), key=lambda k: indexes[k])
                indexes = [indexes[i] for i in sorting_idx]
            else:
                indexes = [next((x for i, x in enumerate(idx) if x is not None), None) for idx in indexes]
                sorting_idx = sorted(range(len(indexes)), key=lambda k: indexes[k])
                indexes = [indexes[i] for i in sorting_idx]
            f.write(f"Step: {step}\n")
            for mod in metrics:
                if mod == 'idx':
                    continue
                f.write(f"{mod}:\n")
                f.write("\tFRAME:\t" + "\t\t".join([str(val) for val in indexes]))
                f.write("\t\tAVG\n")
                for metr in metrics[mod]:
                    metrics[mod][metr] = [metrics[mod][metr][i] for i in sorting_idx]
                    f.write(f"\t{metr}:\t" + "\t".join([
                        f"{val:.3f}" if not isinstance(val, str) else f"{val}\t"
                        for val in metrics[mod][metr]
                    ]))
                    avg = sum(val for val in metrics[mod][metr] if not isinstance(val, str)) / \
                          sum(1 for val in metrics[mod][metr] if not isinstance(val, str))
                    f.write(
                        f"\t{avg:.3f}\n"
                    )
            f.write("\n")
            f.write(content)

    def merge_metrics(self, all_metrics):
        """Merge and order the metrics computed on views belonging to the training and evaluation sets."""
        unique_ids = list({
            next((x for i, x in enumerate(idx) if x is not None), None)
            for idx in all_metrics['idx']
        })
        unique_ids.sort()
        modalities = self.datamanager.modalities
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

    def export_mesh(self, step):
        """Generate and save the 3D mesh"""
        if self.config.export_mesh:
            with torch.no_grad():
                self.mesh_extractor.extract(
                    sdf_fn=self.model.surface_model.surface_field.single_output,
                    step=step,
                )

    def export_poses(self, step):
        """Export the optimized camera poses"""
        if self.config.export_poses:
            with torch.no_grad():
                self.pose_extractor.extract(
                    step=step,
                )

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
        self.mosaick_mask_per_modality = self.datamanager.train_dataset.mosaick_mask_per_modality
        self.mosaick_mask_across_modalities = self.datamanager.train_dataset.mosaick_mask_across_modalities

    def render_view(
            self,
            step: int,
            dataset: BaseDataset,
            ray_generator: RayGenerator,
            iter_dataloader = None,
            relative_view_idx: Union[List[int], int] = None,
            view_idx: Union[List[int], int] = None,
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
                first_mod = list(self.datamanager.modalities.keys())[0]
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
                relative_view_idx[self.datamanager.modalities.index(first_valid_mod)]
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
                                      modalities=self.datamanager.modalities,
                                      c2w=c2w,
                                      step=step,
                                      forward_fn=self.model,
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
        if outputs is None:
            outputs = eval_model_query(
                ray_bundles=ray_bundles,
                num_rays_per_chunk=eval_num_rays_per_chunk,
                model_fn=forward_fn,
                step=step
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
            for mod in self.datamanager.modalities
        }
        pixel_coords_per_aligned_modality = {
            mod: pixel_coords_per_modality[first_valid_mod]
            for mod in self.datamanager.modalities
        }

        if "polarization" in self.datamanager.modalities and ray_bundles["polarization"] is not None:
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
        extra_renderings = {f"accumulation_{mod}": extra_renderings[f"accumulation_{mod}"]
                                if len(extra_renderings[f"accumulation_{mod}"]) != 0
                                else None
                                for mod in self.datamanager.modalities
                            }

        return renderings, mosaicked_renderings, side_by_side_renderings, aligned_renderings, geometry_renderings, extra_renderings

    def select_right_channel_per_rendered_pixel(self, pixel_coords_per_modality, renderings, mosaick_mask_per_modality):
        """
        Select the right channel per pixel for the rendered frames, according to the mosaick mask.
        """
        if isinstance(renderings, dict):
            mosaicked_renderings = {}
            for mod in self.datamanager.modalities:
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
        single_channels = True if step == self.pipeline.trainer_config.steps_per_eval_all_images else single_channels
        export_renderings(
            renderings=demosaicked_renderings,
            export_path=os.path.join(output_path, 'radiance_renderings', "demosaicked"),
            step=step,
            view_idx=view_idx,
            scale=scale,
            single_channels=single_channels,
        )
        export_renderings(
            renderings=aligned_renderings,
            export_path=os.path.join(output_path, 'radiance_renderings', "aligned"),
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
        export_renderings(
            renderings=extra_renderings,
            export_path=os.path.join(output_path, 'extra_renderings'),
            step=step,
            view_idx=view_idx,
            scale=scale
        )
