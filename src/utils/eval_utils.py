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
Utils for evaluating the model.
"""

import os
from collections import defaultdict
from typing import List, Dict, Union
from torchtyping import TensorType

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import torch
from torchmetrics.functional.image import peak_signal_noise_ratio, structural_similarity_index_measure

from cameras.rays import RayBundle
from utils import writer
from utils.writer import EventName, TimeWriter

def eval_model_query(
        ray_bundles: Dict[str, RayBundle],
        num_rays_per_chunk: int,
        model_fn,
        step,
):
    """
    Performs a model query on the given ray bundles.
    Args:
        ray_bundles: Dictionary of ray bundles to query.
        num_rays_per_chunk: Number of rays to process in each chunk.
        model_fn: Model function to query.
        step: Current training step.
    Returns:
        outputs: List of model outputs for each ray bundle.
    """
    outputs = []
    max_pixel_number = max([len(x) if x is not None else 0 for x in ray_bundles.values()])

    with TimeWriter(writer, EventName.TEST_RAYS_PER_SEC, write=False) as test_t:
        for i in range(0, max_pixel_number, num_rays_per_chunk):
            start_idx = i
            end_idx = i + num_rays_per_chunk
            ray_bundles_chunk = {}
            for mod in ray_bundles:
                if ray_bundles[mod] is None:
                    ray_bundles_chunk[mod] = None
                elif len(ray_bundles[mod]) <= start_idx:
                    ray_bundles_chunk[mod] = None
                elif len(ray_bundles[mod]) < end_idx:
                    ray_bundles_chunk[mod] = ray_bundles[mod][start_idx:]
                else:
                    ray_bundles_chunk[mod] = ray_bundles[mod][start_idx:end_idx]

            with torch.no_grad():
                output = model_fn(ray_bundles_chunk)
            outputs.append(output)

    writer.put_time(
        name=EventName.TEST_RAYS_PER_SEC,
        duration=sum([len(x) if x is not None else 0 for x in ray_bundles.values()]) / test_t.duration,
        step=step,
        avg_over_steps=True,
    )
    return outputs

def render_outputs(
        outputs: List[Dict[str, Dict[str, TensorType]]],
        modalities: List[str],
        gt_frames: Dict[str, TensorType],
        c2w: TensorType,
):
    """
    Renders the outputs of the model and returns the renderings for each modality.
    Args:
        outputs: List of model outputs for each ray bundle.
        modalities: List of modalities to render.
        gt_frames: Dictionary of ground truth frames for each modality.
        c2w: Camera to world transformation matrix.
    Returns:
        renderings: Dictionary of rendered frames for each modality.
        aligned_renderings: Dictionary of aligned frames for each modality, aligned to the first modality.
        geometry_renderings: Dictionary of geometry renderings (normals, depths), aligned to the first modality.
        extra_renderings: Dictionary of extra renderings (accumulation, etc.).
    """
    renderings = defaultdict(list)
    geometry_renderings = defaultdict(list)
    aligned_renderings = defaultdict(list)
    extra_renderings = defaultdict(list)

    first_valid_mod = next((mod for mod, frame in gt_frames.items() if frame is not None), None)
    for current_output in outputs:
        for mod in modalities:
            if current_output[mod] is not None:
                data = current_output[mod][mod]
                mask = data is not None
                data = data[mask].squeeze(dim=0)
                renderings[mod].append(data)

                data = current_output[mod]['accumulation']
                extra_renderings[f'accumulation_{mod}'].append(data)
            else:
                len(renderings[mod]) # Trick to keep the modalities order

            if mod == first_valid_mod and current_output[mod] is not None:
                for m in modalities:
                    aligned_renderings[m].append(current_output[mod][m])

                normals = current_output[mod]["normals"]
                mask = normals is not None
                normals = normals[mask].squeeze(dim=0)
                geometry_renderings['normals'].append(normals)

                depths = current_output[mod]['depth']
                mask = depths is not None
                depths = depths[mask].squeeze(dim=0)
                geometry_renderings['depths'].append(depths)

                extra_modalities = current_output[mod].keys() - modalities - {'depth', "normals", 'accumulation'}
                for m in extra_modalities:
                    data = current_output[mod][m]
                    if m == "specular_tint":
                        data = data[..., :3]
                    extra_renderings[m].append(data)

    for mod in modalities:
        if renderings[mod] is None or len(renderings[mod]) == 0:
            renderings[mod] = None
            aligned_renderings[mod] = None
            continue
        frame = torch.cat(renderings[mod])
        frame = frame.view((*(gt_frames[mod].shape[:-1]), -1))
        renderings[mod] = frame

        frame = torch.cat(aligned_renderings[mod])
        frame = frame.view(*(gt_frames[first_valid_mod].shape[:-1]), -1)
        aligned_renderings[mod] = frame

    for mod, geom_rendering in geometry_renderings.items():
        frame = torch.cat(geom_rendering)
        if mod == 'depths':
            frame = frame.view(*gt_frames[first_valid_mod].shape[:-1], -1)
            frame = frame.cpu().numpy()
            mask = frame != 0
            frame[mask] = (frame[mask] - frame[mask].min()) / (frame[mask].max() - frame[mask].min())
            colored_frame = plt.get_cmap('viridis')(frame[mask]).squeeze()[..., :3]
            frame = 0.5 * np.ones((*frame.shape[:-1], 3))
            mask = mask.squeeze()
            frame[mask] = colored_frame
            frame = frame.astype(np.float32)
        elif mod == 'normals':
            w2c = torch.linalg.inv(c2w[:3, :3]).to(frame.device)
            frame = (w2c @ frame.T).T
            frame = frame.view(*gt_frames[first_valid_mod].shape[:-1], 3)
            frame = (frame + 1) / 2
            frame = frame.cpu().numpy()
        geometry_renderings[mod] = frame

    accumulation_list = [f"accumulation_{mod}" for mod in modalities]
    for extra_key, extra_rendering in extra_renderings.items():
        frame = torch.cat(extra_rendering)
        extra_mod = list(modalities.keys())[accumulation_list.index(extra_key)]
        frame = frame.view((*(gt_frames[extra_mod].shape[:-1]), -1)) \
            if extra_key in accumulation_list \
            else frame.view((*(gt_frames[first_valid_mod].shape[:-1]), -1))
        extra_renderings[extra_key] = frame

    return renderings, aligned_renderings, geometry_renderings, extra_renderings

def combine_renderings(
        renderings: Dict[str, TensorType],
        aligned_renderings: Dict[str, TensorType],
        geometry_renderings: Dict[str, TensorType],
        gt_frames: Dict[str, TensorType],
):
    """
    Combines the renderings in single images
    Args:
        renderings: Dictionary of rendered frames for each modality.
        aligned_renderings: Dictionary of aligned frames for each modality, aligned to the first modality.
        geometry_renderings: Dictionary of geometry renderings (normals, depths).
        gt_frames: Dictionary of ground truth frames for each modality.
    Returns:
        side_by_side_renderings: side-by-side renderings of different modality in a single image.
        aligned_renderings: Aligned renderings for each modality in a single image.
        geometry_renderings: Geometry renderings aligned to the first modality in a single image.
    """
    side_by_side_renderings = defaultdict(list)

    first_valid_mod = next((mod for mod in gt_frames if gt_frames[mod] is not None), None)
    for mod in renderings.keys():
        if gt_frames[mod] is None:
            side_by_side_renderings[mod] = None
            continue
        frame = renderings[mod]
        gt = gt_frames[mod]

        # Side by side frames [rendering, GT, difference]
        diff = torch.linalg.norm(frame.clip(0., 1.) - gt, dim=-1)
        diff = diff.unsqueeze(dim=-1).expand(frame.shape)
        side_by_side_renderings[mod] = torch.cat([frame, gt, diff], dim=1).cpu().numpy()

        # Aligned renderings
        if mod == first_valid_mod:
            channels = [p.shape[-1] for p in gt_frames.values() if p is not None]
            for aligned_mod, aligned_frame in aligned_renderings.items():
                if aligned_frame is None:
                    continue
                if aligned_frame.shape[-1] > 3:
                    aligned_frame = torch.mean(aligned_frame, dim=-1).unsqueeze(dim=-1)
                if 3 in channels and aligned_frame.shape[-1] != 3:
                    aligned_frame = aligned_frame.expand(*aligned_frame.shape[:-1], 3)
                aligned_renderings[aligned_mod] = aligned_frame
            aligned_renderings = torch.cat(
                [frame for frame in aligned_renderings.values() if frame is not None],
                dim=1
            ).cpu().numpy()

    # Geometry renderings
    geometry_renderings = np.concatenate([geometry_renderings[mod] for mod in geometry_renderings.keys()], axis=1)

    return side_by_side_renderings, aligned_renderings, geometry_renderings

def export_renderings(
        renderings: Union[Dict[str, Union[np.ndarray, TensorType]], Union[np.ndarray, TensorType]],
        export_path: str,
        step: int = None,
        view_idx: Union[List[int], int] = None,
        export_name: str = None,
        scale: float = 1.,
        single_channels: bool = False
):
    """
    Exports the renderings to the given path.
    Args:
        renderings: Rendered frames to export.
        export_path: Path to export the frames.
        step: Current training step.
        view_idx: View index for the frame.
        export_name: Name of the exported file.
        scale: Scale factor for resizing the frames.
        single_channels: Whether to export single channels or not.
    """
    os.makedirs(export_path, exist_ok=True)
    if isinstance(renderings, dict):
        for j, (mod, frame) in enumerate(renderings.items()):
            if frame is None:
                continue
            if export_name is None:
                if isinstance(view_idx, list):
                    view_id = view_idx[j] if j < len(view_idx) else view_idx[0]
                else:
                    view_id = view_idx
                final_export_name = f"{step:07}_{view_id}" #if isinstance(view_idx, list) else f"{step:07}_{view_idx}"
            else:
                final_export_name = export_name
            if isinstance(frame, torch.Tensor):
                frame = frame.cpu().numpy()
            if single_channels and frame.shape[-1] > 1:
                for i in range(frame.shape[-1]):
                    channel = frame[..., i]
                    channel = cv.resize(channel, (0, 0), fx=scale, fy=scale)
                    channel = (channel * 65535.).astype(np.uint16)
                    cv.imwrite(os.path.join(export_path, f"{final_export_name}_{mod}_{i}.png"), channel)
            frame = cv.resize(frame, (0,0), fx=scale, fy=scale)
            if len(frame.shape) < 3:
                frame = np.expand_dims(frame, axis=-1)
            elif frame.shape[-1] == 3:
                frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
            frame = np.clip(frame, 0, 1.)
            frame = (frame * 65535.).astype(np.uint16)
            if mod == "angle_of_polarization":
                frame = cv.applyColorMap(np.right_shift(frame, 8).astype(np.uint8), cv.COLORMAP_TWILIGHT)
            if frame.shape[-1] > 3:
                np.save(os.path.join(export_path, f"{final_export_name}_{mod}.npy"), frame)
                cv.imwrite(
                    filename=os.path.join(export_path, f"{final_export_name}_{mod}.png"),
                    img=frame.mean(axis=-1).astype(np.uint16)
                )
            else:
                cv.imwrite(os.path.join(export_path, f"{final_export_name}_{mod}.png"), frame)
    else:
        if renderings is None:
            return
        frame = renderings
        if export_name is None:
            if isinstance(view_idx, list):
                first_valid_idx = next((i for i in range(len(view_idx)) if view_idx[i] is not None), None)
            final_export_name = f"{step:07}_{view_idx[first_valid_idx]}" \
                if isinstance(view_idx, list) \
                else f"{step:07}_{view_idx}"
        else:
            final_export_name = export_name
        if isinstance(frame, torch.Tensor):
            frame = frame.cpu().numpy()
        if single_channels:
            for i in range(frame.shape[-1]):
                channel = frame[..., i]
                channel = cv.resize(channel, (0, 0), fx=scale, fy=scale)
                channel = (channel * 65535.).astype(np.uint16)
                cv.imwrite(os.path.join(export_path, f"{final_export_name}_{i}.png"), channel)
        frame = cv.resize(frame, (0, 0), fx=scale, fy=scale)
        if len(frame.shape) < 3:
            frame = np.expand_dims(frame, axis=-1)
        elif frame.shape[-1] == 3:
            frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
        frame = np.clip(frame, 0, 1.)
        frame = (frame * 65535.).astype(np.uint16)
        if frame.shape[-1] > 3:
            np.save(os.path.join(export_path, f"{final_export_name}.npy"), frame)
            cv.imwrite(os.path.join(export_path, f"{final_export_name}.png"), frame.mean(axis=-1).astype(np.uint16))
        else:
            cv.imwrite(os.path.join(export_path, f"{final_export_name}.png"), frame)

def compute_metrics(
        output: Union[Dict[str, Dict[str, TensorType]], TensorType],
        gt: Union[Dict[str, TensorType], TensorType],
        mask: Union[Dict[str, TensorType], TensorType] = None,
        modalities: List[str] = None,
        eval_step = False
):
    """
    Computes the PSNR and SSIM metrics for the given output and ground truth.
    The same function is used for both for computing metrics on single sparse pixels and on full images.
    Args:
        output: Model output to compute the metrics on.
        gt: Ground truth data to compare against.
        mask: Mask to apply to the output and ground truth.
        modalities: List of modalities to compute the metrics for.
        eval_step: Whether the function is called during eval step or not.
    Returns:
        metrics: Dictionary of computed metrics for each modality.
    """
    if isinstance(output, dict):
        metrics = {}
        s_val = []
        for mod in modalities:
            renderings = output[mod][mod].clip(0., 1.)
            renderings = torch.moveaxis(renderings, -1, 0)[None, ..., None]
            gt_data = gt[mod]
            gt_data = torch.moveaxis(gt_data, -1, 0)[None, ..., None]
            if mask is not None:
                mask = mask[mod]
                mask = torch.moveaxis(mask, -1, 0)[None, ...]
                output = output[mask]
                gt = gt[mask]
            psnr = peak_signal_noise_ratio(renderings, gt_data, data_range=1.0)
            metrics[mod] = {
                "PSNR": psnr,
            }
        if not eval_step:
            for mod in modalities:
                if "inv_s" in output[mod]:
                    s_val.append(output[mod]["inv_s"])
                    param_name = "inv_s"
                elif "beta" in output[mod]:
                    s_val.append(output[mod]["beta"])
                    param_name = "beta"
            metrics[param_name] = torch.stack(s_val, dim=-1).mean(dim=-1)
        return metrics
    else:
        metrics = {}
        if output is None:
            metrics["PSNR"] = "-"
            metrics["SSIM"] = "-"
            return metrics
        if isinstance(output, np.ndarray):
            output = torch.tensor(output)
            gt = torch.tensor(gt)
            mask = torch.tensor(mask) if mask is not None else None
        output = output.clip(0., 1.)
        output = torch.moveaxis(output, -1, 0)[None, ...]
        gt = torch.moveaxis(gt, -1, 0)[None, ...]
        if mask is not None:
            mask = torch.moveaxis(mask, -1, 0)[None, ...]
            mask = mask.expand(output.shape)
            _, full_image_ssim = structural_similarity_index_measure(output, gt, data_range=1.0, return_full_image=True)
            metrics["SSIM"] = full_image_ssim[mask].mean()
            output = output[mask]
            gt = gt[mask]
        else:
            metrics["SSIM"] = structural_similarity_index_measure(output, gt, data_range=1.0)
        metrics["PSNR"] = peak_signal_noise_ratio(output, gt, data_range=1.0)
        return metrics
