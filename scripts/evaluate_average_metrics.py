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
Script to compute metrics of rendered multimodal frames over several scenes.

It evaluates the frames rendered by a training/evaluation run against the ground truth frames of
the dataset, restricted to the foreground masks, and reports PSNR / SSIM / LPIPS for the mosaicked,
the demosaicked and the separately rendered demosaicked frames.
"""

import os
import argparse
from datetime import datetime
from collections import defaultdict

import numpy as np
import cv2 as cv
import polanalyser as pa
import h5py
from tqdm import tqdm

import torch
from scipy.interpolate import RegularGridInterpolator
from torchmetrics.image import StructuralSimilarityIndexMeasure, LearnedPerceptualImagePatchSimilarity

# NB: Only for distorted frames

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ssim_fn = StructuralSimilarityIndexMeasure(reduction='none', data_range=1.0, return_full_image=True)
lpips_fn = None  # instantiated in compute_metrics, unless LPIPS is disabled

mosaick_patterns = {
    "rgb": [[1, 2], [0, 1]],
    "infrared": [[0]],
    "mono": [[0]],
    "polarization": [[2, 1], [3, 0]],
    "multispectral": [[4, 5, 6], [2, 1, 0], [3, 8, 7]],
}

# Modality-specific demosaicking functions
demosaicking_fns = {
    "rgb": lambda x: cv.demosaicing(x, cv.COLOR_BayerGR2BGR_EA),
    "infrared": lambda x: np.copy(x),
    "mono": lambda x: np.copy(x),
    "polarization": lambda x: np.stack(pa.demosaicing(x.squeeze(), pa.COLOR_PolarMono_EA), axis=-1),
    "multispectral": lambda x: multispectral_sorting(multispectral_demosaicking(x)),
}

def multispectral_sorting(frame):
    """Function to sort the channels of a multispectral frame."""
    frame = frame[:,:,[5, 4, 3, 6, 0, 1, 2, 8, 7]]
    return frame

def multispectral_demosaicking(frame):
    """Function to demosaick a multispectral frame captured by the SILIOS CMS-C1"""
    channels = []
    for i in range(9):
        x = i // 3
        y = i % 3
        mask = np.zeros((3, 3))
        mask[x, y] = 1
        n_bayer_x = frame.shape[0] // 3
        n_bayer_y = frame.shape[1] // 3
        mask = np.tile(mask, (n_bayer_x+1, n_bayer_y+1)).astype(bool)
        mask = mask[:frame.shape[0], :frame.shape[1]]
        x_pixels = np.arange(x, frame.shape[0], 3)
        y_pixels = np.arange(y, frame.shape[1], 3)
        interpolator = RegularGridInterpolator(
            (x_pixels, y_pixels),
            frame[mask].reshape(x_pixels.shape[0], y_pixels.shape[0]),
            bounds_error=False,
            fill_value=None
        )
        points = np.stack(
            np.meshgrid(np.arange(0, frame.shape[0], 1), np.arange(0, frame.shape[1], 1), indexing='ij'),
            axis=-1).reshape(-1, 2)
        values = interpolator(points).reshape(frame.shape[0], frame.shape[1])
        values = values.clip(0, 65535)
        values = np.round(values).astype(np.uint16)
        channels.append(values)

    channels = np.stack(channels, axis=-1)
    return channels

def read_frame(path):
    """Read a frame from the given path."""
    file_format = path.strip().split(".")[-1]
    additional_output = None
    if file_format in ["jpg", "JPG", "jpeg", "JPEG", "png", "PNG"]:
        frame = cv.imread(path, cv.IMREAD_UNCHANGED)
    elif path.endswith(".npy"):
        frame = np.load(path)
    elif path.endswith(".h5"):
        with h5py.File(path, 'r') as f:
            frame = [f[key][:] for key in f.keys()]
            additional_output = {"keys": list(f.keys())}
        frame = np.stack(frame, axis=-1)
    return frame, additional_output

def read_frame_given_index(path, idx):
    """Read a frame given its index."""
    frame_list = os.listdir(path)
    frame_list = [x for x in frame_list if f"{idx:04}" in x]
    frame_name = frame_list[0]
    for name in frame_list:
        if name.endswith(".npy"):
            frame_name = name
            break
    path = os.path.join(path, frame_name)
    return read_frame(path)

def read_frame_given_name(path, name):
    """Read a frame given its name."""
    frame_list = os.listdir(path)
    frame_list = [x for x in frame_list if name in x]
    frame_name = frame_list[0]
    for name in frame_list:
        if name.endswith(".npy"):
            frame_name = name
            break
    path = os.path.join(path, frame_name)
    return read_frame(path)

def get_max_val(arr1, arr2):
    """Get the maximum data type value for the PSNR calculation."""
    assert arr1.dtype == arr2.dtype
    if arr1.dtype == np.uint8:
        return 255
    elif arr1.dtype == np.uint16:
        return 65535
    else:
        return 1

def masked_ssim(img1, img2, mask):
    """Compute SSIM metric for masked images."""
    if img1.ndim == 2:
        img1 = img1[..., np.newaxis]
    if img2.ndim == 2:
        img2 = img2[..., np.newaxis]
    if mask.ndim == 2:
        mask = mask[..., np.newaxis]
        mask = np.repeat(mask, img1.shape[-1], axis=2)
    rendering_t = torch.tensor(img1).permute(2, 0, 1).unsqueeze(0).to(DEVICE) / 65535.
    gt_t = torch.tensor(img2).permute(2, 0, 1).unsqueeze(0).to(DEVICE) / 65535.
    mask = torch.tensor(mask).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    _, ssim_img = ssim_fn(rendering_t, gt_t)
    ssim = ssim_img[mask].mean().item()
    return ssim

def masked_lpips(img1, img2, mask):
    """Compute LPIPS metric for masked images."""
    if img1.ndim == 2:
        img1 = img1[..., np.newaxis]
        img1 = np.repeat(img1, 3, axis=2)
    if img2.ndim == 2:
        img2 = img2[..., np.newaxis]
        img2 = np.repeat(img2, 3, axis=2)
    if mask.ndim == 2:
        mask = mask[..., np.newaxis]
        mask = np.repeat(mask, 3, axis=2)
    rendering_t = torch.tensor(img1).permute(2, 0, 1).unsqueeze(0).to(DEVICE) / 65535.
    gt_t = torch.tensor(img2).permute(2, 0, 1).unsqueeze(0).to(DEVICE) / 65535.
    mask = torch.tensor(mask).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    masked_rendering_t = torch.zeros_like(rendering_t, device=rendering_t.device)
    masked_rendering_t[mask] = rendering_t[mask]
    masked_rendering_t = masked_rendering_t
    masked_gt_t = torch.zeros_like(gt_t, device=gt_t.device)
    masked_gt_t[mask] = gt_t[mask]
    masked_gt_t = masked_gt_t
    lpips = lpips_fn(masked_rendering_t, masked_gt_t).item()
    return lpips


def compute_metrics(
        output_path,
        source_data_path,
        mask_path,
        scenes,
        modalities,
        num_train_iterations,
        eval_indexes,
        is_raw=True,
        multiscene=False,
        masks_from_accumulation=False,
        compute_lpips=True,
        rendered_demosaicked=True,
):
    """Compute the average metrics of the rendered frames of every scene and modality.

    Args:
        output_path: path to the "validation" output folder of the run to evaluate. Unless
            `multiscene` is set, the scene name is substituted for the "PLACEHOLDER" token.
        source_data_path: path to the dataset folder containing the ground truth frames.
        mask_path: path to the foreground masks. By default the masks shipped with the dataset;
            with `masks_from_accumulation`, the output folder of a run trained on all the views.
        scenes: list of scene names to evaluate.
        modalities: list of modalities to evaluate.
        num_train_iterations: training iteration the rendered frames were exported at.
        eval_indexes: indices of the evaluation views.
        is_raw: whether the rendered and the ground truth frames are raw (mosaicked).
        multiscene: whether the run is a multi-scene (pre-training) one, whose output folder
            holds all the scenes.
        masks_from_accumulation: read the masks from the accumulation maps of a mask run
            instead of the masks shipped with the dataset.
        compute_lpips: whether to compute LPIPS. When False, LPIPS is reported as NaN.
        rendered_demosaicked: whether to also evaluate the separately rendered demosaicked
            frames. Requires the run to have exported them.

    Returns:
        metrics: per-scene, per-modality array of the 9 averaged metrics.
        average_over_scenes: per-modality metrics stacked over the scenes.
    """
    global lpips_fn
    if compute_lpips:
        lpips_fn = LearnedPerceptualImagePatchSimilarity(normalize=True).to(DEVICE)
    else:
        lpips_fn = lambda x, y: torch.tensor([float("nan")], device=DEVICE)

    metrics = {}
    for scene_name in tqdm(scenes, desc="Scenes"):
        gt_path = os.path.join(source_data_path, scene_name)
        if multiscene:
            # Pre-training (multi-scene) runs store every scene under a single output folder
            rendering_path = os.path.join(output_path, scene_name, "radiance_renderings", "validation")
            demosaicked_rendering_path = os.path.join(output_path, scene_name, "radiance_renderings", "demosaicked")
        else:
            rendering_path = os.path.join(output_path, "radiance_renderings", "validation").replace("PLACEHOLDER", scene_name)
            demosaicked_rendering_path = os.path.join(output_path, "radiance_renderings", "demosaicked").replace("PLACEHOLDER", scene_name)
        if masks_from_accumulation:
            # Masks rendered by a training run performed on all the views (accumulation maps)
            scene_mask_path = os.path.join(mask_path, "evaluation", "extra_renderings").replace("PLACEHOLDER", scene_name)
        else:
            # Foreground masks shipped with the dataset
            scene_mask_path = os.path.join(mask_path, scene_name, "modalities")
        metrics[scene_name] = {}

        for mod in modalities:
            gt_frame_path = os.path.join(gt_path, "modalities", mod)
            demosaicking_fn = demosaicking_fns[mod]
            mosaick_pattern = mosaick_patterns[mod]

            mosaicked_psnr_all = []
            demosaicked_psnr_all = []
            full_rendering_psnr_all = []
            mosaicked_ssim_all = []
            demosaicked_ssim_all = []
            full_rendering_ssim_all = []
            mosaicked_lpips_all = []
            demosaicked_lpips_all = []
            full_rendering_lpips_all = []
            for idx in eval_indexes:
                gt, _ = read_frame_given_index(gt_frame_path, idx)
                rendering, _ = read_frame_given_name(rendering_path, f"{num_train_iterations:07}_{idx}_{mod}")
                rendering, _, _ = np.split(rendering, 3, axis=1)
                if masks_from_accumulation:
                    mask, _ = read_frame_given_name(scene_mask_path, f"{num_train_iterations+1:07}_{idx}_accumulation_{mod}")
                    mask = (mask / 65535.) > 0.9
                else:
                    mask, _ = read_frame_given_name(os.path.join(scene_mask_path, mod), f"{idx:04}")
                    mask = mask.astype(bool)
                mosaick_mask = np.tile(
                    mosaick_pattern,
                    (rendering.shape[0]//len(mosaick_pattern)+1, rendering.shape[1]//len(mosaick_pattern)+1)
                )
                mosaick_mask = mosaick_mask[:rendering.shape[0], :rendering.shape[1]]
                max_val = get_max_val(rendering, gt)
                psnr = cv.PSNR(rendering[mask], gt[mask], R=max_val)
                ssim = masked_ssim(rendering, gt, mask)
                if gt.ndim == 3 and gt.shape[-1] > 3:
                    lpips = masked_lpips(
                        rendering.mean(axis=-1).astype(np.float32),
                        gt.mean(axis=-1).astype(np.float32),
                        mask
                    )
                else:
                    lpips = masked_lpips(rendering, gt, mask)

                if is_raw:
                    # Mosaicked Metrics
                    mosaicked_psnr = psnr
                    mosaicked_ssim = ssim
                    mosaicked_lpips = lpips
                    # Demosaicked PSNR
                    gt = demosaicking_fn(gt)
                    rendering = demosaicking_fn(rendering+1-1)
                    demosaicked_psnr = cv.PSNR(rendering[mask], gt[mask], R=max_val)
                    demosaicked_ssim = masked_ssim(rendering, gt, mask)
                    if gt.ndim == 3 and gt.shape[-1] > 3:
                        demosaicked_lpips = masked_lpips(
                            rendering.mean(axis=-1).astype(np.float32),
                            gt.mean(axis=-1).astype(np.float32),
                            mask
                        )
                    else:
                        demosaicked_lpips = masked_lpips(rendering, gt, mask)
                    # Rendered demosaicked metrics: available only if the training run exported the
                    # demosaicked renderings (evaluator.export_demosaicked_renderings)
                    if rendered_demosaicked:
                        rendering, _ = read_frame_given_name(
                            demosaicked_rendering_path,
                            f"{num_train_iterations:07}_{idx}_{mod}."
                        )
                        full_rendering_psnr = cv.PSNR(rendering[mask], gt[mask], R=max_val)
                        full_rendering_ssim = masked_ssim(rendering, gt, mask)
                        if gt.ndim == 3 and gt.shape[-1] > 3:
                            full_rendering_lpips = masked_lpips(
                                rendering.mean(axis=-1).astype(np.float32),
                                gt.mean(axis=-1).astype(np.float32),
                                mask
                            )
                        else:
                            full_rendering_lpips = masked_lpips(rendering, gt, mask)
                    else:
                        full_rendering_psnr = full_rendering_ssim = full_rendering_lpips = np.nan
                else:
                    # Rendered demosaicked PSNR
                    full_rendering_psnr = psnr
                    full_rendering_ssim = ssim
                    full_rendering_lpips = lpips
                    # Mosaicked PSNR
                    if len(gt.shape) < 3:
                        gt = gt[..., np.newaxis]
                    if len(rendering.shape) < 3:
                        rendering = rendering[..., np.newaxis] 
                    gt = np.take_along_axis(gt, mosaick_mask[...,np.newaxis], axis=-1)
                    rendering = np.take_along_axis(rendering, mosaick_mask[...,np.newaxis], axis=-1)
                    mosaicked_psnr = cv.PSNR(rendering[mask], gt[mask], R=max_val)
                    mosaicked_ssim = masked_ssim(rendering, gt, mask)
                    if gt.ndim == 3 and gt.shape[-1] > 3:
                        mosaicked_lpips = masked_lpips(
                            rendering.mean(axis=-1).astype(np.float32),
                            gt.mean(axis=-1).astype(np.float32),
                            mask
                        )
                    else:
                        mosaicked_lpips = masked_lpips(rendering.squeeze(), gt.squeeze(), mask)
                    # Demosaicked PSNR
                    gt = demosaicking_fn(gt)
                    rendering = demosaicking_fn(rendering)
                    demosaicked_psnr = cv.PSNR(rendering[mask], gt[mask], R=max_val)
                    demosaicked_ssim = masked_ssim(rendering, gt, mask)
                    if gt.ndim == 3 and gt.shape[-1] > 3:
                        demosaicked_lpips = masked_lpips(
                            rendering.mean(axis=-1).astype(np.float32),
                            gt.mean(axis=-1).astype(np.float32),
                            mask
                        )
                    else:
                        demosaicked_lpips = masked_lpips(rendering.squeeze(), gt.squeeze(), mask)
                mosaicked_psnr_all.append(mosaicked_psnr)
                demosaicked_psnr_all.append(demosaicked_psnr)
                full_rendering_psnr_all.append(full_rendering_psnr)
                mosaicked_ssim_all.append(mosaicked_ssim)
                demosaicked_ssim_all.append(demosaicked_ssim)
                full_rendering_ssim_all.append(full_rendering_ssim)
                mosaicked_lpips_all.append(mosaicked_lpips)
                demosaicked_lpips_all.append(demosaicked_lpips)
                full_rendering_lpips_all.append(full_rendering_lpips)
            metrics[scene_name][mod] = np.stack([
                mosaicked_psnr_all, mosaicked_ssim_all, mosaicked_lpips_all,
                demosaicked_psnr_all, demosaicked_ssim_all, demosaicked_lpips_all,
                full_rendering_psnr_all, full_rendering_ssim_all, full_rendering_lpips_all
            ], axis=0).mean(axis=1)


    average_over_scenes = defaultdict(list)
    for scene_name in scenes:
        for mod in modalities:
            average_over_scenes[mod].append(metrics[scene_name][mod])
    for mod in modalities:
        average_over_scenes[mod] = np.stack(average_over_scenes[mod], axis=-1)

    return metrics, average_over_scenes


# Metric slots that are meaningful for each modality: mosaicked (0-2), demosaicked (3-5) and
# separately rendered demosaicked (6-8) PSNR / SSIM / LPIPS. The remaining ones are reported as "-".
REPORTED_SLOTS = {
    "rgb": [0, 3, 4, 5, 6, 7, 8],
    "mono": [0, 1, 2, 6, 7, 8],
    "infrared": [0, 1, 2, 6, 7, 8],
    "polarization": [0, 3, 4, 6, 7],
    "multispectral": [0, 3, 4, 6, 7],
}


def format_report(metrics, average_over_scenes, scenes, modalities, header=()):
    """Render the metrics as the usual per-modality table, per scene and averaged over scenes."""

    def row(label, values, width):
        slots = REPORTED_SLOTS.get(label.lower(), list(range(9)))
        cells = []
        for i in range(9):
            v = values[i]
            cells.append(f"{v:.3f}" if i in slots and not np.isnan(v) else "-")
        return f"{label.upper()+':':<{width}}" + "".join(f"{c:<12}" for c in cells)

    width = max(len(m) for m in modalities) + 2
    head = f"{'':<{width}}" + f"{'MOSAICKED':<36}{'DEMOSAICKED':<36}{'RENDERED DEMOSAICKED':<36}"
    cols = f"{'':<{width}}" + "".join(f"{c:<12}" for c in ["PSNR", "SSIM", "LPIPS"] * 3)

    lines = list(header)
    lines += ["Average metrics per modality over all scenes", head, cols]
    lines += [row(mod, average_over_scenes[mod].mean(axis=-1), width) for mod in modalities]
    lines += ["", "Average metrics per modality per scene"]
    for scene_name in scenes:
        lines += [scene_name.upper(), head, cols]
        lines += [row(mod, metrics[scene_name][mod], width) for mod in modalities]
        lines += [""]
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate average metrics of rendered multimodal frames.")
    parser.add_argument("--output_folder", required=True, type=str, help='Path to the "validation" output folder created by the training script. Replace the scene name with "PLACEHOLDER" in the path (not needed with --multiscene).')
    parser.add_argument("--gt_path", required=True, type=str, help="Path to the dataset folder containing the ground truth frames.")
    parser.add_argument("--mask_path", required=True, type=str, help='Path to the foreground masks shipped with the dataset. With --masks_from_accumulation, the output folder of a run trained on all the views; replace the scene name with "PLACEHOLDER" there.')
    parser.add_argument("--metrics_output_path", type=str, default=None, help="Folder where the metrics are saved. If omitted, they are only printed.")
    parser.add_argument("--file_name", type=str, default=None, help="Name of the file the metrics are saved to. If it already exists, a timestamp is appended.")
    parser.add_argument("--scene_names", nargs="+", type=str, default=[
        "birdhouse", "africanart", "book", "clock", "forestgang1", "gamepads", "laptop", "pillow", "steelpot", "toys",
        "vases", "aloe", "bouquet", "easteregg", "forestgang2", "glassclock", "laurelwreath", "plant", "teddybear",
        "trophies", "wateringcan1", "fan", "fruits", "globe", "legoship", "tinbox1", "tinbox2", "truck", "wateringcan2",
        "chess", "makeup", "orchid"
    ], help="Scene names to evaluate.")
    parser.add_argument("--modalities", nargs="+", type=str, default=["rgb", "infrared", "mono", "polarization", "multispectral"], help="Modalities to evaluate.")
    parser.add_argument("--num_train_iterations", type=int, default=100000, help="Training iteration the evaluated frames were rendered at.")
    parser.add_argument("--eval_indexes", nargs="+", type=int, default=[9, 19, 29, 39, 49], help="Indices of the evaluation views.")
    parser.add_argument("--is_raw", action="store_true", help="Whether the rendered and the ground truth frames are raw (mosaicked).")
    parser.add_argument("--multiscene", action="store_true", help="Evaluate a multi-scene (pre-training) run, whose output folder holds every scene.")
    parser.add_argument("--masks_from_accumulation", action="store_true", help="Read the masks from the accumulation maps of a run trained on all the views, instead of the masks shipped with the dataset.")
    parser.add_argument("--no_lpips", action="store_true", help="Skip the LPIPS computation; LPIPS is reported as '-'.")
    parser.add_argument("--no_rendered_demosaicked", action="store_true", help="Skip the metrics on the separately rendered demosaicked frames. Use it when the run did not export them (evaluator.export_demosaicked_renderings: False).")
    args = parser.parse_args()

    metrics, average_over_scenes = compute_metrics(
        args.output_folder,
        args.gt_path,
        args.mask_path,
        args.scene_names,
        args.modalities,
        args.num_train_iterations,
        args.eval_indexes,
        is_raw=args.is_raw,
        multiscene=args.multiscene,
        masks_from_accumulation=args.masks_from_accumulation,
        compute_lpips=not args.no_lpips,
        rendered_demosaicked=not args.no_rendered_demosaicked,
    )

    report = format_report(
        metrics, average_over_scenes, args.scene_names, args.modalities,
        header=[
            f"Test folder: {args.output_folder}",
            f"Number of scenes: {len(args.scene_names)}",
            f"Scenes: {args.scene_names}",
            "",
        ],
    )
    print(report)

    if args.metrics_output_path is not None:
        metrics_output_path = args.metrics_output_path
        if len(args.scene_names) == 1:
            metrics_output_path = metrics_output_path.replace("PLACEHOLDER", args.scene_names[0])
        os.makedirs(metrics_output_path, exist_ok=True)
        if args.file_name is not None:
            if os.path.exists(os.path.join(metrics_output_path, f"{args.file_name}.txt")):
                file_name = f"{args.file_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            else:
                file_name = f"{args.file_name}.txt"
        else:
            file_name = f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(os.path.join(metrics_output_path, file_name), "w") as f:
            f.write(report + "\n")
        print(f"\nMetrics saved to: {os.path.join(metrics_output_path, file_name)}")
