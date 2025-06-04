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
Script to metrics of rendered multimodal frames of several scenes.
"""

import os
from collections import defaultdict

import numpy as np
import cv2 as cv
import polanalyser as pa
import h5py

import torch
from scipy.interpolate import RegularGridInterpolator
from torchmetrics.image import StructuralSimilarityIndexMeasure, LearnedPerceptualImagePatchSimilarity

# NB: Only for distorted frames

# Whether the rendered frames and GT frames are raw or demosaicked
IS_RAW = True

scenes = [
    "birdhouse", "africanart", "book", "clock", "forestgang1", "gamepads", "laptop", "pillow", "steelpot", "toys",
    "vases", "aloe", "bouquet", "easteregg", "forestgang2", "glassclock", "laurelwreath", "plant", "teddybear",
    "trophies", "wateringcan1", "fan", "fruits", "globe", "legoship", "tinbox1", "tinbox2", "truck", "wateringcan2",
    "chess", "makeup", "orchid"
]
modalities = ["rgb", "infrared", "mono", "polarization", "multispectral"]

general_path = "path_to_training_output_folder"
consistent_mask_path = "path_to_mask_training_output_folder" # Training performed on all the views to generate the masks
source_data_path = "path_to_gt_data"

# Number of training iterations
num_train_iterations = 100000

# Indices of evaluation views
eval_indexes = [9, 19, 29, 39, 49]

# Modality-specific demosaicking functions
mosaick_patterns = {
    "rgb": [[1, 2], [0, 1]],
    "infrared": [[0]],
    "mono": [[0]],
    "polarization": [[2, 1], [3, 0]],
    "multispectral": [[4, 5, 6], [2, 1, 0], [3, 8, 7]],
}

# Modality-specific demosaicking functions
demosaicking_fns = {
    "rgb": lambda x: cv.demosaicking(x, cv.COLOR_BayerGR2BGR_EA),
    "infrared": lambda x: np.copy(x),
    "mono": lambda x: np.copy(x),
    "polarization": lambda x: np.stack(pa.demosaicking(x.squeeze(), pa.COLOR_PolarMono_EA), axis=-1),
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
    rendering_t = torch.tensor(img1).permute(2, 0, 1).unsqueeze(0).cuda() / 65535.
    gt_t = torch.tensor(img2).permute(2, 0, 1).unsqueeze(0).cuda() / 65535.
    mask = torch.tensor(mask).permute(2, 0, 1).unsqueeze(0).cuda()
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
    rendering_t = torch.tensor(img1).permute(2, 0, 1).unsqueeze(0).cuda() / 65535.
    gt_t = torch.tensor(img2).permute(2, 0, 1).unsqueeze(0).cuda() / 65535.
    mask = torch.tensor(mask).permute(2, 0, 1).unsqueeze(0).cuda()
    masked_rendering_t = torch.zeros_like(rendering_t, device=rendering_t.device)
    masked_rendering_t[mask] = rendering_t[mask]
    masked_rendering_t = masked_rendering_t
    masked_gt_t = torch.zeros_like(gt_t, device=gt_t.device)
    masked_gt_t[mask] = gt_t[mask]
    masked_gt_t = masked_gt_t
    lpips = lpips_fn(masked_rendering_t, masked_gt_t).item()
    return lpips

metrics = {}
for scene_name in scenes:
    gt_path = os.path.join(source_data_path, scene_name)
    rendering_path = os.path.join(
        general_path,
        "validation", "radiance_renderings", "validation"
    ).replace("PLACEHOLDER", scene_name)
    demosaicked_rendering_path = os.path.join(
        general_path,
        "validation", "radiance_renderings", "demosaicked"
    ).replace("PLACEHOLDER", scene_name)
    mask_path = os.path.join(
        consistent_mask_path,
        "evaluation", "extra_renderings"
    ).replace("PLACEHOLDER", scene_name)
    metrics[scene_name] = {}

    ssim_fn = StructuralSimilarityIndexMeasure(reduction='none', data_range=1.0, return_full_image=True)
    lpips_fn = LearnedPerceptualImagePatchSimilarity(normalize=True).cuda()

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
            mask, _ = read_frame_given_name(mask_path, f"{num_train_iterations+1:07}_{idx}_accumulation_{mod}")
            mask = (mask / 65535.) > 0.9
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

            if IS_RAW:
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
                # Rendered demosaicked PSNR
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

max_key_length = max([len(key) + 1 for key in average_over_scenes.keys()])
n_tabs = max_key_length // 8
front_space_labels = "".join(["\t" for _ in range(n_tabs+2)]) + "\t\t"

print("Average metrics per modality over all scenes")
print(f"{front_space_labels}MOSAICKED\t\t\t\t\t\t\tDEMOSAICKED\t\t\t\t\t\t\tRENDERED DEMOSAICKED")
print(f"{front_space_labels}PSNR\t\tSSIM\t\tLPIPS\t\tPSNR\t\tSSIM\t\tLPIPS\t\tPSNR\t\tSSIM\t\tLPIPS")
for mod in modalities:
    n_tabs = round((max_key_length - len(mod) - 1) / 8)
    front_space = "".join(["\t" for _ in range(n_tabs+2)])
    if mod in ["rgb", "mono", "infrared"]:
        front_space = front_space + "\t"
    if mod in ["mono", "infrared"]:
        print(f'{mod.upper()}:{front_space}'+\
              '-\t\t\t-\t\t\t-\t\t\t'+\
              '-\t\t\t-\t\t\t-\t\t\t'+\
              f'{average_over_scenes[mod].mean(axis=-1)[6]:.3f}\t\t{average_over_scenes[mod].mean(axis=-1)[7]:.3f}\t\t{average_over_scenes[mod].mean(axis=-1)[8]:.3f}'
              )
    elif mod in ["polarization", "multispectral"]:
        print(
            f'{mod.upper()}:{front_space}'+\
            f'{average_over_scenes[mod].mean(axis=-1)[0]:.3f}\t\t-\t\t\t-'+\
            f'\t\t\t{average_over_scenes[mod].mean(axis=-1)[3]:.3f}\t\t{average_over_scenes[mod].mean(axis=-1)[4]:.3f}\t\t-'+\
            f'\t\t\t{average_over_scenes[mod].mean(axis=-1)[6]:.3f}\t\t{average_over_scenes[mod].mean(axis=-1)[7]:.3f}\t\t-'
        )
    else:
        print(
            f'{mod.upper()}:{front_space}'+\
            f'{average_over_scenes[mod].mean(axis=-1)[0]:.3f}\t\t-\t\t\t-'+\
            f'\t\t\t{average_over_scenes[mod].mean(axis=-1)[3]:.3f}\t\t{average_over_scenes[mod].mean(axis=-1)[4]:.3f}\t\t{average_over_scenes[mod].mean(axis=-1)[5]:.3f}'+\
            f'\t\t{average_over_scenes[mod].mean(axis=-1)[6]:.3f}\t\t{average_over_scenes[mod].mean(axis=-1)[7]:.3f}\t\t{average_over_scenes[mod].mean(axis=-1)[8]:.3f}'
        )

print("\nAverage metrics per modality per scene")
for scene_name in scenes:
    print(f"{scene_name.upper()}")
    print(f"{front_space_labels}MOSAICKED\t\t\t\t\t\t\tDEMOSAICKED\t\t\t\t\t\t\tRENDERED DEMOSAICKED")
    print(f"{front_space_labels}PSNR\t\tSSIM\t\tLPIPS\t\tPSNR\t\tSSIM\t\tLPIPS\t\tPSNR\t\tSSIM\t\tLPIPS")
    for mod in modalities:
        n_tabs = (max_key_length - len(mod) - 1) // 8
        front_space = "".join(["\t" for _ in range(n_tabs+2)])
        if mod in ["rgb", "mono", "infrared"]:
            front_space = front_space + "\t"
        if mod in ["mono", "infrared"]:
            print(f'{mod.upper()}:{front_space}' + \
                  '-\t\t\t-\t\t\t-\t\t\t' + \
                  '-\t\t\t-\t\t\t-\t\t\t' + \
                  f'{metrics[scene_name][mod][6]:.3f}\t\t{metrics[scene_name][mod][7]:.3f}\t\t{metrics[scene_name][mod][8]:.3f}'
                  )
        elif mod in ["polarization", "multispectral"]:
            print(
                f'{mod.upper()}:{front_space}'+\
                f'{metrics[scene_name][mod][0]:.3f}\t\t-\t\t\t-'+\
                f'\t\t\t{metrics[scene_name][mod][3]:.3f}\t\t{metrics[scene_name][mod][4]:.3f}\t\t-'+\
                f'\t\t\t{metrics[scene_name][mod][6]:.3f}\t\t{metrics[scene_name][mod][7]:.3f}\t\t-'
            )
        else:
            print(
                f'{mod.upper()}:{front_space}'+\
                f'{metrics[scene_name][mod][0]:.3f}\t\t-\t\t\t-'+\
                f'\t\t\t{metrics[scene_name][mod][3]:.3f}\t\t{metrics[scene_name][mod][4]:.3f}\t\t{metrics[scene_name][mod][5]:.3f}'+\
                f'\t\t{metrics[scene_name][mod][6]:.3f}\t\t{metrics[scene_name][mod][7]:.3f}\t\t{metrics[scene_name][mod][8]:.3f}'
            )
    print("\n")
