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
Script to preprocess MMS-DATA dataset and prepare it for training.
"""

import os
import sys
import json
import argparse
from datetime import datetime

import cv2 as cv
import numpy as np
import polanalyser as pa

from preprocessing.utils import multispectral_sorting, multispectral_demosaicking, generate_bounding_box, build_metadata
from utils import extract_camera_model, check_cameras, process_camera_matrix, process_frames
from colmap import feature_extractor, feature_matcher, mapper, converter, prepare_images, compute_colmap_scale

demosaicking_fns = {}
### Define here your demosaicking functions
### If not defined, they will be initialized with the identity function
# E.g.
demosaicking_fns["rgb"] = lambda x: cv.demosaicing(x, cv.COLOR_BayerGR2BGR_EA)
demosaicking_fns["polarization"] = lambda x: np.stack(pa.demosaicing(x, pa.COLOR_PolarMono_EA), axis=-1)
demosaicking_fns["multispectral"] = lambda x: multispectral_sorting(multispectral_demosaicking(x))
###

mosaick_patterns = {}
### Define here your mosaick patterns
### If not defined, they will be initialized with the mosaick pattern in calibration JSON file
# E.g.
mosaick_patterns["rgb"] = [[1, 2], [0, 1]]
mosaick_patterns["polarization"] = [[2, 1], [3, 0]]
mosaick_patterns["multispectral"] = [[4, 5, 6], [2, 1, 0], [3, 8, 7]]
mosaick_patterns["infrared"] = [[0]]
mosaick_patterns["mono"] = [[0]]
###

modality_roi = {}
### Define here the Region of Interest (ROI) for each modality
### If not defined, they will be initialized with the full image
# E.g.
modality_roi["multispectral"] = (6, 21, 1269, 981)
###

# Minimum distance between COLMAP sparse 3D points to consider them as a single cluster
# This is used to identify the object of interest and generate the bounding box
MIN_POINT_DISTANCE = 0.35

if __name__ == '__main__':
    print("### MULTIMODALSTUDIO MMS-DATA PREPROCESSING ###\n")

    parser = argparse.ArgumentParser(description='Prepare custom dataset for training.')

    parser.add_argument('--source-path', required=True, type=str, help='source input data path')
    parser.add_argument('--colmap-path', type=str, default="", help='colmap input/output data path')
    parser.add_argument('--run-colmap', action='store_const', const=True, default=False, help='Whether to run colmap or not. If not, colmap-path must be provided. If --calibration is specified, it runs COLMAP only on the first modality.')
    parser.add_argument('--output-path', required=True, type=str, help='output data path')
    parser.add_argument('--modalities', required=True, nargs='+', help='modality names list.')
    parser.add_argument('--calibration', type=str, default=None, help='Path to calibration JSON file. The first modality in the dict is considered as reference modality. If not provided, COLMAP is run on all modalities.')
    parser.add_argument('--scale', type=float, default=1.0, help='Scaling factor.')
    parser.add_argument('--undistort', action='store_const', const=True, default=False, help='Whether to perform image undistortion')
    parser.add_argument('--demosaick', action='store_const', const=True, default=False, help='Whether to perform image demosaicking')
    args = parser.parse_args()

    start_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    if not os.path.exists(args.source_path):
        print("Source data path does not exist!")
        sys.exit()

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    else:
        args.output_path = os.path.join(args.output_path, f"output_{start_time}")
        os.makedirs(args.output_path)
    print(f"Output path: {args.output_path}")

    os.makedirs(args.colmap_path, exist_ok=True)
    print(f"Colmap path: {args.colmap_path}")

    print(f"Modalities: {args.modalities}")

    calibration = None
    if args.calibration is not None:
        with open(args.calibration, 'r') as f:
            calibration = json.load(f)
        print(f"Calibration data: {args.calibration}")

    d_fns = {}
    for mod in args.modalities - demosaicking_fns.keys():
        d_fns[mod] = lambda x: x
    demosaicking_fns.update(d_fns)

    m_patterns = {}
    for mod in args.modalities - mosaick_patterns.keys():
        m_patterns[mod] = calibration[mod]["mosaick_pattern"]
    mosaick_patterns.update(m_patterns)

    if args.run_colmap:
        # Run colmap
        print("Running COLMAP...")

        if len(os.listdir(args.colmap_path)) != 0:
            args.colmap_path = os.path.join(args.colmap_path, os.pardir, f"colmap_{start_time}")
            os.makedirs(args.colmap_path)

        if calibration is not None:
            reference_camera = calibration[next(iter(calibration))]
            camera_params = [
                reference_camera["fx"],
                reference_camera["fy"],
                reference_camera["cx"],
                reference_camera["cy"]
            ] + reference_camera["distortion_params"][:-2]

        prepare_images(
            args.source_path,
            args.colmap_path,
            args.modalities if calibration is None else [next(iter(calibration))],
            mosaicked=True,
            demosaick_fns=demosaicking_fns
        )

        feature_extractor(
            os.path.join(args.colmap_path, "temp_images", "modalities"),
            args.colmap_path,
            camera_model="OPENCV",
            single_camera=True,
            camera_params=camera_params
        )
        feature_matcher(args.colmap_path)
        mapper(
            os.path.join(args.colmap_path, "temp_images", "modalities"),
            args.colmap_path,
            ba_refine_focal_length=0,
            ba_refine_principal_point=0,
            ba_refine_extra_params=0,
        )
        converter(os.path.join(args.colmap_path, "sparse", "0"))

        # os.remove(os.path.join(args.colmap_path, "colmap_output.txt"))
        # os.remove(os.path.join(args.colmap_path, "database.db"))

    # Extract camera model parameters
    print("Extracting camera model...")
    modality_data = extract_camera_model(args.modalities, calibration, modality_roi)

    # Process camera matrices
    print("Processing camera matrices...")
    modality_data = process_camera_matrix(args.modalities, modality_data, args.undistort, args.scale)

    # Process images (perform undistortion and rescaling if needed)
    print("Processing frames...")
    channels_per_mod = process_frames(
        args.source_path,
        args.output_path,
        args.modalities,
        modality_data,
        args.undistort,
        args.scale,
        args.demosaick,
        demosaicking_fns
    )

    # Compute colmap scale factor
    print("Computing GT2WORLD transformation...")
    colmap2gt_scale = compute_colmap_scale(
        args.colmap_path,
        next(iter(calibration)),
        modality_data
    )

    # Generate bounding box
    print("Generating bounding box...")
    gt2w, bbox = generate_bounding_box(
        args.output_path,
        args.colmap_path,
        radius=MIN_POINT_DISTANCE,
        scale=colmap2gt_scale,
        pointcloud_filtering=True,
        reorient_axis=True
    )

    # Build metadata
    print("Building metadata...")
    build_metadata(
        args.output_path,
        args.colmap_path,
        args.modalities,
        modality_data,
        gt2w,
        bbox,
        scale=colmap2gt_scale,
        calibration=calibration,
        channels_per_mod=channels_per_mod,
        undistorted=args.undistort,
        mosaicked=not args.demosaick,
        mosaick_patterns=mosaick_patterns,
    )

    # Check cameras
    print("Exporting camera poses...")
    check_cameras(args.output_path, args.modalities)

    # if args.run_colmap:
    #     shutil.rmtree(os.path.join(args.colmap_path, "temp_images"))
    #     shutil.rmtree(os.path.join(args.output_path, "sparse"))
    print("Preprocessing done!")
