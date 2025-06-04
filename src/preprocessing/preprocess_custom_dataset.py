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
Script to preprocess a custom dataset and prepare it for training.
"""

import json
import os
import sys
import argparse
from datetime import datetime

import cv2 as cv
import numpy as np
import polanalyser as pa

from preprocessing.colmap import prepare_images
from utils import generate_bounding_box, process_frames, build_metadata, check_cameras, \
    process_camera_matrix, extract_camera_model, multispectral_sorting, multispectral_demosaicking
from colmap import extract_camera_model_from_colmap, feature_extractor, feature_matcher, mapper, converter

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

# COLMAP parameters
CAMERA_MODEL = "SIMPLE_RADIAL"
REFINE_FOCAL_LENGTH = 1
REFINE_PRINCIPAL_POINT = 1
REFINE_EXTRA_PARAMS = 1

# Minimum distance between COLMAP sparse 3D points to consider them as a single cluster
# This is used to identify the object of interest and generate the bounding box
MIN_POINT_DISTANCE = 0.3

if __name__ == '__main__':
    print("### MULTIMODALSTUDIO PREPROCESSING CUSTOM (OBJECT-CENTRIC) DATASET ###\n")

    parser = argparse.ArgumentParser(description='Prepare custom object-centric dataset for training.')

    parser.add_argument('--source-path', required=True, type=str, help='source input data path')
    parser.add_argument('--colmap-path', type=str, default="", help='colmap input/output data path')
    parser.add_argument('--run-colmap', action='store_const', const=True, default=False, help='Whether to run colmap or not. If not, colmap-path must be provided. If --calibration is specified, it runs COLMAP only on the first modality.')
    parser.add_argument('--output-path', required=True, type=str, help='output data path')
    parser.add_argument('--modalities', required=True, nargs='+', help='modality names list.')
    parser.add_argument('--calibration', type=str, default=None, help='Path to calibration JSON file. The first modality in the dict is considered as reference modality. If not provided, COLMAP is run on all modalities.')
    parser.add_argument('--scale', type=float, default=1.0, help='Scaling factor.')
    parser.add_argument('--undistort', action='store_const', const=True, default=False, help='Whether to perform image undistortion')
    parser.add_argument('--demosaick', action='store_const', const=True, default=False, help='Whether to perform image demosaicking')
    parser.add_argument('--raw-input', action='store_const', const=True, default=False, help='Whether the input images are raw')
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

    # modalities = args.modalities
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
            ] + reference_camera["distortion_params"]

        prepare_images(
            args.source_path,
            args.colmap_path,
            args.modalities if calibration is None else [next(iter(calibration))],
            mosaicked=args.raw_input,
            demosaick_fns=demosaicking_fns
        )

        feature_extractor(
            os.path.join(args.colmap_path, "temp_images", "modalities"),
            args.colmap_path,
            camera_model=CAMERA_MODEL if calibration is None else reference_camera["camera_model"],
            single_camera=False if calibration is None and len(args.modalities) > 1 else True,
            camera_params=None if calibration is None else camera_params
        )
        feature_matcher(args.colmap_path)
        mapper(
            os.path.join(args.colmap_path, "temp_images", "modalities"),
            args.colmap_path,
            ba_refine_focal_length = REFINE_FOCAL_LENGTH if calibration is None else False,
            ba_refine_principal_point = REFINE_PRINCIPAL_POINT if calibration is None else False,
            ba_refine_extra_params = REFINE_EXTRA_PARAMS if calibration is None else False,
        )
        converter(os.path.join(args.colmap_path, "sparse", "0"))

        # os.remove(os.path.join(args.colmap_path, "colmap_output.txt"))
        # os.remove(os.path.join(args.colmap_path, "database.db"))

    modalities = os.listdir(os.path.join(args.colmap_path, "temp_images", "modalities"))
    modalities.sort()

    # Extract camera model parameters
    print("Extracting camera model...")
    if calibration is not None:
        modality_data = extract_camera_model(modalities, calibration, modality_roi)
    else:
        modality_data = extract_camera_model_from_colmap(args.colmap_path, modalities, modality_roi)

    # Process camera matrices
    print("Processing camera matrices...")
    modality_data = process_camera_matrix(modalities, modality_data, args.undistort, args.scale)

    # Process images (perform undistortion and rescaling if needed)
    print("Processing frames...")
    channels_per_mod = process_frames(
        args.source_path,
        args.output_path,
        modalities,
        modality_data,
        args.undistort,
        args.scale,
        args.demosaick,
        demosaicking_fns
    )

    # Generate bounding box
    print("Generating bounding box...")
    gt2w, bbox = generate_bounding_box(args.output_path, args.colmap_path, radius=MIN_POINT_DISTANCE)

    # Build metadata
    print("Building metadata...")
    build_metadata(
        args.output_path,
        args.colmap_path,
        modalities,
        modality_data,
        gt2w,
        bbox,
        calibration=calibration,
        channels_per_mod=channels_per_mod,
        undistorted=args.undistort,
        mosaicked=args.raw_input and not args.demosaick,
        mosaick_patterns=mosaick_patterns,
    )

    # Check cameras
    print("Exporting camera poses...")
    check_cameras(args.output_path, modalities)

    # if args.run_colmap:
    #     shutil.rmtree(os.path.join(args.output_path, "sparse"))
    print("Preprocessing done!")
