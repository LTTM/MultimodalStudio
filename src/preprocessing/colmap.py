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
COLMAP utils for dataset preprocessing
"""

import os
import subprocess

import numpy as np
import cv2 as cv

from preprocessing.utils import qvec2rotmat, undistort_frame
from preprocessing.utils import read_frame

def feature_extractor(image_path, output_path, camera_model = 'OPENCV', camera_params=None, single_camera=True):
    """
    Extract features from images using COLMAP.

    Args:
        image_path (str): Path to the images.
        output_path (str): Path to the output directory.
        camera_model (str): Camera model to use. Default is 'OPENCV'.
        camera_params (list): Camera parameters to use. Default is None.
        single_camera (bool): Whether to use a single camera model for all images. Default is True.
    """
    logfile_name = os.path.join(output_path, 'colmap_output.txt')

    ### Feature extraction
    feature_extractor_args = [
        'colmap', 'feature_extractor',
        '--database_path', os.path.join(output_path, 'database.db'),
        '--image_path', image_path,
        '--ImageReader.camera_model', camera_model,
        '--SiftExtraction.use_gpu', '1',
    ]

    if single_camera:
        feature_extractor_args.extend(['--ImageReader.single_camera', '1'])
    else:
        feature_extractor_args.extend(['--ImageReader.single_camera', '0'])
        feature_extractor_args.extend(['--ImageReader.single_camera_per_folder', '1'])

    if camera_params is not None and single_camera:
        camera_params = ','.join([str(param) for param in camera_params])
        feature_extractor_args.extend(['--ImageReader.camera_params', camera_params])

    output = (subprocess.check_output(feature_extractor_args, universal_newlines=True))
    with open(logfile_name, 'w') as logfile:
        logfile.write(output)
    print('Features extracted')

def feature_matcher(output_path):
    """
    Match features using COLMAP.

    Args:
        output_path (str): Path to the directory containing COLMAP database.
    """
    logfile_name = os.path.join(output_path, 'colmap_output.txt')

    ### Feature matching
    matcher_args = [
        'colmap', 'exhaustive_matcher',
        '--database_path', os.path.join(output_path, 'database.db'),
        '--SiftMatching.use_gpu', '1',
        '--TwoViewGeometry.multiple_models', '0',
    ]

    output = (subprocess.check_output(matcher_args, universal_newlines=True))
    with open(logfile_name, 'a') as logfile:
        logfile.write(output)
    print('Features matched')

def mapper(image_path, output_path, **kargs):
    """
    Create a sparse map using COLMAP.

    Args:
        image_path (str): Path to the images.
        output_path (str): Path to the output directory (containing COLMAP database).
    """

    if not os.path.exists(os.path.join(output_path, 'sparse')):
        os.makedirs(os.path.join(output_path, 'sparse'))

    logfile_name = os.path.join(output_path, 'colmap_output.txt')

    mapper_args = [
        'colmap', 'mapper',
            '--database_path', os.path.join(output_path, 'database.db'),
            '--image_path', image_path,
            '--output_path', os.path.join(output_path, 'sparse'),
            '--Mapper.num_threads', '12',
            '--Mapper.init_min_tri_angle', '4',
            '--Mapper.extract_colors', '0',
            '--Mapper.init_num_trials', '400',
            '--Mapper.multiple_models', '0'
    ]

    for key, value in kargs.items():
        mapper_args.extend([f'--Mapper.{key}', f'{value}'])

    output = (subprocess.check_output(mapper_args, universal_newlines=True))
    with open(logfile_name, 'a') as logfile:
        logfile.write(output)
    print('Sparse map created')

def converter(path):
    """
    Convert COLMAP model from BIN to TXT format.

    Args:
        path (str): Path to the COLMAP model directory.
    """
    converter_args = [
        'colmap', 'model_converter',
            '--input_path', path,
            '--output_path', path,
            '--output_type', 'TXT',
    ]
    subprocess.check_output(converter_args, universal_newlines=True)

def prepare_images(source_path, colmap_path, modalities, mosaicked, demosaick_fns):
    """
    Prepares images for COLMAP by demosaicking and converting them to the appropriate format.

    Args:
        source_path (str): Path to the source images.
        colmap_path (str): Path where to create the COLMAP directory and save prepared images.
        modalities (list): List of modalities to process.
        mosaicked (bool): Whether the images are mosaicked.
        demosaick_fns (dict): Dictionary of demosaicking functions for each modality.
    """
    for mod in modalities:
        demosaick_fn = demosaick_fns[mod]
        os.makedirs(os.path.join(colmap_path, "temp_images", "modalities", mod), exist_ok=True)
        file_names = os.listdir(os.path.join(source_path, "modalities", mod))
        file_names.sort()
        for i, file in enumerate(file_names):
            img, _ = read_frame(os.path.join(source_path, "modalities", mod, file))
            data_type = img.dtype
            if mosaicked:
                img = demosaick_fn(img)
            if len(img.shape) == 2:
                img = np.expand_dims(img, axis=-1)
            if img.shape[-1] != 1 and img.shape[-1] != 3:
                img = np.mean(img, axis=-1).astype(data_type)
            img = np.right_shift(img, 8).astype(np.uint8) if data_type == np.uint16 else img
            cv.imwrite(os.path.join(colmap_path, "temp_images", "modalities", mod, f"{i:04}.png"), img)

def compute_colmap_scale(colmap_path, reference_mod, modality_data):
    """
    Compute the scale factor of the COLMAP model with respect to the ground truth reference system.
    It works only for MMS-DATA.

    Args:
        colmap_path: (str): Path to the COLMAP model directory.
        reference_mod: (str): Reference modality to use for scale computation.
        modality_data: (dict): Dictionary containing camera parameters for each modality.
    """
    camera_matrix = modality_data[reference_mod]["original_camera_matrix"]

    dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_100)
    params = cv.aruco.DetectorParameters()

    img1 = cv.imread(os.path.join(colmap_path, "temp_images", "modalities", reference_mod, "0000.png"))
    img1, undistorted_camera_matrix = undistort_frame(img1, camera_matrix, modality_data[reference_mod]["dist_coeffs"])
    marker_corners_1, marker_ids_1, _ = cv.aruco.detectMarkers(img1, dictionary, parameters=params)
    img2 = cv.imread(os.path.join(colmap_path, "temp_images", "modalities", reference_mod, "0025.png"))
    img2, _ = undistort_frame(img2, camera_matrix, modality_data[reference_mod]["dist_coeffs"])
    marker_corners_2, marker_ids_2, _ = cv.aruco.detectMarkers(img2, dictionary, parameters=params)

    intrinsics = np.eye(4)
    intrinsics[:3, :3] = undistorted_camera_matrix

    marker_ids_1 = [x[0] for x in marker_ids_1]
    marker_ids_2 = [x[0] for x in marker_ids_2]

    points2D_1 = []
    points2D_2 = []
    for i, current_marker_ids in enumerate(marker_ids_1):
        if current_marker_ids in marker_ids_2:
            idx = marker_ids_2.index(current_marker_ids)
        else:
            continue
        points2D_1.extend(marker_corners_1[i][0])
        points2D_2.extend(marker_corners_2[idx][0])
    points2D_1 = np.array(points2D_1).T
    points2D_2 = np.array(points2D_2).T

    with open(os.path.join(colmap_path, "sparse", "0", "images.txt"), 'r') as f:
        lines = f.readlines()
    for i in range(0, len(lines), 2):
        if lines[i].startswith('#'):
            continue
        line = lines[i].strip().split()
        if line[9].replace("\\", "/").split("/")[-1][:-4] == "0000":
            q_vec = np.array([float(x) for x in line[1:5]])
            t_vec = np.array([float(x) for x in line[5:8]])
            R = qvec2rotmat(q_vec)
            w2c_1 = R
            w2c_1[:3, 3] = t_vec
        elif line[9].replace("\\", "/").split("/")[-1][:-4] == "0025":
            q_vec = np.array([float(x) for x in line[1:5]])
            t_vec = np.array([float(x) for x in line[5:8]])
            R = qvec2rotmat(q_vec)
            w2c_2 = R
            w2c_2[:3, 3] = t_vec

    camera_matrix_1 = intrinsics @ w2c_1
    camera_matrix_2 = intrinsics @ w2c_2

    points3D = cv.triangulatePoints(
        camera_matrix_1[:3],
        camera_matrix_2[:3],
        points2D_1,
        points2D_2
    ).T
    points3D = cv.convertPointsFromHomogeneous(points3D).reshape(-1, 4, 3)
    points_shifted = np.roll(points3D, 1, axis=1)
    dists = np.linalg.norm(points_shifted - points3D, axis=2).reshape(-1, 1)
    scale = np.mean(0.036 / dists)

    return scale

def extract_camera_model_from_colmap(colmap_path, modalities, modality_roi):
    """
    Extract camera model information from COLMAP's cameras.txt file.

    Args:
        colmap_path (str): Path to the COLMAP model directory.
        modalities (list): List of modalities to process.
        modality_roi (dict): Dictionary with information on how to crop modality frames.
    """
    info = {}
    with open(os.path.join(colmap_path, "sparse", "0", "cameras.txt"), 'r') as f:
        lines = f.readlines()
        lines = [line.strip().split() for line in lines if line[0] != '#']
    for line in lines:

        camera_id = int(line[0]) - 1
        camera_model = line[1]
        width, height = int(line[2]), int(line[3])

        if camera_model == "SIMPLE_PINHOLE":
            fx = float(line[4])
            fy = fx
            cx, cy = float(line[5]), float(line[6])
            k1, k2, k3, k4, k5, k6 = (0, 0, 0, 0, 0, 0)
            p1, p2 = (0, 0)
        elif camera_model == "SIMPLE_RADIAL":
            fx = float(line[4])
            fy = fx
            cx, cy = float(line[5]), float(line[6])
            k1 = float(line[7])
            k2, k3, k4, k5, k6 = (0, 0, 0, 0, 0)
            p1, p2 = (0, 0)
        elif camera_model == "RADIAL":
            fx = float(line[4])
            fy = fx
            cx, cy = float(line[5]), float(line[6])
            k1, k2 = float(line[7]), float(line[8])
            k3, k4, k5, k6 = (0, 0, 0, 0)
            p1, p2 = (0, 0)
        else:
            fx, fy = float(line[4]), float(line[5])
            cx, cy = float(line[6]), float(line[7])
            if camera_model == "PINHOLE":
                k1, k2, k3, k4, k5, k6 = (0, 0, 0, 0, 0, 0)
                p1, p2 = (0, 0)
            elif camera_model == "OPENCV":
                k1, k2 = float(line[8]), float(line[9])
                p1, p2 = float(line[10]), float(line[11])
                k3, k4, k5, k6 = (0, 0, 0, 0)
            elif camera_model == "FULL_OPENCV":
                k1, k2 = float(line[8]), float(line[9])
                p1, p2 = float(line[10]), float(line[11])
                k3, k4, k5, k6 = float(line[12]), float(line[13]), float(line[14]), float(line[15])
            else:
                raise ValueError(f"Camera model {camera_model} not supported.")

        mod = modalities[camera_id]
        camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        dist_coeffs = np.array([k1, k2, p1, p2, k3, k4, k5, k6])
        roi = (0, 0, width, height)
        if mod in modality_roi.keys():
            roi = modality_roi[mod]

        modality = {
            "original_camera_matrix": camera_matrix,
            "dist_coeffs": dist_coeffs,
            "original_roi": roi,
        }

        info[mod] = modality

    return info
