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
Preprocessing utilities for handling camera models, frames, and metadata.
"""

import os
import json

import numpy as np
import cv2 as cv
import h5py
import trimesh

from scipy.interpolate import RegularGridInterpolator

COLORS = {
    "infrared": [255, 0, 0],
    "rgb": [0, 255, 0],
    "polarization": [0, 0, 255],
    "multispectral": [255, 255, 0],
    "mono": [255, 0, 255],
}

def qvec2rotmat(q_vec):
    """Convert quaternion to rotation matrix."""
    n = 1.0 / np.sqrt(np.sum(q_vec**2))
    q_vec = q_vec * n
    qw, qx, qy, qz = q_vec
    rot = np.array([[1.0 - 2.0 * qy * qy - 2.0 * qz * qz, 2.0 * qx * qy - 2.0 * qz * qw, 2.0 * qx * qz + 2.0 * qy * qw, 0.0],
        [2.0 * qx * qy + 2.0 * qz * qw, 1.0 - 2.0 * qx * qx - 2.0 * qz * qz, 2.0 * qy * qz - 2.0 * qx * qw, 0.0],
        [2.0 * qx * qz - 2.0 * qy * qw, 2.0 * qy * qz + 2.0 * qx * qw, 1.0 - 2.0 * qx * qx - 2.0 * qy * qy, 0.0],
        [0.0, 0.0, 0.0, 1.0]])
    return rot

def generate_bounding_box(
        output_path,
        colmap_path,
        radius=0.5,
        scale=1.0,
        pointcloud_filtering=False,
        reorient_axis=False
):
    """
    Generate a bounding box from the COLMAP sparse point cloud.

    Args:
        output_path (str): Path to save the bounding box.
        colmap_path (str): Path to the COLMAP sparse point cloud.
        radius (float): Radius for clustering points.
                        Points closer than this distance are considered part of the same cluster.
        scale (float): Scale factor for the point cloud.
        pointcloud_filtering (bool): Whether to filter the point cloud by removing points not belonging to the region
                                     of interest.
        reorient_axis (bool): Whether to reorient the axis of the point cloud to be axis-aligned with reference system.
                              Working only with MMS-DATA dataset.
    """
    with open(os.path.join(colmap_path, 'sparse', "0", 'points3D.txt'), 'r') as file:
        lines = file.readlines()

    points = []
    for i, line in enumerate(lines):
        if i < 3 or i == len(lines) - 1:
            continue
        point = np.array(tuple(map(float, line.split()[1:4])))
        points.append(point)

    # Rescale colmap pointcloud
    pointcloud = np.array(points)
    pointcloud = pointcloud * scale
    # trimesh.PointCloud(pointcloud).export(os.path.join(output_path, 'sparse_points_gt_scale.ply'))

    # Extract ROI pointcloud
    clusters = list(trimesh.grouping.clusters(pointcloud, radius=radius))
    clusters = [x for x in clusters if x.shape[0] > 100]
    idxs = np.argsort([x.shape[0] for x in clusters])[::-1][:2]
    stds = [np.mean(np.std(pointcloud[clusters[x]], axis=0)) for x in idxs]
    idx = idxs[np.argmin(stds)]
    pointcloud = pointcloud[clusters[idx]]
    # trimesh.PointCloud(pointcloud).export(os.path.join(output_path, 'selected_cluster_gt_scale.ply'))

    if pointcloud_filtering:
        # Clean ROI pointcloud
        clusters = list(trimesh.grouping.clusters(pointcloud, radius=radius * 0.2))
        idx1, idx2, idx3 = np.argsort([x.shape[0] for x in clusters])[::-1][:3]
        selected = np.concatenate([clusters[idx1], clusters[idx2], clusters[idx3]], axis=0)
        pointcloud = pointcloud[selected]
        # trimesh.PointCloud(pointcloud).export(os.path.join(output_path, 'clean_cluster_gt_scale.ply'))

    # Fit pointcloud to bounding box
    ab_min = np.min(pointcloud, axis=0)
    ab_max = np.max(pointcloud, axis=0)

    center = (ab_max + ab_min) / 2
    rad = np.max(np.linalg.norm(pointcloud - center, axis=-1)) * 1.

    transform1 = np.eye(4)
    transform1[:3, :3] *= rad
    transform1[:3, 3] = center
    transform1 = np.linalg.inv(transform1)

    pointcloud = (transform1 @ np.concatenate([pointcloud, np.ones((pointcloud.shape[0], 1))], axis=-1).T).T[:, :3]
    # trimesh.PointCloud(pointcloud).export(os.path.join(output_path, 'sparse_points_interest.ply'))

    transform2 = np.eye(4)
    if reorient_axis:
        # Orient pointcloud to be orthogonal to the ground
        mask = np.abs(pointcloud) > 0.5
        mask = np.any(mask, axis=-1)
        clusters = list(trimesh.grouping.clusters(pointcloud[mask], radius=radius * rad * 0.20))
        idx1, idx2 = np.argsort([x.shape[0] for x in clusters])[::-1][:2]
        selected = np.concatenate([clusters[idx1], clusters[idx2]], axis=0)

        # trimesh.PointCloud(pointcloud[mask][selected]).export(os.path.join(output_path, 'checkerboard.ply'))

        rotation = trimesh.bounds.oriented_bounds(pointcloud[mask][selected])[0]
        transform2 = rotation
        transform2[:3, 3] = 0
        # Permute axis
        permutation = np.array([[0, 0, -1, 0],
                                [0, 1, 0, 0],
                                [1, 0, 0, 0],
                                [0, 0, 0, 1]])
        transform2 = permutation @ transform2
        pointcloud = (transform2 @ np.concatenate([pointcloud, np.ones((pointcloud.shape[0], 1))], axis=-1).T).T[:, :3]

    # Center pointcloud
    ab_min = np.min(pointcloud, axis=0)
    ab_max = np.max(pointcloud, axis=0)
    center = (ab_max + ab_min) / 2
    transform3 = np.eye(4)
    transform3[:3, 3] = -center
    pointcloud = (transform3 @ np.concatenate([pointcloud, np.ones((pointcloud.shape[0], 1))], axis=-1).T).T[:, :3]

    gt2w = transform3 @ transform2 @ transform1

    trimesh.PointCloud(pointcloud).export(os.path.join(output_path, 'pointcloud.ply'))
    return gt2w, [ab_min.tolist(), ab_max.tolist()]

def extract_camera_model(modalities, calibration, modality_roi):
    """
    Extract camera model parameters from calibration data.

    Args:
        modalities (list): List of modalities.
        calibration (dict): Calibration data for each modality.
        modality_roi (dict): Region of interest for each modality.
    """
    info = {}
    for mod in modalities:
        data = calibration[mod]
        width, height = data["width"], data["height"]
        fx, fy = data["fx"], data["fy"]
        cx, cy = data["cx"], data["cy"]
        k1, k2, p1, p2, k3, k4 = data["distortion_params"]

        camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        dist_coeffs = np.array([k1, k2, p1, p2, k3, k4, 0, 0])
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

def read_frame(path):
    """
    Read a frame from a file. The file format is determined by the file extension.
    """
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

def write_frame(path, frame, **kwargs):
    """
    Write a frame to a file. The file format is determined by the file extension.
    """
    file_format = path.strip().split(".")[-1]
    n_channels = frame.shape[-1] if len(frame.shape) == 3 else 1
    if file_format in ["jpg", "JPG", "png", "PNG"] and n_channels in (1, 3):
        cv.imwrite(path, frame)
    elif file_format == "h5":
        with h5py.File(path, 'w') as f:
            for i in range(n_channels):
                f.create_dataset(kwargs["keys"][i], data=frame[..., i])
    else:
        path = path[:-4] + ".npy"
        np.save(path, frame)

def multispectral_demosaicking(frame):
    """
    Demosaicking function for multispectral images captured by the SILIOS CMS-C1 camera.
    """
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

def multispectral_sorting(frame):
    """
    Reorder the channels of the multispectral image captured by the SILIOS CMS-C1 camera to be
    """
    frame = frame[:,:,[5, 4, 3, 6, 0, 1, 2, 8, 7]]
    return frame

def crop_camera_matrix(modality_data):
    """
    Correct the camera matrix based on the cropping defined in the modality data.
    """
    cropped_camera_matrix = modality_data["original_camera_matrix"].copy()
    x, y, _, _ = modality_data["original_roi"]
    cropped_camera_matrix[0, 2] -= x
    cropped_camera_matrix[1, 2] -= y
    modality_data["cropped_camera_matrix"] = cropped_camera_matrix
    modality_data["current_camera_matrix"] = cropped_camera_matrix.copy()
    modality_data["current_roi"] = modality_data["original_roi"]
    return modality_data

def undistort_camera_matrix(modality_data):
    """
    Compute the undistorted camera matrix based on the original camera matrix and distortion coefficients.
    """
    camera_matrix = modality_data["current_camera_matrix"].copy()
    dist_coeffs = modality_data["dist_coeffs"]
    _, _, w, h = modality_data["current_roi"]
    undistorted_camera_matrix, undistorted_roi = cv.getOptimalNewCameraMatrix(
        camera_matrix,
        dist_coeffs,
        imageSize=(w, h),
        alpha=1
    )
    modality_data["undistorted_camera_matrix"] = undistorted_camera_matrix
    modality_data["undistorted_roi"] = undistorted_roi
    current_camera_matrix = undistorted_camera_matrix.copy()
    current_camera_matrix[0, 2] -= undistorted_roi[0]
    current_camera_matrix[1, 2] -= undistorted_roi[1]
    modality_data["current_camera_matrix"] = current_camera_matrix
    modality_data["current_roi"] = undistorted_roi
    return modality_data

def scale_camera_matrix(modality_data, scale=1.0):
    """
    Scale the camera matrix based on the given scale factor.
    """
    camera_matrix = modality_data["current_camera_matrix"].copy()
    _, _, w, h = modality_data["current_roi"]
    camera_matrix[0, 0] *= scale
    camera_matrix[1, 1] *= scale
    camera_matrix[0, 2] *= scale
    camera_matrix[1, 2] *= scale
    w = round(w * scale)
    h = round(h * scale)
    modality_data["current_camera_matrix"] = camera_matrix
    modality_data["current_roi"] = (0, 0, w, h)
    return modality_data

def process_camera_matrix(modalities, modality_data, undistort=False, scale=1.0):
    """
    Process the camera matrix for each modality based on the given parameters.

    Args:
        modalities (list): List of modalities.
        modality_data (dict): Dictionary containing camera data for each modality.
        undistort (bool): Whether to undistort the camera matrix.
        scale (float): Scaling factor for the camera matrix.

    Returns:
        dict: Updated modality data with processed camera matrices.
    """
    for mod in modalities:
        modality_data[mod]["current_camera_matrix"] = modality_data[mod]["original_camera_matrix"].copy()
        modality_data[mod]["current_roi"] = modality_data[mod]["original_roi"]

        modality_data[mod] = crop_camera_matrix(modality_data[mod])
        if undistort:
            modality_data[mod] = undistort_camera_matrix(modality_data[mod])
        if scale != 1:
            modality_data[mod] = scale_camera_matrix(modality_data[mod], scale=scale)

    return modality_data

def adjust_frame(frame, modality_data, undistort=False, scale=1.0, demosaick=False, demosaicking_fn = lambda x: x):
    """
    Adjust the frame based on the modality data, including cropping, undistorting, and scaling.

    Args:
        frame (numpy.ndarray): The input frame to be adjusted.
        modality_data (dict): Dictionary containing camera data for the modality.
        undistort (bool): Whether to undistort the frame.
        scale (float): Scaling factor for the frame.
        demosaick (bool): Whether to perform demosaicking on the frame.
        demosaicking_fn (function): Function to perform demosaicking.

    Returns:
        numpy.ndarray: The adjusted frame.
    """
    x, y, w, h = modality_data["original_roi"]
    frame = frame[y:y + h, x:x + w]
    if demosaick:
        frame = demosaicking_fn(frame)
    if undistort:
        frame = cv.undistort(
            src=frame,
            cameraMatrix=modality_data["cropped_camera_matrix"],
            distCoeffs=modality_data["dist_coeffs"],
            newCameraMatrix=modality_data["undistorted_camera_matrix"]
        )
        x, y, w, h = modality_data["undistorted_roi"]
        frame = frame[y:y + h, x:x + w]
    if scale != 1.0:
        frame = cv.resize(frame, (0, 0), fx=scale, fy=scale, interpolation=cv.INTER_AREA)
    return frame

def undistort_frame(frame, distorted_camera_matrix, dist_coeffs):
    """
    Perform undistortion on the input frame using the given camera matrix and distortion coefficients.

    Args:
        frame (numpy.ndarray): The input frame to be undistorted.
        distorted_camera_matrix (numpy.ndarray): The camera matrix for the distorted image.
        dist_coeffs (numpy.ndarray): The distortion coefficients for the camera.

    Returns:
        tuple: The undistorted frame and the new camera matrix.
    """
    h, w, _ = frame.shape
    undistorted_camera_matrix, undistorted_roi = cv.getOptimalNewCameraMatrix(
        cameraMatrix=distorted_camera_matrix,
        distCoeffs=dist_coeffs,
        imageSize=(w, h),
        alpha=1
    )
    frame = cv.undistort(frame, distorted_camera_matrix, dist_coeffs, newCameraMatrix=undistorted_camera_matrix)
    x, y, w, h = undistorted_roi
    frame = frame[y:y + h, x:x + w]
    undistorted_camera_matrix[0, 2] -= undistorted_roi[0]
    undistorted_camera_matrix[1, 2] -= undistorted_roi[1]
    return frame, undistorted_camera_matrix

def process_frames(source_path, output_path, modalities, modality_data, undistort, scale, demosaick, demosaick_fns):
    """
    Process the frames for each modality, including cropping, undistorting, and scaling.

    Args:
        source_path (str): Path to the source images.
        output_path (str): Path to save the processed images.
        modalities (list): List of modalities.
        modality_data (dict): Dictionary containing camera data for each modality.
        undistort (bool): Whether to undistort the frames.
        scale (float): Scaling factor for the frames.
        demosaick (bool): Whether to perform demosaicking on the frames.
        demosaick_fns (dict): Dictionary containing demosaicking functions for each modality.

    Returns:
        dict: Dictionary containing the number of channels for each modality.
    """
    source_path = os.path.join(source_path, "modalities")
    output_path = os.path.join(output_path, "modalities")
    channels_per_mod = {}
    for mod in modalities:

        os.makedirs(os.path.join(output_path, mod), exist_ok=True)

        mod_data = modality_data[mod]
        demosaick_fn = demosaick_fns[mod]

        frame_names = os.listdir(os.path.join(source_path, mod))
        frame_names.sort()

        for i, frame_name in enumerate(frame_names):

            extension = frame_name.split(".")[-1]
            frame, additional_data = read_frame(os.path.join(source_path, mod, frame_name))
            frame = adjust_frame(frame, mod_data, undistort, scale, demosaick, demosaick_fn)
            write_frame(os.path.join(output_path, mod, f"{i:04}.{extension}"), frame, **additional_data) \
                if isinstance(additional_data, dict) \
                else write_frame(os.path.join(output_path, mod, f"{i:04}.{extension}"), frame)
            if additional_data is not None:
                # hdf5 files
                channels = -1
            elif len(frame.shape) == 3:
                channels = frame.shape[-1]
            else:
                channels = 1
            channels_per_mod[mod] = channels
    return channels_per_mod

def build_metadata(
        output_path,
        colmap_path,
        modalities,
        modality_data,
        gt2world,
        bbox,
        channels_per_mod,
        calibration=None,
        scale=1.0,
        undistorted=False,
        mosaicked=False,
        mosaick_patterns=None
):
    """
    Generate metadata json file for the processed scene.
    It contains information about the camera models, camera poses, frames, and bounding box.

    Args:
        output_path (str): Path to save the metadata file.
        colmap_path (str): Path to the COLMAP model.
        modalities (list): List of modalities.
        modality_data (dict): Dictionary containing camera data for each modality.
        gt2world (numpy.ndarray): Transformation matrix from ground truth to COLMAP world coordinates.
        bbox (list): Bounding box coordinates.
        channels_per_mod (dict): Dictionary containing the number of channels for each modality.
        calibration (dict, optional): Calibration data for each modality. Defaults to None.
        scale (float, optional): Scale factor for the point cloud. Defaults to 1.0.
        undistorted (bool, optional): Whether the images are undistorted. Defaults to False.
        mosaicked (bool, optional): Whether the images are mosaicked. Defaults to False.
        mosaick_patterns (dict, optional): Mosaicking patterns for each modality. Defaults to None.
    """
    metadata = {}

    metadata["undistorted"] = undistorted
    metadata["raw"] = mosaicked
    metadata["pixel_offset"] = 0.0

    metadata["scene_box"] = {
        "aabb": bbox,
        "collider_type": "sphere",
        "radius": 1.0,
    }

    metadata["worldtogt"] = np.linalg.inv(gt2world).tolist()

    modalities_data = {}

    with open(os.path.join(colmap_path, 'sparse', '0', 'images.txt'), 'r') as file:
        pose_lines = file.readlines()
        pose_lines = [line.strip().split() for line in pose_lines if line[0] != '#']

    for mod in modalities:
        data = modality_data[mod]
        _, _, w, h = modality_data[mod]["current_roi"]
        camera_matrix = data["current_camera_matrix"]
        fx = camera_matrix[0, 0]
        fy = camera_matrix[1, 1]
        cx = camera_matrix[0, 2]
        cy = camera_matrix[1, 2]
        distortion_params = data["dist_coeffs"][:6]
        modality = {
            "camera_model": "PINHOLE",
            "width": w,
            "height": h,
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
        }

        if not undistorted:
            modality["camera_model"] = "OPENCV"
            modality["distortion_params"] = distortion_params.tolist()

        if mosaicked:
            modality["mosaick_pattern"] = mosaick_patterns[mod]

        camera2reference = np.eye(4)
        if calibration is not None:
            camera2reference = calibration[mod]["camera2reference"]

        poses = []
        for i in range(0, len(pose_lines), 2):
            line = pose_lines[i]
            if calibration is None and int(line[8]) != modalities.index(mod)+1:
                continue
            q_vec = np.array([float(x) for x in line[1:5]])
            t_vec = np.array([float(x) for x in line[5:8]])
            R = qvec2rotmat(q_vec)

            # Original camera poses (scaled accordingly to colmap2world_scale)
            gt2c = R
            gt2c[:3, 3] = t_vec * scale
            c2gt = np.linalg.inv(gt2c)

            # Adjust with respect to reference camera
            c2gt = c2gt @ camera2reference

            # Adjust with respect to gt2w transform
            t_vec = gt2world @ c2gt[:4, 3]
            rot = (gt2world[:3, :3] @ c2gt[:3, :3]) / np.linalg.norm(gt2world[:3, 0])
            c2w = np.eye(4)
            c2w[:4, 3] = t_vec
            c2w[:3, :3] = rot

            # Change coordinate system
            rdf2rub = np.eye(4)
            rdf2rub[1, 1] = -1
            rdf2rub[2, 2] = -1

            c2w = c2w @ rdf2rub

            frame_id = int(line[9].replace("\\", "/").split("/")[-1][:-4])
            frame_name = line[9].replace("\\", "/").split("/")[-1][:-4]
            frame_name = f"{int(frame_name):04}"

            if channels_per_mod[mod] == 1 or channels_per_mod[mod] == 3:
                frame_name += ".png"
            elif channels_per_mod[mod] == -1:
                frame_name += ".h5"
            else:
                frame_name += ".npy"
            poses.append({
                "frame_id": frame_id,
                "file_name": frame_name,
                "camtoworld": c2w[:3, :].tolist(),
            })

        modality['frames'] = poses
        modalities_data[mod] = modality

    metadata["modalities"] = modalities_data
    with open(os.path.join(output_path, 'meta_data.json'), 'w') as outfile:
        json.dump(metadata, outfile, indent=4)

def check_cameras(out_folder, modalities):
    """
    Export a point cloud of the camera poses in the scene.
    """
    # Read image pose.
    metadata_path = os.path.join(out_folder, "meta_data.json")
    points = []
    colors = []

    with open(metadata_path, 'r') as file:
        metadata = json.load(file)

    for mod in modalities:
        for frame in metadata['modalities'][mod]['frames']:
            c2w = frame['camtoworld']
            pose = c2w @ np.array([0,0,0,1])
            # dirs = c2w @ np.array([0,0,1,1])
            points.append(pose[:3].tolist())
            # points.append(dirs[:3].tolist())
            colors.append(COLORS.get(mod, [0, 0, 0]))
            # colors.append([255,0,0])

    trimesh.PointCloud(np.array(points), colors=colors).export(os.path.join(out_folder, 'camera_poses.ply'))
