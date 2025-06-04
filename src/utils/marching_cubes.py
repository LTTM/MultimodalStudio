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
Function to extract a mesh from a signed distance function (SDF) using the marching cubes algorithm.
"""

import numpy as np
import torch
import trimesh
from skimage import measure

avg_pool_3d = torch.nn.AvgPool3d(2, stride=2)
upsample = torch.nn.Upsample(scale_factor=2, mode="nearest")
max_pool_3d = torch.nn.MaxPool3d(3, stride=1, padding=1)

@torch.no_grad()
def get_surface_sliding(
    sdf_fn,
    resolution=512,
    bounding_box_min=(-1.0, -1.0, -1.0),
    bounding_box_max=(1.0, 1.0, 1.0),
    return_mesh=False,
    coarse_mask=None,
    output_path: str = None,
):
    """
    Extracts a mesh from the signed distance function (SDF) using the marching cubes algorithm.

    Args:
        sdf_fn: Signed distance function to evaluate.
        resolution: Resolution of the mesh.
        bounding_box_min: Minimum coordinates of the bounding box.
        bounding_box_max: Maximum coordinates of the bounding box.
        return_mesh: If True, returns the mesh instead of saving it to a file.
        coarse_mask: Optional mask to filter the points.
        output_path: Path to save the mesh file.

    Returns:
        If return_mesh is True, returns the extracted mesh as a trimesh object.
    """
    assert resolution % 256 == 0
    if coarse_mask is not None:
        # we need to permute here as pytorch's grid_sample use (z, y, x)
        coarse_mask = coarse_mask.permute(2, 1, 0)[None, None].cuda().float()

    resN = resolution
    cropN = 256
    level = 0
    N = resN // cropN

    grid_min = bounding_box_min
    grid_max = bounding_box_max
    xs = np.linspace(grid_min[0], grid_max[0], N + 1)
    ys = np.linspace(grid_min[1], grid_max[1], N + 1)
    zs = np.linspace(grid_min[2], grid_max[2], N + 1)

    meshes = []
    for i in range(N):
        for j in range(N):
            for k in range(N):
                x_min, x_max = xs[i], xs[i + 1]
                y_min, y_max = ys[j], ys[j + 1]
                z_min, z_max = zs[k], zs[k + 1]

                x = np.linspace(x_min, x_max, cropN)
                y = np.linspace(y_min, y_max, cropN)
                z = np.linspace(z_min, z_max, cropN)

                xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
                points = torch.tensor(np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T, dtype=torch.float16).cuda()

                def evaluate(points):
                    z = []
                    for _, pnts in enumerate(torch.split(points, 100000, dim=0)):
                        z.append(sdf_fn(pnts))
                    z = torch.cat(z, axis=0)
                    return z

                # construct point pyramids
                points = points.reshape(cropN, cropN, cropN, 3).permute(3, 0, 1, 2)
                if coarse_mask is not None:
                    # breakpoint()
                    points_tmp = points.permute(1, 2, 3, 0)[None].cuda()
                    current_mask = torch.nn.functional.grid_sample(coarse_mask, points_tmp)
                    current_mask = (current_mask > 0.0).cpu().numpy()[0, 0]
                else:
                    current_mask = None

                points_pyramid = [points]
                for _ in range(3):
                    points = avg_pool_3d(points[None])[0]
                    points_pyramid.append(points)
                points_pyramid = points_pyramid[::-1]

                # evalute pyramid with mask
                mask = None
                threshold = 2 * (x_max - x_min) / cropN * 8
                for pid, pts in enumerate(points_pyramid):
                    coarse_N = pts.shape[-1]
                    pts = pts.reshape(3, -1).permute(1, 0).contiguous()

                    if mask is None:
                        # only evaluate
                        if coarse_mask is not None:
                            pts_sdf = torch.ones_like(pts[:, 1])
                            valid_mask = (
                                torch.nn.functional.grid_sample(coarse_mask, pts[None, None, None])[0, 0, 0, 0] > 0
                            )
                            if valid_mask.any():
                                pts_sdf[valid_mask] = evaluate(pts[valid_mask].contiguous())
                        else:
                            pts_sdf = evaluate(pts)
                    else:
                        mask = mask.reshape(-1)
                        pts_to_eval = pts[mask]

                        if pts_to_eval.shape[0] > 0:
                            pts_sdf_eval = evaluate(pts_to_eval.contiguous())
                            pts_sdf[mask] = pts_sdf_eval
                        # print("ratio", pts_to_eval.shape[0] / pts.shape[0])

                    if pid < 3:
                        # update mask
                        mask = torch.abs(pts_sdf) < threshold
                        mask = mask.reshape(coarse_N, coarse_N, coarse_N)[None, None]
                        mask = upsample(mask.float()).bool()

                        pts_sdf = pts_sdf.reshape(coarse_N, coarse_N, coarse_N)[None, None]
                        pts_sdf = upsample(pts_sdf)
                        pts_sdf = pts_sdf.reshape(-1)

                    threshold /= 2.0

                z = pts_sdf.detach().cpu().numpy()

                # skip if no surface found
                if current_mask is not None:
                    valid_z = z.reshape(cropN, cropN, cropN)[current_mask]
                    if valid_z.shape[0] <= 0 or (np.min(valid_z) > level or np.max(valid_z) < level):
                        continue

                if not (np.min(z) > level or np.max(z) < level):
                    z = z.astype(np.float32)
                    verts, faces, normals, _ = measure.marching_cubes(
                        volume=z.reshape(cropN, cropN, cropN),  # .transpose([1, 0, 2]),
                        level=level,
                        spacing=(
                            (x_max - x_min) / (cropN - 1),
                            (y_max - y_min) / (cropN - 1),
                            (z_max - z_min) / (cropN - 1),
                        ),
                        mask=current_mask,
                    )
                    # print(np.array([x_min, y_min, z_min]))
                    # print(verts.min(), verts.max())
                    verts = verts + np.array([x_min, y_min, z_min])
                    # print(verts.min(), verts.max())

                    meshcrop = trimesh.Trimesh(verts, faces, normals)
                    # meshcrop.export(f"{i}_{j}_{k}.ply")
                    meshes.append(meshcrop)

    combined = trimesh.util.concatenate(meshes)

    if return_mesh:
        return combined
    else:
        filename = str(output_path)
        combined.merge_vertices(digits_vertex=6)
        combined.export(filename)
