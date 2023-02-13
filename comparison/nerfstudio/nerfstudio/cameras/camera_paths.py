# Copyright 2022 The Nerfstudio Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Code for camera paths.
"""

from typing import Any, Dict, Optional, Tuple

import torch

import nerfstudio.utils.poses as pose_utils
from nerfstudio.cameras import camera_utils
from nerfstudio.cameras.camera_utils import get_interpolated_poses_many
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.viewer.server.utils import three_js_perspective_camera_focal_length


def get_interpolated_camera_path(cameras: Cameras, steps: int) -> Cameras:
    """Generate a camera path between two cameras.

    Args:
        cameras: Cameras object containing intrinsics of all cameras.
        steps: The number of steps to interpolate between the two cameras.

    Returns:
        A new set of cameras along a path.
    """
    Ks = cameras.get_intrinsics_matrices().cpu().numpy()
    poses = cameras.camera_to_worlds().cpu().numpy()
    poses, Ks = get_interpolated_poses_many(poses, Ks, steps_per_transition=steps)

    cameras = Cameras(fx=Ks[:, 0, 0], fy=Ks[:, 1, 1], cx=Ks[0, 0, 2], cy=Ks[0, 1, 2], camera_to_worlds=poses)
    return cameras


def get_spiral_path(
    camera: Cameras,
    steps: int = 30,
    radius: Optional[float] = None,
    radiuses: Optional[Tuple[float]] = None,
    rots: int = 2,
    zrate: float = 0.5,
) -> Cameras:
    """
    Returns a list of camera in a sprial trajectory.

    Args:
        camera: The camera to start the spiral from.
        steps: The number of cameras in the generated path.
        radius: The radius of the spiral for all xyz directions.
        radiuses: The list of radii for the spiral in xyz directions.
        rots: The number of rotations to apply to the camera.
        zrate: How much to change the z position of the camera.

    Returns:
        A spiral camera path.
    """

    assert radius is not None or radiuses is not None, "Either radius or radiuses must be specified."
    assert camera.ndim == 1, "We assume only one batch dim here"
    if radius is not None and radiuses is None:
        rad = torch.tensor([radius] * 3, device=camera.device)
    elif radiuses is not None and radius is None:
        rad = torch.tensor(radiuses, device=camera.device)
    else:
        raise ValueError("Only one of radius or radiuses must be specified.")

    up = camera.camera_to_worlds[0, :3, 2]  # scene is z up
    focal = torch.min(camera.fx[0], camera.fy[0])
    target = torch.tensor([0, 0, -focal], device=camera.device)  # camera looking in -z direction

    c2w = camera.camera_to_worlds[0]
    c2wh_global = pose_utils.to4x4(c2w)

    local_c2whs = []
    for theta in torch.linspace(0.0, 2.0 * torch.pi * rots, steps + 1)[:-1]:
        center = (
            torch.tensor([torch.cos(theta), -torch.sin(theta), -torch.sin(theta * zrate)], device=camera.device) * rad
        )
        lookat = center - target
        c2w = camera_utils.viewmatrix(lookat, up, center)
        c2wh = pose_utils.to4x4(c2w)
        local_c2whs.append(c2wh)

    new_c2ws = []
    for local_c2wh in local_c2whs:
        c2wh = torch.matmul(c2wh_global, local_c2wh)
        new_c2ws.append(c2wh[:3, :4])
    new_c2ws = torch.stack(new_c2ws, dim=0)

    return Cameras(
        fx=camera.fx[0],
        fy=camera.fy[0],
        cx=camera.cx[0],
        cy=camera.cy[0],
        camera_to_worlds=new_c2ws,
    )

import numpy as np
from scipy.interpolate import UnivariateSpline
def smooth_vec(vec, time, s):
    smoothed = np.zeros_like(vec)
    for i in range(vec.shape[1]):
        spl = UnivariateSpline(time, vec[..., i])
        spl.set_smoothing_factor(s)
        smoothed[..., i] = spl(time)
    return smoothed

def smooth_poses_spline(poses, st=0.3, sr=4):
    poses[:, 0] = -poses[:, 0]
    posesnp = poses.numpy()
    scale = 2e-2 / np.median(np.linalg.norm(posesnp[1:, :3, 3] - posesnp[:-1, :3, 3], axis=-1))
    posesnp[:, :3, 3] *= scale
    time = np.linspace(0, 1, len(posesnp))
    
    t = smooth_vec(posesnp[..., 3], time, st)
    z = smooth_vec(posesnp[..., 2], time, sr)
    z /= np.linalg.norm(z, axis=-1)[:, None]
    y_ = smooth_vec(posesnp[..., 1], time, sr)
    x = np.cross(z, y_)
    x /= np.linalg.norm(x, axis=-1)[:, None]
    y = np.cross(x, z)

    smooth_posesnp = np.stack([x, y, z, t], -1)
    poses[:, 0] = -poses[:, 0]
    smooth_posesnp[:, 0] = -smooth_posesnp[:, 0]
    smooth_posesnp[:, :3, 3] /= scale
    return torch.Tensor(smooth_posesnp.astype(np.float32))

def get_path_from_json(meta: Dict[str, Any], config, spline) -> Cameras:
    """Takes a camera path dictionary and returns a trajectory as a Camera instance.

    Args:
        camera_path: A dictionary of the camera path information coming from the viewer.

    Returns:
        A Cameras instance with the camera path.
    """

    # image_height = camera_path["render_height"]
    # image_width = camera_path["render_width"]
    # if "camera_type" in camera_path:
    #     camera_type = camera_path["camera_type"]
    # else:
    #     camera_type = "perspective"

    # if "camera_type" not in camera_path:
    #     camera_type = CameraType.PERSPECTIVE
    # elif camera_path["camera_type"] == "fisheye":
    #     camera_type = CameraType.FISHEYE
    # elif camera_path["camera_type"] == "equirectangular":
    #     camera_type = CameraType.EQUIRECTANGULAR
    # else:
    #     camera_type = CameraType.PERSPECTIVE

    # c2ws = []
    # fxs = []
    # fys = []
    # for camera in camera_path["camera_path"]:
    #     # pose
    #     c2w = torch.tensor(camera["camera_to_world"]).view(4, 4)[:3]
    #     c2ws.append(c2w)
    #     if camera_type == CameraType.EQUIRECTANGULAR:
    #         fxs.append(image_width / 2)
    #         fys.append(image_height)
    #     else:
    #         # field of view
    #         fov = camera["fov"]
    #         focal_length = three_js_perspective_camera_focal_length(fov, image_height)
    #         fxs.append(focal_length)
    #         fys.append(focal_length)

    # camera_to_worlds = torch.stack(c2ws, dim=0)
    # fx = torch.tensor(fxs)
    # fy = torch.tensor(fys)
    # return Cameras(
    #     fx=fx,
    #     fy=fy,
    #     cx=image_width / 2,
    #     cy=image_height / 2,
    #     camera_to_worlds=camera_to_worlds,
    #     camera_type=camera_type,
    # )

    image_filenames = []
    mask_filenames = []
    poses = []

    fx_fixed = "fl_x" in meta
    fy_fixed = "fl_y" in meta
    cx_fixed = "cx" in meta
    cy_fixed = "cy" in meta
    height_fixed = "h" in meta
    width_fixed = "w" in meta
    distort_fixed = False
    for distort_key in ["k1", "k2", "k3", "p1", "p2"]:
        if distort_key in meta:
            distort_fixed = True
            break
    fx = []
    fy = []
    cx = []
    cy = []
    height = []
    width = []
    distort = []

    import os
    import numpy as np
    from pathlib import Path, PurePath

    meta["frames"] = sorted(meta["frames"], key=lambda d: d['file_path'])
    for frame in meta["frames"]:
        filepath = PurePath(f"images/{os.path.basename(frame['file_path'])}")
        fname = filepath
        # if not fname.exists():
        #     num_skipped_image_filenames += 1
        #     continue

        image_filenames.append(fname)
        poses.append(np.array(frame["transform_matrix"]))
    assert (
        len(image_filenames) != 0
    ), """
    No image files found. 
    You should check the file_paths in the transforms.json file to make sure they are correct.
    """
    assert len(mask_filenames) == 0 or (
        len(mask_filenames) == len(image_filenames)
    ), """
    Different number of image and mask filenames.
    You should check that mask_path is specified for every frame (or zero frames) in transforms.json.
    """   
    import os
    i_train, i_eval = [], []
    idx = 0
    for file_path in image_filenames:
        index = int(os.path.splitext(os.path.basename(file_path))[0])

        if index % 10:
            i_train.append(idx)
        else:
            i_eval.append(idx)
        idx += 1
    i_train = np.array(i_train)
    i_eval = np.array(i_eval)
        
    if spline:
        indices = np.arange(len(image_filenames))
    else:
        indices = i_eval

    if "orientation_override" in meta:
        orientation_method = meta["orientation_override"]
    else:
        orientation_method = config.orientation_method

    poses = torch.from_numpy(np.array(poses).astype(np.float32))
    poses, transform_matrix = camera_utils.auto_orient_and_center_poses(
        poses,
        method=orientation_method,
        center_poses=config.center_poses,
    )

    # Scale poses
    scale_factor = 1.0
    if config.auto_scale_poses:
        scale_factor /= float(torch.max(torch.abs(poses[:, :3, 3])))
    scale_factor *= config.scale_factor

    poses[:, :3, 3] *= scale_factor

    # Choose image_filenames and poses based on split, but after auto orient and scaling the poses.
    image_filenames = [image_filenames[i] for i in indices]
    mask_filenames = [mask_filenames[i] for i in indices] if len(mask_filenames) > 0 else []
    poses = poses[indices]
    
    if spline:
        poses = smooth_poses_spline(poses)

    # in x,y,z order
    # assumes that the scene is centered at the origin

    from nerfstudio.cameras.cameras import CAMERA_MODEL_TO_TYPE, Cameras, CameraType
    if "camera_model" in meta:
        camera_type = CAMERA_MODEL_TO_TYPE[meta["camera_model"]]
    else:
        camera_type = CameraType.PERSPECTIVE

    idx_tensor = torch.tensor(indices, dtype=torch.long)
    fx = float(meta["fl_x"]) if fx_fixed else torch.tensor(fx, dtype=torch.float32)[idx_tensor]
    fy = float(meta["fl_y"]) if fy_fixed else torch.tensor(fy, dtype=torch.float32)[idx_tensor]
    cx = float(meta["cx"]) if cx_fixed else torch.tensor(cx, dtype=torch.float32)[idx_tensor]
    cy = float(meta["cy"]) if cy_fixed else torch.tensor(cy, dtype=torch.float32)[idx_tensor]
    height = int(meta["h"]) if height_fixed else torch.tensor(height, dtype=torch.int32)[idx_tensor]
    width = int(meta["w"]) if width_fixed else torch.tensor(width, dtype=torch.int32)[idx_tensor]
    if distort_fixed:
        distortion_params = camera_utils.get_distortion_params(
            k1=float(meta["k1"]) if "k1" in meta else 0.0,
            k2=float(meta["k2"]) if "k2" in meta else 0.0,
            k3=float(meta["k3"]) if "k3" in meta else 0.0,
            k4=float(meta["k4"]) if "k4" in meta else 0.0,
            p1=float(meta["p1"]) if "p1" in meta else 0.0,
            p2=float(meta["p2"]) if "p2" in meta else 0.0,
        )
    else:
        distortion_params = torch.stack(distort, dim=0)[idx_tensor]

    return Cameras(
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        distortion_params=distortion_params,
        height=height,
        width=width,
        camera_to_worlds=poses[:, :3, :4],
        camera_type=camera_type,
    )