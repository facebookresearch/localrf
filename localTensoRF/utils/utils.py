# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2022 Anpei Chen

import cv2
import numpy as np
import scipy.signal
import torch
import torchvision.transforms as T
from PIL import Image
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline

def pts2px(pts, f, center):
    pts[..., 1] = -pts[..., 1]
    pts[..., 2] = -pts[..., 2]
    pts[..., 2] = torch.clip(pts[..., 2].clone(), min=1e-6)
    return torch.stack(
        [pts[..., 0] / pts[..., 2] * f + center[0] - 0.5, pts[..., 1] / pts[..., 2] * f + center[1] - 0.5], 
        dim=-1)
    
def inverse_pose(pose):
    pose_inv = torch.zeros_like(pose)
    pose_inv[:, :3, :3] = torch.transpose(pose[:, :3, :3], 1, 2)
    pose_inv[:, :3, 3] = -torch.bmm(pose_inv[:, :3, :3].clone(), pose[:, :3, 3:])[..., 0]
    return pose_inv

def get_cam2cams(cam2worlds, indices, offset):
    idx = torch.clamp(indices + offset, 0, len(cam2worlds) - 1)
    world2cam = inverse_pose(cam2worlds[idx])
    cam2cams = torch.zeros_like(world2cam)
    cam2cams[:, :3, :3] = torch.bmm(world2cam[:, :3, :3], cam2worlds[indices, :3, :3])
    cam2cams[:, :3, 3] = torch.bmm(world2cam[:, :3, :3], cam2worlds[indices, :3, 3:])[..., 0]
    cam2cams[:, :3, 3] += world2cam[:, :3, 3]
    return cam2cams

def get_fwd_bwd_cam2cams(cam2worlds, indices):
    fwd_cam2cams = get_cam2cams(cam2worlds, indices, 1)
    bwd_cam2cams = get_cam2cams(cam2worlds, indices, -1)
    return fwd_cam2cams, bwd_cam2cams

def get_pred_flow(pts, ij, cam2cams, focal, center):
    new_pts = torch.transpose(torch.bmm(cam2cams[:, :3, :3], torch.transpose(pts, 1, 2)), 1, 2)
    new_pts = new_pts + cam2cams[:, None, :3, 3]
    new_ij = pts2px(new_pts, focal, center)

    return new_ij - ij.float()

def compute_depth_loss(dyn_depth, gt_depth):
    t_d = torch.median(dyn_depth, dim=-1, keepdim=True).values
    s_d = torch.mean(torch.abs(dyn_depth - t_d), dim=-1, keepdim=True)
    dyn_depth_norm = (dyn_depth - t_d) / s_d

    t_gt = torch.median(gt_depth, dim=-1, keepdim=True).values
    s_gt = torch.mean(torch.abs(gt_depth - t_gt), dim=-1, keepdim=True)
    gt_depth_norm = (gt_depth - t_gt) / s_gt

    return dyn_depth_norm, gt_depth_norm, (dyn_depth_norm - gt_depth_norm) ** 2

def encode_flow(flow, mask):
    flow = 2**15 + flow * (2**8)
    mask &= np.max(flow, axis=-1) < (2**16 - 1)
    mask &= 0 < np.min(flow, axis=-1)
    return np.concatenate([flow.astype(np.uint16), mask[..., None].astype(np.uint16) * (2**16 - 1)], axis=-1)

def decode_flow(encoded_flow):
    flow = encoded_flow[..., :2].astype(np.float32)
    flow -= 2**15
    flow /= 2**8
    return flow, (encoded_flow[..., 2] > 2**15).astype(np.float32)

def get_camera_mesh(pose, depth=1):
    vertices = (
        torch.tensor(
            [[-0.5, -0.5, -1], [0.5, -0.5, -1], [0.5, 0.5, -1], [-0.5, 0.5, -1], [0, 0, 0]]
        )
        * depth
    )
    faces = torch.tensor(
        [[0, 1, 2], [0, 2, 3], [0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]]
    )
    vertices = vertices @ pose[:, :3, :3].transpose(-1, -2)
    vertices += pose[:, None, :3, 3]
    vertices[..., 1:] *= -1 # Axis flip
    wireframe = vertices[:, [0, 1, 2, 3, 0, 4, 1, 2, 4, 3]]
    return vertices, faces, wireframe


def merge_wireframes(wireframe):
    wireframe_merged = [[], [], []]
    for w in wireframe:
        wireframe_merged[0] += [float(n) for n in w[:, 0]]
        wireframe_merged[1] += [float(n) for n in w[:, 1]]
        wireframe_merged[2] += [float(n) for n in w[:, 2]]
    return wireframe_merged


def draw_poses(poses, colours):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    centered_poses = poses.clone()
    centered_poses[:, :, 3] -= torch.mean(centered_poses[:, :, 3], dim=0, keepdim=True)

    vertices, faces, wireframe = get_camera_mesh(
        centered_poses, 0.05
    )
    center = vertices[:, -1]
    ps = max(torch.max(center).item(), 0.1)
    ms = min(torch.min(center).item(), -0.1)
    ax.set_xlim3d(ms, ps)
    ax.set_ylim3d(ms, ps)
    ax.set_zlim3d(ms, ps)
    wireframe_merged = merge_wireframes(wireframe)
    for c in range(center.shape[0]):
        ax.plot(
            wireframe_merged[0][c * 10 : (c + 1) * 10],
            wireframe_merged[1][c * 10 : (c + 1) * 10],
            wireframe_merged[2][c * 10 : (c + 1) * 10],
            color=colours[c],
        )

    plt.tight_layout()
    fig.canvas.draw()
    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return img # np.zeros([5, 5, 3], dtype=np.uint8)

def compute_tv_norm(values, l=2, weighting=None):  # pylint: disable=g-doc-args
  """Returns TV norm for input values.
  Note: The weighting / masking term was necessary to avoid degenerate
  solutions on GPU; only observed on individual DTU scenes.
  """
  v00 = values[:, :-1, :-1]
  v01 = values[:, :-1, 1:]
  v10 = values[:, 1:, :-1]

  if l == 2:
    loss = ((v00 - v01) ** 2) + ((v00 - v10) ** 2)
  elif l == 1:
    loss = torch.abs(v00 - v01) + torch.abs(v00 - v10)
  else:
    loss = (torch.abs(v00 - v01) + 1e-6) ** l + (torch.abs(v00 - v10) + 1e-6) ** l

  if weighting is not None:
    loss = loss * weighting
  return loss

def mse2psnr(x):
    return -10.0 * torch.log(x) / torch.log(torch.Tensor([10.0]))


def visualize_depth_numpy(depth, minmax=None, cmap=cv2.COLORMAP_JET):
    """
    depth: (H, W)
    """

    x = np.nan_to_num(depth)  # change nan to 0
    if minmax is None:
        mi = np.min(x[x > 0])  # get minimum positive depth (ignore background)
        ma = np.max(x)
    else:
        mi, ma = minmax

    x = (x - mi) / (ma - mi + 1e-8)  # normalize to 0~1
    x = (255 * x).astype(np.uint8)
    x_ = cv2.applyColorMap(x, cmap)
    return x_, [mi, ma]


def init_log(log, keys):
    for key in keys:
        log[key] = torch.tensor([0.0], dtype=float)
    return log


def visualize_depth(depth, minmax=None, cmap=cv2.COLORMAP_JET):
    """
    depth: (H, W)
    """
    if type(depth) is not np.ndarray:
        depth = depth.cpu().numpy()

    x = np.nan_to_num(depth)  # change nan to 0
    if minmax is None:
        mi = np.min(x[x > 0])  # get minimum positive depth (ignore background)
        ma = np.max(x)
    else:
        mi, ma = minmax

    x = (x - mi) / (ma - mi + 1e-8)  # normalize to 0~1
    x = (255 * np.clip(x, 0, 1)).astype(np.uint8)
    x_ = Image.fromarray(cv2.applyColorMap(x, cmap))
    x_ = T.ToTensor()(x_)  # (3, H, W)
    return x_, [mi, ma]


def N_to_reso(n_voxels, bbox):
    xyz_min, xyz_max = bbox
    voxel_size = ((xyz_max - xyz_min).prod() / n_voxels).pow(1 / 3)
    return ((xyz_max - xyz_min) / voxel_size).long().tolist()


def cal_n_samples(reso, step_ratio=0.5):
    return int(np.linalg.norm(reso) / step_ratio)


__LPIPS__ = {}


def init_lpips(net_name, device):
    assert net_name in ["alex", "vgg"]
    import lpips

    print(f"init_lpips: lpips_{net_name}")
    return lpips.LPIPS(net=net_name, version="0.1").eval().to(device)


def rgb_lpips(np_gt, np_im, net_name, device):
    if net_name not in __LPIPS__:
        __LPIPS__[net_name] = init_lpips(net_name, device)
    gt = torch.from_numpy(np_gt).permute([2, 0, 1]).contiguous().to(device)
    im = torch.from_numpy(np_im).permute([2, 0, 1]).contiguous().to(device)
    return __LPIPS__[net_name](gt, im, normalize=True).item()

""" Evaluation metrics (ssim, lpips)
"""


def rgb_ssim(
    img0,
    img1,
    max_val,
    filter_size=11,
    filter_sigma=1.5,
    k1=0.01,
    k2=0.03,
    return_map=False,
):
    # Modified from https://github.com/google/mipnerf/blob/16e73dfdb52044dcceb47cda5243a686391a6e0f/internal/math.py#L58
    assert len(img0.shape) == 3
    assert img0.shape[-1] == 3
    assert img0.shape == img1.shape

    # Construct a 1D Gaussian blur filter.
    hw = filter_size // 2
    shift = (2 * hw - filter_size + 1) / 2
    f_i = ((np.arange(filter_size) - hw + shift) / filter_sigma) ** 2
    filt = np.exp(-0.5 * f_i)
    filt /= np.sum(filt)

    # Blur in x and y (faster than the 2D convolution).
    def convolve2d(z, f):
        return scipy.signal.convolve2d(z, f, mode="valid")

    def filt_fn(z):
        return np.stack(
            [
                convolve2d(convolve2d(z[..., i], filt[:, None]), filt[None, :])
                for i in range(z.shape[-1])
            ],
            -1,
        )

    mu0 = filt_fn(img0)
    mu1 = filt_fn(img1)
    mu00 = mu0 * mu0
    mu11 = mu1 * mu1
    mu01 = mu0 * mu1
    sigma00 = filt_fn(img0**2) - mu00
    sigma11 = filt_fn(img1**2) - mu11
    sigma01 = filt_fn(img0 * img1) - mu01

    # Clip the variances and covariances to valid values.
    # Variance must be non-negative:
    sigma00 = np.maximum(0.0, sigma00)
    sigma11 = np.maximum(0.0, sigma11)
    sigma01 = np.sign(sigma01) * np.minimum(np.sqrt(sigma00 * sigma11), np.abs(sigma01))
    c1 = (k1 * max_val) ** 2
    c2 = (k2 * max_val) ** 2
    numer = (2 * mu01 + c1) * (2 * sigma01 + c2)
    denom = (mu00 + mu11 + c1) * (sigma00 + sigma11 + c2)
    ssim_map = numer / denom
    ssim = np.mean(ssim_map)
    return ssim_map if return_map else ssim


import torch.nn as nn


class TVLoss(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        # batch_size = x.size()[1]
        h_x = x.size()[2]
        w_x = x.size()[3]
        # count_h = self._tensor_size(x[:,:,1:,:])
        # count_w = self._tensor_size(x[:,:,:,1:])
        tv = 0
        if h_x > 1:
            tv += torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).mean()
        if w_x > 1:
            tv += torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).mean()
        return self.TVLoss_weight*2*tv

    def _tensor_size(self,t):
        return t.size()[0]*t.size()[2]*t.size()[3]


import plyfile
import skimage.measure


def convert_sdf_samples_to_ply(
    pytorch_3d_sdf_tensor,
    ply_filename_out,
    bbox,
    level=0.5,
    offset=None,
    scale=None,
):
    """
    Convert sdf samples to .ply

    :param pytorch_3d_sdf_tensor: a torch.FloatTensor of shape (n,n,n)
    :voxel_grid_origin: a list of three floats: the bottom, left, down origin of the voxel grid
    :voxel_size: float, the size of the voxels
    :ply_filename_out: string, path of the filename to save to

    This function adapted from: https://github.com/RobotLocomotion/spartan
    """

    numpy_3d_sdf_tensor = pytorch_3d_sdf_tensor.numpy()
    voxel_size = list((bbox[1] - bbox[0]) / np.array(pytorch_3d_sdf_tensor.shape))

    verts, faces, normals, values = skimage.measure.marching_cubes(
        numpy_3d_sdf_tensor, level=level, spacing=voxel_size
    )
    faces = faces[..., ::-1]  # inverse face orientation

    # transform from voxel coordinates to camera coordinates
    # note x and y are flipped in the output of marching_cubes
    mesh_points = np.zeros_like(verts)
    mesh_points[:, 0] = bbox[0, 0] + verts[:, 0]
    mesh_points[:, 1] = bbox[0, 1] + verts[:, 1]
    mesh_points[:, 2] = bbox[0, 2] + verts[:, 2]

    # apply additional offset and scale
    if scale is not None:
        mesh_points = mesh_points / scale
    if offset is not None:
        mesh_points = mesh_points - offset

    # try writing to the ply file

    num_verts = verts.shape[0]
    num_faces = faces.shape[0]

    verts_tuple = np.zeros((num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])

    for i in range(0, num_verts):
        verts_tuple[i] = tuple(mesh_points[i, :])

    faces_building = []
    for i in range(0, num_faces):
        faces_building.append(((faces[i, :].tolist(),)))
    faces_tuple = np.array(faces_building, dtype=[("vertex_indices", "i4", (3,))])

    el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
    el_faces = plyfile.PlyElement.describe(faces_tuple, "face")

    ply_data = plyfile.PlyData([el_verts, el_faces])
    print("saving mesh to %s" % (ply_filename_out))
    ply_data.write(ply_filename_out)

def sixD_to_mtx(r):
    b1 = r[..., 0]
    b1 = b1 / torch.norm(b1, dim=-1)[:, None]
    b2 = r[..., 1] - torch.sum(b1 * r[..., 1], dim=-1)[:, None] * b1
    b2 = b2 / torch.norm(b2, dim=-1)[:, None]
    b3 = torch.cross(b1, b2)

    return torch.stack([b1, b2, b3], dim=-1)


def mtx_to_sixD(r):
    return torch.stack([r[..., 0], r[..., 1]], dim=-1)

def strided_app(a, L, S):  # Window len = L, Stride len/stepsize = S
    nrows = ((a.size-L)//S)+1
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(a, shape=(nrows,L), strides=(S*n,n))

def filter1d(vec, time, W):
    stepsize = 2 * W + 1
    filtered = np.median(strided_app(vec, stepsize, stepsize), axis=-1)
    pre_smoothed = np.interp(time, time[W:-W:stepsize], filtered)
    return pre_smoothed

def smooth_vec(vec, time, s, median_prefilter):
    if median_prefilter:
        vec = np.stack([
            filter1d(vec[..., 0], time, 5),
            filter1d(vec[..., 1], time, 5),
            filter1d(vec[..., 2], time, 5)
        ], axis=-1)
    smoothed = np.zeros_like(vec)
    for i in range(vec.shape[1]):
        spl = UnivariateSpline(time, vec[..., i])
        spl.set_smoothing_factor(s)
        smoothed[..., i] = spl(time)
    return smoothed

def smooth_poses_spline(poses, st=0.5, sr=4, median_prefilter=True):
    if len(poses) < 30:
        median_prefilter = False
    poses[:, 0] = -poses[:, 0]
    posesnp = poses.cpu().numpy()
    scale = 2e-2 / np.median(np.linalg.norm(posesnp[1:, :3, 3] - posesnp[:-1, :3, 3], axis=-1))
    posesnp[:, :3, 3] *= scale
    time = np.linspace(0, 1, len(posesnp)) 
    
    t = smooth_vec(posesnp[..., 3], time, st, median_prefilter)
    z = smooth_vec(posesnp[..., 2], time, sr, median_prefilter)
    z /= np.linalg.norm(z, axis=-1)[:, None]
    y_ = smooth_vec(posesnp[..., 1], time, sr, median_prefilter)
    x = np.cross(z, y_)
    x /= np.linalg.norm(x, axis=-1)[:, None]
    y = np.cross(x, z)

    smooth_posesnp = np.stack([x, y, z, t], -1)
    poses[:, 0] = -poses[:, 0]
    smooth_posesnp[:, 0] = -smooth_posesnp[:, 0]
    smooth_posesnp[:, :3, 3] /= scale
    return torch.Tensor(smooth_posesnp.astype(np.float32)).to(poses)