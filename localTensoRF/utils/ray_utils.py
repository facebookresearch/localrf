import re

import numpy as np
import torch
from kornia import create_meshgrid
from torch import searchsorted

def contract(x):
    x_norm = torch.clamp(x.abs().amax(dim=-1, keepdim=True), 1e-6)
    z = torch.where(x_norm <= 1, x, ((2 * x_norm - 1) / (x_norm**2)) * x)
    return z

def uncontract(pts):
    contracted_pts_norm = torch.abs(pts.clone())
    outer_mask = contracted_pts_norm > 1.0
    pts_norm = 1 / (2 - contracted_pts_norm[outer_mask])
    pts[outer_mask] = pts[outer_mask] * pts_norm / (2 - 1 / pts_norm)

def depth2dist(z_vals, cos_angle):
    # z_vals: [N_ray N_sample]
    device = z_vals.device
    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat(
        [dists, torch.Tensor([1e10]).to(device).expand(dists[..., :1].shape)], -1
    )  # [N_rays, N_samples]
    dists = dists * cos_angle.unsqueeze(-1)
    return dists


def ndc2dist(ndc_pts, cos_angle):
    dists = torch.norm(ndc_pts[:, 1:] - ndc_pts[:, :-1], dim=-1)
    dists = torch.cat(
        [dists, 1e10 * cos_angle.unsqueeze(-1)], -1
    )  # [N_rays, N_samples]
    return dists


def get_ray_directions(H, W, focal, center=None):
    """
    Get ray directions for all pixels in camera coordinate.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems
    Inputs:
        H, W, focal: image height, width and focal length
    Outputs:
        directions: (H, W, 3), the direction of the rays in camera coordinate
    """
    grid = create_meshgrid(H, W, normalized_coordinates=False)[0] + 0.5

    i, j = grid.unbind(-1)
    # the direction here is without +0.5 pixel centering as calibration is not so accurate
    # see https://github.com/bmild/nerf/issues/24
    cent = center if center is not None else [W / 2, H / 2]
    directions = torch.stack(
        [(i - cent[0]) / focal[0], -(j - cent[1]) / focal[1], -torch.ones_like(i)], -1
    )  # (H, W, 3)

    return directions


def get_ray_directions_lean(i, j, focal, center):
    """
    get_ray_directions but returns only relevant rays
    Inputs:
        focal: (2), focal length
    Outputs:
        directions: (b, 3), the direction of the rays in camera coordinate
    """
    i, j = i.float() + 0.5, j.float() + 0.5
    directions = torch.stack([(i - center[0]) / focal, -(j - center[1]) / focal, -torch.ones_like(i)], -1)  # (b, 3)
    return directions

def get_rays_lean(directions, c2w):
    """
    get_rays but returns only relevant rays
    Inputs:
        directions: (B, 3) ray directions in camera coordinate
        c2w: (B, 3, 4) transformation matrix from camera coordinate to world coordinate
    Outputs:
        rays_o: (B, 3), the origin of the rays in world coordinate
        rays_d: (B, 3), the normalized direction of the rays in world coordinate
    """
    # Rotate ray directions from camera coordinate to the world coordinate
    # rays_d = torch.matmul(directions, torch.transpose(c2w[:, :3, :3], 1, 2))  # (B, 3)

    # The origin of all rays is the camera origin in world coordinate
    rays_o = c2w[:, :3, 3]  # (B, 3)
    rays_d = torch.bmm(c2w[:, :3, :3], directions[..., None])[..., 0]  # (B, 3)

    return rays_o, rays_d


def get_ray_directions_blender(H, W, focal, center=None):
    """
    Get ray directions for all pixels in camera coordinate.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems
    Inputs:
        H, W, focal: image height, width and focal length
    Outputs:
        directions: (H, W, 3), the direction of the rays in camera coordinate
    """
    grid = create_meshgrid(H, W, normalized_coordinates=False)[0] + 0.5
    i, j = grid.unbind(-1)
    # the direction here is without +0.5 pixel centering as calibration is not so accurate
    # see https://github.com/bmild/nerf/issues/24
    cent = center if center is not None else [W / 2, H / 2]
    directions = torch.stack(
        [(i - cent[0]) / focal[0], -(j - cent[1]) / focal[1], -torch.ones_like(i)], -1
    )  # (H, W, 3)

    return directions


def get_rays(directions, c2w):
    """
    Get ray origin and normalized directions in world coordinate for all pixels in one image.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems
    Inputs:
        directions: (H, W, 3) precomputed ray directions in camera coordinate
        c2w: (3, 4) transformation matrix from camera coordinate to world coordinate
    Outputs:
        rays_o: (H*W, 3), the origin of the rays in world coordinate
        rays_d: (H*W, 3), the normalized direction of the rays in world coordinate
    """
    # Rotate ray directions from camera coordinate to the world coordinate
    rays_d = directions @ c2w[:3, :3].T  # (H, W, 3)
    # rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    # The origin of all rays is the camera origin in world coordinate
    rays_o = c2w[:3, 3].expand(rays_d.shape)  # (H, W, 3)

    rays_d = rays_d.view(-1, 3)
    rays_o = rays_o.view(-1, 3)

    return rays_o, rays_d


def get_rays_with_batch(directions, c2w):
    """
    Get ray origin and normalized directions in world coordinate for all pixels in one image.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems
    Inputs:
        directions: (B, H, W, 3) precomputed ray directions in camera coordinate
        c2w: (B, 3, 4) transformation matrix from camera coordinate to world coordinate
    Outputs:
        rays_o: (B, H*W, 3), the origin of the rays in world coordinate
        rays_d: (B, H*W, 3), the normalized direction of the rays in world coordinate
    """
    # Rotate ray directions from camera coordinate to the world coordinate
    rays_d = torch.matmul(
        directions, torch.transpose(c2w[:, :3, :3], 1, 2)
    )  # (B, H, W, 3)
    # rays_d = directions @ c2w[:3, :3].T  # (B, H, W, 3)
    # rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    # The origin of all rays is the camera origin in world coordinate
    rays_o = torch.tile(
        torch.unsqueeze(c2w[:, :3, 3], 1), (1, directions.shape[1], 1)
    )  # (B, H*W, 3)
    # rays_o = c2w[:3, 3].expand(rays_d.shape)  # (B, H*W, 3)

    rays_d = rays_d.view(c2w.shape[0], -1, 3)
    rays_o = rays_o.view(c2w.shape[0], -1, 3)

    return rays_o, rays_d


def ndc_rays_blender(H, W, focal, near, rays_o, rays_d):
    # Shift ray origins to near plane
    t = -(near + rays_o[..., 2]) / rays_d[..., 2]
    rays_o = rays_o + t[..., None] * rays_d

    # Projection
    o0 = -1.0 / (W / (2.0 * focal)) * rays_o[..., 0] / rays_o[..., 2]
    o1 = -1.0 / (H / (2.0 * focal)) * rays_o[..., 1] / rays_o[..., 2]
    o2 = 1.0 + 2.0 * near / rays_o[..., 2]

    d0 = (
        -1.0
        / (W / (2.0 * focal))
        * (rays_d[..., 0] / rays_d[..., 2] - rays_o[..., 0] / rays_o[..., 2])
    )
    d1 = (
        -1.0
        / (H / (2.0 * focal))
        * (rays_d[..., 1] / rays_d[..., 2] - rays_o[..., 1] / rays_o[..., 2])
    )
    d2 = -2.0 * near / rays_o[..., 2]

    rays_o = torch.stack([o0, o1, o2], -1)
    rays_d = torch.stack([d0, d1, d2], -1)

    return rays_o, rays_d


def ndc_rays_blender2(H, W, focal, near, rays_o, rays_d):
    # Shift ray origins to near plane
    t = -(near + rays_o[..., 2]) / rays_d[..., 2]
    rays_o = rays_o + t[..., None] * rays_d

    # Projection
    o0 = -1. / (W / (2. * focal[0])) * rays_o[..., 0] / rays_o[..., 2]
    o1 = -1. / (H / (2. * focal[1])) * rays_o[..., 1] / rays_o[..., 2]
    o2 = 1. + 2. * near / rays_o[..., 2]

    d0 = -1. / (W / (2. * focal[0])) * (rays_d[..., 0] / rays_d[..., 2] - rays_o[..., 0] / rays_o[..., 2])
    d1 = -1. / (H / (2. * focal[1])) * (rays_d[..., 1] / rays_d[..., 2] - rays_o[..., 1] / rays_o[..., 2])
    d2 = -2. * near / rays_o[..., 2]

    rays_o = torch.stack([o0, o1, o2], -1)
    rays_d = torch.stack([d0, d1, d2], -1)

    return rays_o, rays_d


def ndc_rays(H, W, focal, near, rays_o, rays_d):
    # Shift ray origins to near plane
    t = (near - rays_o[..., 2]) / rays_d[..., 2]
    rays_o = rays_o + t[..., None] * rays_d

    # Projection
    o0 = 1.0 / (W / (2.0 * focal)) * rays_o[..., 0] / rays_o[..., 2]
    o1 = 1.0 / (H / (2.0 * focal)) * rays_o[..., 1] / rays_o[..., 2]
    o2 = 1.0 - 2.0 * near / rays_o[..., 2]

    d0 = (
        1.0
        / (W / (2.0 * focal))
        * (rays_d[..., 0] / rays_d[..., 2] - rays_o[..., 0] / rays_o[..., 2])
    )
    d1 = (
        1.0
        / (H / (2.0 * focal))
        * (rays_d[..., 1] / rays_d[..., 2] - rays_o[..., 1] / rays_o[..., 2])
    )
    d2 = 2.0 * near / rays_o[..., 2]

    rays_o = torch.stack([o0, o1, o2], -1)
    rays_d = torch.stack([d0, d1, d2], -1)

    return rays_o, rays_d


# Hierarchical sampling (section 5.2)
def sample_pdf(bins, weights, N_samples, det=False, pytest=False):
    device = weights.device
    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(0.0, 1.0, steps=N_samples, device=device)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples], device=device)

    # Pytest, overwrite u with numpy's fixed random numbers
    if pytest:
        np.random.seed(0)
        new_shape = list(cdf.shape[:-1]) + [N_samples]
        if det:
            u = np.linspace(0.0, 1.0, N_samples)
            u = np.broadcast_to(u, new_shape)
        else:
            u = np.random.rand(*new_shape)
        u = torch.Tensor(u)

    # Invert CDF
    u = u.contiguous()
    inds = searchsorted(cdf.detach(), u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = cdf_g[..., 1] - cdf_g[..., 0]
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples


def dda(rays_o, rays_d, bbox_3D):
    inv_ray_d = 1.0 / (rays_d + 1e-6)
    t_min = (bbox_3D[:1] - rays_o) * inv_ray_d  # N_rays 3
    t_max = (bbox_3D[1:] - rays_o) * inv_ray_d
    t = torch.stack((t_min, t_max))  # 2 N_rays 3
    t_min = torch.max(torch.min(t, dim=0)[0], dim=-1, keepdim=True)[0]
    t_max = torch.min(torch.max(t, dim=0)[0], dim=-1, keepdim=True)[0]
    return t_min, t_max

def ndc_bbox(all_rays):
    near_min = torch.min(all_rays[..., :3].view(-1, 3), dim=0)[0]
    near_max = torch.max(all_rays[..., :3].view(-1, 3), dim=0)[0]
    far_min = torch.min((all_rays[..., :3] + all_rays[..., 3:6]).view(-1, 3), dim=0)[0]
    far_max = torch.max((all_rays[..., :3] + all_rays[..., 3:6]).view(-1, 3), dim=0)[0]
    print(
        f"===> ndc bbox near_min:{near_min} near_max:{near_max} far_min:{far_min} far_max:{far_max}"
    )
    return torch.stack(
        (torch.minimum(near_min, far_min), torch.maximum(near_max, far_max))
    )
