# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2022 Anpei Chen

import numpy as np
import torch
from kornia import create_meshgrid
from torch import searchsorted

def contract(x):
    x_norm = torch.clamp(x.abs().amax(dim=-1, keepdim=True), 1e-6)
    z = torch.where(x_norm <= 1, x, ((2 * x_norm - 1) / (x_norm**2)) * x)
    return z

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

def sphere2xyz(r, theta, phi):
    x = torch.cos(phi) * torch.sin(theta)
    y = torch.sin(phi)
    z = torch.cos(phi) * torch.cos(theta)
    return torch.stack([r*x, r*y, r*z], axis=-1)

def get_ray_directions_360(i, j, W, H):
    i, j = i.float() + 0.5, j.float() + 0.5
    phi = j * torch.pi / H - torch.pi / 2.
    theta = i * 2. * torch.pi / W + torch.pi
    directions = sphere2xyz(torch.ones_like(theta), theta, phi)
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
    # The origin of all rays is the camera origin in world coordinate
    rays_o = c2w[:, :3, 3]  # (B, 3)
    # Rotate ray directions from camera coordinate to the world coordinate
    rays_d = torch.bmm(c2w[:, :3, :3], directions[..., None])[..., 0]  # (B, 3)

    return rays_o, rays_d