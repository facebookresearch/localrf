#  Copyright 2021 The PlenOctree Authors.
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#
#  1. Redistributions of source code must retain the above copyright notice,
#  this list of conditions and the following disclaimer.
#
#  2. Redistributions in binary form must reproduce the above copyright notice,
#  this list of conditions and the following disclaimer in the documentation
#  and/or other materials provided with the distribution.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS 'AS IS'
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
#  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
#  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
#  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
#  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
#  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
#  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#  POSSIBILITY OF SUCH DAMAGE.

from argparse import Namespace
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from svox import N3Tree
from svox.helpers import _get_c_extension
from torch import nn
from tqdm import tqdm

from mega_nerf.models.model_utils import get_nerf
from mega_nerf.opts import get_opts_base

_C = _get_c_extension()


def _get_extraction_opts() -> Namespace:
    parser = get_opts_base()

    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--alpha_thresh', type=float, default=0.01)
    parser.add_argument('--scale_alpha_thresh', type=float, default=0.01)
    parser.add_argument('--max_refine_prop', type=float, default=0.5)
    parser.add_argument('--tree_branch_n', type=int, default=2)
    parser.add_argument('--init_grid_depth', type=int, default=8)
    parser.add_argument('--samples_per_cell', type=int, default=256)
    parser.add_argument('--masking_mode', type=str, default='weight', choices=['sigma', 'weight'])
    parser.add_argument('--weight_thresh', type=float, default=0.001)
    parser.add_argument('--embedding_index', type=int, default=0)
    parser.add_argument('--camera_params', type=int, nargs='+', default=[800, 800, 400, 400, 400, 400])
    parser.add_argument('--renderer_step_size', type=float, default=1e-6)

    return parser.parse_known_args()[0]


def _auto_scale(hparams: Namespace, nerf: nn.Module, center: List[float], radius: List[float],
                device: torch.device) -> Tuple[List[float], List[float]]:
    print('Step 0: Auto scale')
    reso = 2 ** hparams.init_grid_depth

    radius = torch.tensor(radius, dtype=torch.float32)
    center = torch.tensor(center, dtype=torch.float32)
    scale = 0.5 / radius
    offset = 0.5 * (1.0 - center / radius)

    arr = (torch.arange(0, reso, dtype=torch.float32) + 0.5) / reso
    xx = (arr - offset[0]) / scale[0]
    yy = (arr - offset[1]) / scale[1]
    zz = (arr - offset[2]) / scale[2]

    grid = torch.stack(torch.meshgrid(xx, yy, zz)).reshape(3, -1).T

    approx_delta = 2.0 / reso
    sigma_thresh = -np.log(1.0 - hparams.scale_alpha_thresh) / approx_delta

    lc = None
    uc = None

    for i in tqdm(range(0, grid.shape[0], hparams.model_chunk_size)):
        grid_chunk = grid[i:i + hparams.model_chunk_size].to(device)

        output = nerf(False, grid_chunk, sigma_only=True) if hparams.use_cascade else nerf(grid_chunk, sigma_only=True)
        sigmas = output[:, 0]
        mask = sigmas >= sigma_thresh
        grid_chunk = grid_chunk[mask]
        del mask

        if grid_chunk.shape[0] > 0:
            if lc is None:
                lc = grid_chunk.min(dim=0)[0]
                uc = grid_chunk.max(dim=0)[0]
            else:
                lc = torch.minimum(lc, grid_chunk.min(dim=0)[0])
                uc = torch.maximum(uc, grid_chunk.max(dim=0)[0])

        del grid_chunk

    lc = lc - 0.5 / reso
    uc = uc + 0.5 / reso
    return ((lc + uc) * 0.5).tolist(), ((uc - lc) * 0.5).tolist()


def _calculate_grid_weights(hparams: Namespace, tree: N3Tree, poses: torch.Tensor, sigmas: torch.Tensor,
                            reso: int) -> torch.Tensor:
    opts = _C.RenderOptions()
    opts.step_size = hparams.renderer_step_size
    opts.sigma_thresh = 0.0
    opts.ndc_width = -1

    cam = _C.CameraSpec()
    cam.fx = hparams.camera_params[2]
    cam.fy = hparams.camera_params[3]
    cam.width = hparams.camera_params[0]
    cam.height = hparams.camera_params[1]

    grid_data = sigmas.reshape((reso, reso, reso))
    maximum_weight = torch.zeros_like(grid_data)

    for idx in tqdm(range(poses.shape[0])):
        cam.c2w = poses[idx].to(sigmas.device)
        grid_weight, _ = _C.grid_weight_render(
            grid_data,
            cam,
            opts,
            tree.offset,
            tree.invradius,
        )

        maximum_weight = torch.max(maximum_weight, grid_weight)

    return maximum_weight


def _step1(hparams: Namespace, nerf: nn.Module, tree: N3Tree, poses: torch.Tensor, device: torch.device):
    print('Step 1: Grid eval')
    reso = 2 ** (hparams.init_grid_depth + 1)
    offset = tree.offset.cpu()
    scale = tree.invradius.cpu()

    arr = (torch.arange(0, reso, dtype=torch.float32) + 0.5) / reso
    xx = (arr - offset[0]) / scale[0]
    yy = (arr - offset[1]) / scale[1]
    zz = (arr - offset[2]) / scale[2]

    grid = torch.stack(torch.meshgrid(xx, yy, zz)).reshape(3, -1).T

    approx_delta = 2.0 / reso
    sigma_thresh = -np.log(1.0 - hparams.alpha_thresh) / approx_delta

    out_chunks = []
    for i in tqdm(range(0, grid.shape[0], hparams.model_chunk_size)):
        grid_chunk = grid[i:i + hparams.model_chunk_size].to(device)
        result = nerf(False, grid_chunk, sigma_only=True) if hparams.use_cascade else nerf(grid_chunk, sigma_only=True)
        del grid_chunk
        out_chunks.append(result[:, 0])

    sigmas = torch.cat(out_chunks, 0)
    del out_chunks

    if hparams.masking_mode == 'sigma':
        mask = sigmas >= sigma_thresh
    elif hparams.masking_mode == 'weight':
        print('Calculating grid weights')
        grid_weights = _calculate_grid_weights(hparams, tree, poses, sigmas, reso)
        mask = grid_weights.reshape(-1) >= hparams.weight_thresh
        del grid_weights
    else:
        raise Exception('Unsupported masking mode: {}'.format(hparams.masking_mode))
    del sigmas

    grid = grid[mask]
    del mask

    print('Building octree')

    tree = tree.cpu()

    for i in range(hparams.init_grid_depth):
        tree[grid].refine()

    print(tree)


def _step2(hparams: Namespace, nerf: nn.Module, tree: N3Tree, device: torch.device):
    print('Step 2: AA with {} samples per cell'.format(hparams.samples_per_cell))

    chunk_size = hparams.model_chunk_size // hparams.samples_per_cell
    for i in tqdm(range(0, tree.n_leaves, chunk_size)):
        points = tree[i:i + chunk_size].sample(hparams.samples_per_cell)  # (n_cells, n_samples, 3)
        points = points.view(-1, 3).to(device)

        if hparams.pos_dir_dim > 0:
            dirs = torch.zeros_like(points)
            dirs[:, 0] = 1
            points = torch.cat([points, dirs], -1)

        if hparams.appearance_dim > 0:
            points = torch.cat([points, hparams.embedding_index * torch.ones(points.shape[0], 1, device=points.device)],
                               -1)

        rgba = nerf(False, points) if hparams.use_cascade else nerf(points)
        rgba = rgba.reshape(-1, hparams.samples_per_cell, tree.data_dim).mean(dim=1)

        tree[i:i + chunk_size] = rgba.cpu()


@torch.inference_mode()
def main(hparams: Namespace) -> None:
    assert hparams.ckpt_path is not None or hparams.container_path is not None
    assert hparams.ray_altitude_range is not None

    dataset_path = Path(hparams.dataset_path)
    train_path_candidates = sorted(list((dataset_path / 'train' / 'metadata').iterdir()))
    train_paths = [train_path_candidates[i] for i in
                   range(0, len(train_path_candidates), hparams.train_every)]

    metadata_paths = train_paths + list((dataset_path / 'val' / 'metadata').iterdir())

    poses = torch.cat([torch.load(x, map_location='cpu')['c2w'].unsqueeze(0) for x in metadata_paths])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    nerf = get_nerf(hparams, poses.shape[0]).to(device).eval()

    coordinate_info = torch.load(dataset_path / 'coordinates.pt', map_location='cpu')
    origin_drb = coordinate_info['origin_drb']
    pose_scale_factor = coordinate_info['pose_scale_factor']

    max_values = poses[:, :3, 3].max(0)[0]
    min_values = poses[:, :3, 3].min(0)[0]

    ray_altitude_range = [(x - origin_drb[0]) / pose_scale_factor for x in hparams.ray_altitude_range]

    min_values[0] = ray_altitude_range[0]
    max_values[0] = ray_altitude_range[1]

    print('Min and Max values: {} {}'.format(min_values, max_values))

    center = ((max_values + min_values) * 0.5).tolist()
    radius = ((max_values - min_values) * 0.5).tolist()
    print('Center and radius before autoscale: {}, {}'.format(center, radius))

    center, radius = _auto_scale(hparams, nerf, center, radius, device)
    print('Center and radius after autoscale: {}, {}'.format(center, radius))

    sh_deg = hparams.sh_deg if hparams.sh_deg is not None else 0
    num_rgb_channels = 3 * (sh_deg + 1) ** 2
    data_dim = 1 + num_rgb_channels  # alpha + rgb

    print('Data dim is', data_dim)

    print('Creating tree')
    data_format = f'SH{(sh_deg + 1) ** 2}' if sh_deg > 0 else 'RGBA'
    tree = N3Tree(N=hparams.tree_branch_n,
                  data_dim=data_dim,
                  init_refine=0,
                  init_reserve=500000,
                  geom_resize_fact=1.0,
                  depth_limit=hparams.init_grid_depth,
                  radius=radius,
                  center=center,
                  data_format=data_format,
                  device=device)

    _step1(hparams, nerf, tree, poses, device)
    _step2(hparams, nerf, tree, device)

    tree.shrink_to_fit()

    print('Filling in internal nodes')
    child = tree.child.clone()
    parent_depth = tree.parent_depth.clone()
    n_free = tree._n_free.item()
    while tree.n_frontier > 1:
        print('Internal {} leaves {} frontier {} free {}'.format(tree.n_internal, tree.n_leaves, tree.n_frontier,
                                                                 tree._n_free))
        tree.merge()

    tree.child.set_(child)
    tree.parent_depth.set_(parent_depth)
    tree._n_free.fill_(n_free)

    print(tree)

    print('Saving tree to: {}'.format(hparams.output))
    Path(hparams.output).parent.mkdir(parents=True, exist_ok=True)
    tree.save(hparams.output, compress=False)


if __name__ == '__main__':
    main(_get_extraction_opts())
