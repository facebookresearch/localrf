import os
import traceback
from argparse import Namespace
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from torch.distributed.elastic.multiprocessing.errors import record
from tqdm import tqdm

from mega_nerf.image_metadata import ImageMetadata
from mega_nerf.opts import get_opts_base
from mega_nerf.runner import Runner


def _get_render_opts() -> Namespace:
    parser = get_opts_base()

    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--centroids_path', type=str, required=True)
    parser.add_argument('--save_depth_npz', default=False, action='store_true')
    parser.add_argument('--resume', default=False, action='store_true')

    return parser.parse_args()


@torch.inference_mode()
def _render_images(hparams: Namespace) -> None:
    runner = Runner(hparams, False)

    input = Path(hparams.input)
    centroids = torch.load(hparams.centroids_path, map_location='cpu')['centroids']

    c2ws = []
    poses_path = input / 'poses.txt'
    with poses_path.open() as f:
        for line in f:
            c2ws.append(torch.FloatTensor([float(x) for x in line.strip().split()]).view(3, 4))

    intrinsics = []
    intrinsics_path = input / 'intrinsics.txt'
    with intrinsics_path.open() as f:
        for line in f:
            intrinsics.append([float(x) / hparams.val_scale_factor for x in line.strip().split()])

    embeddings = []
    embeddings_path = input / 'embeddings.txt'
    with embeddings_path.open() as f:
        for line in f:
            embeddings.append(int(line.strip()))

    output = Path(hparams.output)

    rank = int(os.environ.get('RANK', '0'))
    if rank == 0:
        (output / 'rgbs').mkdir(parents=True, exist_ok=hparams.resume)
        (output / 'depths').mkdir(parents=True, exist_ok=hparams.resume)
        (output / 'cells').mkdir(parents=True, exist_ok=hparams.resume)
        if hparams.save_depth_npz:
            (output / 'depths_npz').mkdir(parents=True, exist_ok=hparams.resume)

    if 'RANK' in os.environ:
        dist.barrier()
        world_size = int(os.environ['WORLD_SIZE'])
    else:
        world_size = 1

    runner.nerf.eval()
    if runner.bg_nerf is not None:
        runner.bg_nerf.eval()

    coordinate_info = torch.load(Path(hparams.dataset_path) / 'coordinates.pt', map_location='cpu')
    pose_scale_factor = coordinate_info['pose_scale_factor']

    for i in tqdm(np.arange(rank, len(c2ws), world_size)):
        cell_path = output / 'cells' / '{0:06d}.jpg'.format(i)

        if hparams.resume and cell_path.exists():
            try:
                test = np.array(Image.open(cell_path))  # verify with last visualization to be written
                print('skipping {} {}'.format(cell_path, test[0]))
                continue
            except:
                traceback.print_exc()
                pass

        W = int(intrinsics[i][0])
        H = int(intrinsics[i][1])
        results, rays = runner.render_image(
            ImageMetadata(Path(''), c2ws[i], W, H, torch.FloatTensor(intrinsics[i][2:]), embeddings[i], None, False))

        typ = 'fine' if 'rgb_fine' in results else 'coarse'
        result_rgbs = results[f'rgb_{typ}']

        result_rgbs = result_rgbs.view(H, W, 3) * 255
        rgbs = result_rgbs.byte().cpu().numpy().astype(np.uint8)
        img = Image.fromarray(rgbs)
        img.save(output / 'rgbs' / '{0:06d}.jpg'.format(i))

        depth = torch.nan_to_num(results[f'depth_{typ}']).view(H, W).cpu()

        if hparams.save_depth_npz:
            np.save(str(output / 'depths_npz' / '{0:06d}.npy'.format(i)), (depth * pose_scale_factor).numpy())

        if f'bg_depth_{typ}' in results:
            to_use = torch.nan_to_num(results[f'fg_depth_{typ}']).view(-1)
            while to_use.shape[0] > 2 ** 24:
                to_use = to_use[::2]
            ma = torch.quantile(to_use, 0.95)
            depth = depth.clamp_max(ma)

        depth_vis = Runner.visualize_scalars(torch.log(depth + 1e-8).view(H, W).cpu())
        Image.fromarray(depth_vis.astype(np.uint8)).save(output / 'depths' / '{0:06d}.jpg'.format(i))

        rays = rays.view(H, W, -1).cpu()
        locations = rays[..., :3] + rays[..., 3:6] * depth.unsqueeze(-1)

        cluster_assignments = torch.cdist(locations.view(-1, 3)[:, :3], centroids).argmin(dim=1).view(H, W).float()
        cluster_assignments /= len(centroids)
        centroid_colors = cv2.cvtColor(cv2.applyColorMap((cluster_assignments * 255).byte().numpy(), cv2.COLORMAP_HSV),
                                       cv2.COLOR_BGR2RGB)

        centroid_colors = cv2.addWeighted(rgbs, 0.7, centroid_colors, 0.3, 0)
        Image.fromarray(centroid_colors.astype(np.uint8)).save(cell_path)


@record
def main(hparams: Namespace) -> None:
    assert hparams.ckpt_path is not None or hparams.container_path is not None

    if hparams.detect_anomalies:
        with torch.autograd.detect_anomaly():
            _render_images(hparams)
    else:
        _render_images(hparams)


if __name__ == '__main__':
    main(_get_render_opts())
