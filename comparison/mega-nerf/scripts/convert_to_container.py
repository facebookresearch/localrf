from argparse import Namespace
from pathlib import Path

import torch
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present

from mega_nerf.models.mega_nerf import MegaNeRF
from mega_nerf.models.mega_nerf_container import MegaNeRFContainer
from mega_nerf.models.model_utils import get_nerf, get_bg_nerf
from mega_nerf.opts import get_opts_base


def _get_merge_opts() -> Namespace:
    parser = get_opts_base()
    parser.add_argument('--output', type=str, required=True)

    return parser.parse_known_args()[0]


@torch.inference_mode()
def main(hparams: Namespace) -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    centroids = torch.zeros(1, 3)

    loaded = torch.load(hparams.ckpt_path, map_location='cpu')
    consume_prefix_in_state_dict_if_present(loaded['model_state_dict'], prefix='module.')

    if hparams.appearance_dim > 0:
        appearance_count = len(loaded['model_state_dict']['embedding_a.weight'])
    else:
        appearance_count = 0

    sub_module = get_nerf(hparams, appearance_count)
    model_dict = sub_module.state_dict()
    model_dict.update(loaded['model_state_dict'])
    sub_module.load_state_dict(model_dict)

    if 'bg_model_state_dict' in loaded:
        consume_prefix_in_state_dict_if_present(loaded['bg_model_state_dict'], prefix='module.')
        bg_sub_module = get_bg_nerf(hparams, appearance_count)
        model_dict = bg_sub_module.state_dict()
        model_dict.update(loaded['bg_model_state_dict'])
        bg_sub_module.load_state_dict(model_dict)

    container = MegaNeRFContainer([sub_module], [bg_sub_module] if 'bg_model_state_dict' in loaded else [], centroids,
                                  torch.IntTensor([1, 1]),
                                  torch.zeros(3),
                                  torch.ones(3),
                                  hparams.pos_dir_dim > 0,
                                  hparams.appearance_dim > 0,
                                  False)
    torch.jit.save(torch.jit.script(container.eval()), hparams.output)
    container = torch.jit.load(hparams.output, map_location='cpu')

    # Test container
    nerf = MegaNeRF([getattr(container, 'sub_module_{}'.format(i)) for i in range(len(container.centroids))],
                    container.centroids, hparams.boundary_margin, False, container.cluster_2d).to(device)

    width = 3
    if hparams.pos_dir_dim > 0:
        width += 3
    if hparams.appearance_dim > 0:
        width += 1

    print('fg test eval: {}'.format(nerf(torch.ones(1, width, device=device))))

    if 'bg_model_state_dict' in loaded:
        bg_nerf = MegaNeRF([getattr(container, 'bg_sub_module_{}'.format(i)) for i in range(len(container.centroids))],
                           container.centroids, hparams.boundary_margin, True, container.cluster_2d).to(device)

        width += 4
        print('bg test eval: {}'.format(bg_nerf(torch.ones(1, width, device=device))))


if __name__ == '__main__':
    main(_get_merge_opts())
