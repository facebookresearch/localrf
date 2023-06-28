# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2020 Virginia Tech Vision and Learning Lab

import numpy as np
import sys
import os
import cv2

sys.path.append('.')
sys.path.append('RAFT/core')
from RAFT.core.raft import RAFT
from RAFT.core.utils.utils import InputPadder
from RAFT.core.utils import flow_viz

from localTensoRF.utils.utils import encode_flow

import argparse
import torch
from tqdm.auto import tqdm


def warp_flow(img, flow):
    h, w = flow.shape[:2]
    flow_new = flow.copy()
    flow_new[:,:,0] += np.arange(w)
    flow_new[:,:,1] += np.arange(h)[:,np.newaxis]

    res = cv2.remap(img, flow_new, None, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return res

def compute_fwdbwd_mask(fwd_flow, bwd_flow, alpha_1=0.05, alpha_2=0.5):
    bwd2fwd_flow = warp_flow(bwd_flow, fwd_flow)
    fwd_lr_error = np.linalg.norm(fwd_flow + bwd2fwd_flow, axis=-1)
    fwd_mask = fwd_lr_error < alpha_1  * (np.linalg.norm(fwd_flow, axis=-1) \
                + np.linalg.norm(bwd2fwd_flow, axis=-1)) + alpha_2

    fwd2bwd_flow = warp_flow(fwd_flow, bwd_flow)
    bwd_lr_error = np.linalg.norm(bwd_flow + fwd2bwd_flow, axis=-1)

    bwd_mask = bwd_lr_error < alpha_1  * (np.linalg.norm(bwd_flow, axis=-1) \
                + np.linalg.norm(fwd2bwd_flow, axis=-1)) + alpha_2

    return fwd_mask, bwd_mask


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='/data/forest1')
    parser.add_argument('--scale', type=float, default=0.5)
    parser.add_argument('--frame_step', type=int, default=1, help="Step between retained frames")
    parser.add_argument('--device_ids', type=int, default=[0], nargs='+')
    parser.add_argument('--raft_model', default='models/raft-things.pth', help="[RAFT] restore checkpoint")
    parser.add_argument('--small', action='store_true', help='[RAFT] use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='[RAFT] use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='[RAFT] use efficent correlation implementation')
    args = parser.parse_args()

    # Initialize optical flow model
    raft_model = torch.nn.DataParallel(RAFT(args), device_ids=args.device_ids)
    raft_model.load_state_dict(torch.load(args.raft_model))
    raft_model.module.to(f"cuda:{args.device_ids[0]}")
    raft_model.eval()

    # Read and preprocess the video
    input_files = sorted(os.listdir(f"{args.data_dir}/images"))
    input_files = input_files[::args.frame_step]
    os.makedirs(f"{args.data_dir}/flow_ds", exist_ok=True)
    os.makedirs(f"{args.data_dir}/flow_vis", exist_ok=True)
    prev_frame_torch = None
    ret = True
    for filename in tqdm(input_files):
        # Read and rescale frame
        frame = cv2.imread(f"{args.data_dir}/images/{filename}")
        ds_frame = cv2.resize(frame, None, fx=args.scale, fy=args.scale, interpolation=cv2.INTER_CUBIC)
        frame_torch = torch.from_numpy(ds_frame[..., ::-1].copy()).permute(2, 0, 1).float()[None].to(f"cuda:{args.device_ids[0]}")
        
        # get optical flow
        if prev_frame_torch is not None:
            with torch.no_grad():
                image1 = torch.cat([prev_frame_torch, frame_torch], dim=0)
                image2 = torch.cat([frame_torch, prev_frame_torch], dim=0)
                padder = InputPadder(image1.shape)
                image1, image2 = padder.pad(image1, image2)
                _, flow_up = raft_model(image1, image2, iters=30, test_mode=True)
                fwd_flow = padder.unpad(flow_up[0]).permute(1,2,0).cpu().numpy()
                bwd_flow = padder.unpad(flow_up[1]).permute(1,2,0).cpu().numpy()

            mask_fwd, mask_bwd = compute_fwdbwd_mask(fwd_flow, bwd_flow)
        else:
            fwd_flow = np.zeros(frame[..., :2].shape, dtype=np.float32)
            bwd_flow = np.zeros(frame[..., :2].shape, dtype=np.float32)
            mask_fwd = np.zeros(frame[..., 0].shape, dtype=bool)
            mask_bwd = np.zeros(frame[..., 0].shape, dtype=bool)

        # Save flow
        fbase = os.path.splitext(filename)[0]
        if args.frame_step != 1:
            fbase = f"step{args.frame_step}_{fbase}"
        cv2.imwrite(f"{args.data_dir}/flow_ds/fwd_{fbase}.png", encode_flow(fwd_flow, mask_fwd))
        cv2.imwrite(f"{args.data_dir}/flow_ds/bwd_{fbase}.png", encode_flow(bwd_flow, mask_bwd))
        cv2.imwrite(f"{args.data_dir}/flow_vis/fwd_{fbase}.jpg", flow_viz.flow_to_image(fwd_flow))
        cv2.imwrite(f"{args.data_dir}/flow_vis/bwd_{fbase}.jpg", flow_viz.flow_to_image(bwd_flow))
        prev_frame_torch = frame_torch.clone()