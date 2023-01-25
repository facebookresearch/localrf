#!/usr/bin/env python3
import imageio
import numpy as np
import sys
sys.path.append('RAFT/core')
sys.path.append('DPT')
import os
import cv2

from RAFT.core.raft import RAFT
from RAFT.core.utils.utils import InputPadder
from RAFT.core.utils import flow_viz

from localTensoRF.utils.utils import encode_flow

import argparse
import torch
from tqdm.auto import tqdm

# VIDEO_NAME = "hike_07_08_gopro_4"
# # VIDEO_NAME = "hike_07_08_2"
# VIDEO_NAME = "office"

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
    parser.add_argument('--video_name', default='intermediate/M60')
    parser.add_argument('--save_format', default='jpg')
    parser.add_argument('--h', type=int, default=1080)
    parser.add_argument('--raft_h', type=int, default=540)
    parser.add_argument('--multiGPU', type=int, default=[0], nargs='+')
    parser.add_argument('--max_frames', type=int, default=1000)
    parser.add_argument('--skip', type=int, default=4)
    parser.add_argument('--raft_model', default='checkpoints/raft-things.pth', help="restore checkpoint")
    parser.add_argument('--dpt_model', default='checkpoints/dpt_large-midas-2f21e586.pt')
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    # Initialize optical flow model
    raft_model = torch.nn.DataParallel(RAFT(args), device_ids=args.multiGPU)
    raft_model.load_state_dict(torch.load(args.raft_model))
    raft_model.module.to(f"cuda:{args.multiGPU[0]}")
    raft_model.eval()

    # Read and preprocess the video
    print(f"Read data/videos/{args.video_name}.mp4")
    cap = cv2.VideoCapture(f"data/videos/{args.video_name}.mp4")
    save_folder = f"data/sequenced/{args.video_name}/skip_{args.skip}"
    if not cap.isOpened():
        input_files = sorted(os.listdir(f"data/sequenced/{args.video_name}/images"))
        input_files = input_files[::(args.skip + 1)]
    os.makedirs(f"{save_folder}/images", exist_ok=True)
    os.makedirs(f"{save_folder}/flow_ds", exist_ok=True)
    os.makedirs(f"{save_folder}/depth", exist_ok=True)
    os.makedirs(f"{save_folder}/flow_vis", exist_ok=True)
    prev_frame_torch = None
    pbar = tqdm()
    for idx in range(args.max_frames):
        # Read and rescale frame
        if cap.isOpened():
            ret, frame = cap.read()
            for _ in range(args.skip):
                cap.read()
        else:
            ret = idx < len(input_files)
            if ret:
                frame = cv2.imread(f"data/sequenced/{args.video_name}/images/{input_files[idx]}")
        if ret:
            scale = args.h / frame.shape[0]
            if scale != 1:
                frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            raft_scale = args.raft_h / frame.shape[0]
            ds_frame = cv2.resize(frame, None, fx=raft_scale, fy=raft_scale, interpolation=cv2.INTER_CUBIC)
            frame_torch = torch.from_numpy(ds_frame[..., ::-1].copy()).permute(2, 0, 1).float()[None].to(f"cuda:{args.multiGPU[0]}")
           
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
                mask_fwd = np.zeros(frame[..., 0].shape, dtype=np.bool)
                mask_bwd = np.zeros(frame[..., 0].shape, dtype=np.bool)

            # Save the images and flow
            cv2.imwrite(f"{save_folder}/images/{idx:06d}.{args.save_format}", frame)
            cv2.imwrite(f"{save_folder}/flow_ds/fwd_{idx:06d}.png", encode_flow(fwd_flow, mask_fwd))
            cv2.imwrite(f"{save_folder}/flow_ds/bwd_{idx:06d}.png", encode_flow(bwd_flow, mask_bwd))
            cv2.imwrite(f"{save_folder}/flow_vis/fwd_{idx:06d}.jpg", flow_viz.flow_to_image(fwd_flow))
            cv2.imwrite(f"{save_folder}/flow_vis/bwd_{idx:06d}.jpg", flow_viz.flow_to_image(bwd_flow))
            idx += 1
            pbar.update(1)
            prev_frame_torch = frame_torch.clone()
        if not ret:
            break
    
    pbar.close()
    cap.release()