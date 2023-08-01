# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2022 Anpei Chen

import os
from signal import Handlers

import cv2
import imageio
import numpy as np
import torch
from tqdm.auto import tqdm

from utils.utils import (compute_depth_loss, get_fwd_bwd_cam2cams, get_pred_flow, rgb_ssim, visualize_depth, draw_poses)

@torch.no_grad()
def render(
    test_dataset,
    poses_mtx,
    local_tensorfs,
    args,
    W, H,
    frame_indices=None,
    savePath=None,
    save_video=False, # Set False to save RAM
    save_frames=False,
    test=False,
    train_dataset=None,
    world2rf=None,
    img_format="jpg",
    annotate=False,
    save_raw_depth=False,
    start=0,
    floater_thresh=0,
    add_frame_to_list=True, # Set False to save RAM. Set True for Tensorboard.
):
    rgb_maps_tb, depth_maps_tb, gt_rgbs_tb, poses_vis = [], [], [], []
    fwd_flow_cmp_tb, bwd_flow_cmp_tb, depth_cmp_tb = [], [], []

    if test:
        idxs = [train_dataset.all_fbases[fbase] for fbase in test_dataset.all_fbases]
        idxs = [idx for idx in idxs if start <= idx < poses_mtx.shape[0]]
    else:
        poses_mtx = poses_mtx[start:]
        idxs = list(range(start, poses_mtx.shape[0]))
        is_test_id = [fbase in test_dataset.all_fbases for fbase in train_dataset.all_fbases]
        if frame_indices is None:
            frame_indices = []
            for pose in poses_mtx:
                t_c2w = torch.stack(list(local_tensorfs.t_c2w), dim=0)
                distances_to_poses = torch.norm(t_c2w - pose[None, :, 3], dim=-1)
                frame_indices.append(torch.argmin(distances_to_poses).item())
            frame_indices = torch.Tensor(frame_indices).to(poses_mtx).long()

    N_rays_all = W * H
    rays_ids = torch.arange(N_rays_all, dtype=torch.long, device=poses_mtx.device)
    metrics = {}
    print(f"Render {len(idxs)} frame with size {W} x {H}")
    for i, idx in tqdm(enumerate(idxs)):
        torch.cuda.empty_cache()
        if frame_indices is None:
            view_ids = torch.Tensor([idx]).to(poses_mtx).long()
        else:
            view_ids = frame_indices[idx][None]

        rgb_map, depth_map, directions, ij = local_tensorfs(
            rays_ids,
            view_ids,
            W,
            H,
            is_train=False,
            cam2world=None if test else poses_mtx[i][None],
            world2rf=world2rf,
            blending_weights=None,
            test_id=test or is_test_id[view_ids.item()],
            chunk=args.batch_size,
            floater_thresh=floater_thresh,
        )              
        
        if test and add_frame_to_list:
            fbase = train_dataset.get_frame_fbase(idx)
            # Flow render
            if test_dataset.all_fwd_flow is not None:
                view_ids = torch.Tensor([idx]).to(depth_map).long()
                cam2world = local_tensorfs.get_cam2world()
                fwd_cam2cams, bwd_cam2cams = get_fwd_bwd_cam2cams(cam2world, view_ids)
                pts = directions[None] * depth_map[None, ..., None]
                center = local_tensorfs.center(W, H)
                pred_fwd_flow = get_pred_flow(pts, ij, fwd_cam2cams, local_tensorfs.focal(W), center).cpu().numpy()
                pred_bwd_flow = get_pred_flow(pts, ij, bwd_cam2cams, local_tensorfs.focal(W), center).cpu().numpy()
                pred_fwd_flow, pred_bwd_flow = pred_fwd_flow.reshape(H, W, 2), pred_bwd_flow.reshape(H, W, 2)
                fwd_flow = cv2.resize(test_dataset.all_fwd_flow[test_dataset.all_fbases[fbase]], (W, H), interpolation=cv2.INTER_NEAREST) 
                fwd_mask = cv2.resize(test_dataset.all_fwd_mask[test_dataset.all_fbases[fbase]], (W, H), interpolation=cv2.INTER_NEAREST)
                bwd_flow = cv2.resize(test_dataset.all_bwd_flow[test_dataset.all_fbases[fbase]], (W, H), interpolation=cv2.INTER_NEAREST)
                bwd_mask = cv2.resize(test_dataset.all_bwd_mask[test_dataset.all_fbases[fbase]], (W, H), interpolation=cv2.INTER_NEAREST)
                fwd_flow_cmp0 = np.vstack([pred_fwd_flow[..., 0], fwd_flow[..., 0]])
                fwd_flow_cmp0 /= np.quantile(fwd_flow_cmp0, 0.9)
                fwd_flow_err0 = np.abs(pred_fwd_flow[..., 0] - fwd_flow[..., 0]) * fwd_mask / W
                fwd_flow_cmp0 = np.vstack([fwd_flow_cmp0, fwd_flow_err0])
                fwd_flow_cmp1 = np.vstack([pred_fwd_flow[..., 1], fwd_flow[..., 1]])
                fwd_flow_cmp1 /= np.quantile(fwd_flow_cmp1, 0.9)
                fwd_flow_err1 = np.abs(pred_fwd_flow[..., 1] - fwd_flow[..., 1]) * fwd_mask / W
                fwd_flow_cmp1 = np.vstack([fwd_flow_cmp1, fwd_flow_err1])
                fwd_flow_cmp = np.hstack([fwd_flow_cmp0, fwd_flow_cmp1])

                bwd_flow_cmp0 = np.vstack([pred_bwd_flow[..., 0], bwd_flow[..., 0]])
                bwd_flow_cmp0 /= np.quantile(bwd_flow_cmp0, 0.9)
                bwd_flow_err0 = np.abs(pred_bwd_flow[..., 0] - bwd_flow[..., 0]) * bwd_mask / W
                bwd_flow_cmp0 = np.vstack([bwd_flow_cmp0, bwd_flow_err0])
                bwd_flow_cmp1 = np.vstack([pred_bwd_flow[..., 1], bwd_flow[..., 1]])
                bwd_flow_cmp1 /= np.quantile(bwd_flow_cmp1, 0.9)
                bwd_flow_err1 = np.abs(pred_bwd_flow[..., 1] - bwd_flow[..., 1]) * bwd_mask / W
                bwd_flow_cmp1 = np.vstack([bwd_flow_cmp1, bwd_flow_err1])
                bwd_flow_cmp = np.hstack([bwd_flow_cmp0, bwd_flow_cmp1])
                fwd_flow_cmp_tb.append(torch.from_numpy(fwd_flow_cmp).clamp(0, 1)) # Only need this one for TensorBoard
                bwd_flow_cmp_tb.append(torch.from_numpy(bwd_flow_cmp).clamp(0, 1)) # Only need this one for TensorBoard

            # Depth error
            if test_dataset.all_invdepths is not None:
                invdepths = torch.from_numpy(cv2.resize(test_dataset.all_invdepths[test_dataset.all_fbases[fbase]], (W, H), interpolation=cv2.INTER_NEAREST)).to(args.device)
                invdepths = invdepths.view(1, -1)
                dyn_depth_norm, gt_depth_norm, depth_loss_arr = compute_depth_loss(1 / depth_map[None].clamp(1e-6), invdepths)

                depth_cmp = torch.vstack([0.5 * dyn_depth_norm[0].reshape(H, W), 0.5 * gt_depth_norm[0].reshape(H, W), depth_loss_arr[0].reshape(H, W)])
                depth_cmp_tb.append(depth_cmp.clamp(0, 1)) # Only need this one for TensorBoard

        # RGB and depth visualization
        rgb_map, depth_map = rgb_map.reshape(H, W, 3), depth_map.reshape(H, W)
        depth_map_vis, _ = visualize_depth(depth_map.cpu().numpy(), [0, 5])

        rgb_map = rgb_map.detach().cpu()
        if annotate:
            rgb_map = (rgb_map.detach().cpu() * 255).byte().numpy()
            weights = local_tensorfs.module.blending_weights[idx].cpu()
            rf_ids = torch.nonzero(weights)[:, 0]
            weights = [round(weight.item(), 1) for weight in weights[rf_ids]]
            cv2.putText(rgb_map, f"id: {idx}", [1, H-70], 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(rgb_map, f"RFs: {rf_ids.tolist()}", [1, H-40], 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(rgb_map, f"W: {weights}", [1, H-10], 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            rgb_map = torch.Tensor(rgb_map) / 255

        all_poses = torch.cat([poses_mtx, poses_mtx[idx][None]], dim=0)
        colours = ["C1"] * poses_mtx.shape[0] + ["C2"]
        pose_vis = draw_poses(all_poses.cpu(), colours)
        pose_vis = cv2.resize(pose_vis, (int(pose_vis.shape[1] * rgb_map.shape[0] / pose_vis.shape[0]), rgb_map.shape[0]))
        depth_map_vis = torch.permute(depth_map_vis.detach().cpu() * 255, [1, 2, 0]).byte()
        if add_frame_to_list or (save_video and savePath is not None):
            rgb_maps_tb.append(rgb_map)  # HWC
            depth_maps_tb.append(depth_map_vis)  # HWC
            poses_vis.append(pose_vis)

        if test:
            fbase = train_dataset.get_frame_fbase(idx)
            gt_rgb = test_dataset.all_rgbs[test_dataset.all_fbases[fbase]]
            gt_rgb = cv2.resize(gt_rgb, (W, H))
            gt_rgb = torch.from_numpy(gt_rgb)
            if add_frame_to_list:
                gt_rgbs_tb.append(torch.Tensor(gt_rgb))  # HWC

            mse = ((gt_rgb - rgb_map) ** 2).mean()
            ssim = rgb_ssim(gt_rgb.numpy(), rgb_map.numpy(), 1)
            metrics[fbase] = {
                "mse": mse,
                "ssim": ssim,
            }

        if save_frames and savePath is not None:
            if not test:
                fbase = f"{i:06d}"
            os.makedirs(f"{savePath}/rgb_maps", exist_ok=True)
            os.makedirs(f"{savePath}/depth_maps", exist_ok=True)
            cv2.imwrite(f"{savePath}/rgb_maps/{fbase}.{img_format}", 255 * rgb_map.numpy()[..., ::-1])
            cv2.imwrite(f"{savePath}/rgb_maps/{fbase}_pose.{img_format}", pose_vis[..., ::-1])
            cv2.imwrite(f"{savePath}/depth_maps/{fbase}.{img_format}", depth_map_vis.numpy()[..., ::-1])
            if save_raw_depth:
                cv2.imwrite(f"{savePath}/depth_maps/{fbase}.tiff", depth_map.cpu().numpy())

    if save_video and savePath is not None:
        os.makedirs(savePath, exist_ok=True)

        with open(f"{savePath}/video.mp4", "wb") as f:
            imageio.mimwrite(f, np.stack(rgb_maps_tb), fps=30, quality=6, format="mp4", output_params=["-f", "mp4"])
        with open(f"{savePath}/posevideo.mp4", "wb") as f:
            imageio.mimwrite(f, np.stack(poses_vis), fps=30, quality=6, format="mp4", output_params=["-f", "mp4"])
        with open(f"{savePath}/depthvideo.mp4", "wb") as f:
            imageio.mimwrite(f, np.stack(depth_maps_tb), fps=30, quality=6, format="mp4", output_params=["-f", "mp4"])

    return rgb_maps_tb, depth_maps_tb, gt_rgbs_tb, fwd_flow_cmp_tb, bwd_flow_cmp_tb, depth_cmp_tb, metrics