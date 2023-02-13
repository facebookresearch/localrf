import os
from signal import Handlers

import cv2
import imageio
import numpy as np
import torch
from tqdm.auto import tqdm

from utils.utils import (rgb_ssim, visualize_depth, draw_poses)

@torch.no_grad()
def render(
    test_dataset,
    poses_mtx,
    local_tensorfs,
    args,
    frame_indices=None,
    savePath=None,
    save_video=False,
    save_frames=False,
    test=False,
    train_dataset=None,
    world2rf=None,
    all_blending_weights=None,
    img_format="jpg",
    annotate=False,
    save_raw_depth=False,
    start=0
):
    rgb_maps_tb, depth_maps_tb, gt_rgbs_tb, poses_vis = [], [], [], []
    W, H = test_dataset.img_wh

    if world2rf is not None:
        world2rf = world2rf.repeat(len(local_tensorfs.device_ids), 1, 1)

    if test:
        # TODO: Single dataset for train & test?
        idxs = [train_dataset.all_fbases[fbase] for fbase in test_dataset.all_fbases]
        idxs = [idx for idx in idxs if start <= idx < poses_mtx.shape[0]]
    else:
        idxs = list(range(start, poses_mtx.shape[0]))
        is_test_id = [fbase in test_dataset.all_fbases for fbase in train_dataset.all_fbases]

    N_rays_all = W * H
    rays_ids = torch.arange(N_rays_all, dtype=torch.long, device=poses_mtx.device)
    metrics = {}
    for idx in tqdm(idxs):
        torch.cuda.empty_cache()
        if frame_indices is None:
            rays_ids_img = rays_ids + N_rays_all * idx
            view_ids = idx * torch.ones([max(len(local_tensorfs.device_ids), 1)], device=poses_mtx.device).long()
        else:
            rays_ids_img = rays_ids + N_rays_all * frame_indices[idx]
            view_ids = frame_indices[idx] * torch.ones([max(len(local_tensorfs.device_ids), 1)], device=poses_mtx.device).long()

        if all_blending_weights is not None:
            blending_weights = all_blending_weights[idx][None].expand(
                N_rays_all, all_blending_weights.shape[1]
            )
        else:
            blending_weights = None

        rgb_map, depth_map, _, _ = local_tensorfs(
            rays_ids_img,
            view_ids,
            W,
            H,
            is_train=False,
            cam2world=poses_mtx[idx][None].repeat(max(len(local_tensorfs.device_ids), 1), 1, 1),
            world2rf=world2rf,
            blending_weights=blending_weights,
            test_id=test or is_test_id[idx],
            chunk=args.batch_size,
        )              

        rgb_map = rgb_map.clamp(0.0, 1.0)
        rgb_map, depth_map = rgb_map.reshape(H, W, 3), depth_map.reshape(H, W)
         
        # # Flow render
        # view_ids = torch.Tensor([idx]).to(depth_map).long()
        # cam2world = local_tensorfs.module.get_cam2world()
        # fwd_cam2cams, bwd_cam2cams = get_fwd_bwd_cam2caget_gtms(cam2world, view_ids)
        # pts = directions[None] * depth_map_gamma[None, ..., None]
        # center = local_tensorfs.module.center(W, H)
        # pred_fwd_flow = get_pred_flow(pts, ij, fwd_cam2cams, local_tensorfs.module.focal(W, H), center).cpu().numpy()
        # pred_bwd_flow = get_pred_flow(pts, ij, bwd_cam2cams, local_tensorfs.module.focal(W, H), center).cpu().numpy()
        # pred_fwd_flow, pred_bwd_flow = pred_fwd_flow.reshape(H, W, 2), pred_bwd_flow.reshape(H, W, 2)
        # # pred_fwd_flows_tb.append(pred_fwd_flow)
        # # pred_bwd_flows_tb.append(pred_bwd_flow)

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
        rgb_maps_tb.append(rgb_map)  # HWC
        depth_maps_tb.append(depth_map_vis)  # HWC
        poses_vis.append(pose_vis)

        if test:
            torch.cuda.empty_cache()
            fbase = train_dataset.get_frame_fbase(idx)
            gt_rgb = torch.from_numpy(test_dataset.all_rgbs[test_dataset.all_fbases[fbase]])
            gt_rgbs_tb.append(torch.Tensor(gt_rgb))  # HWC
            mse = ((gt_rgb - rgb_map) ** 2).mean()
            ssim = rgb_ssim(gt_rgb.numpy(), rgb_map.numpy(), 1)
            metrics[fbase] = {
                "mse": mse,
                "ssim": ssim,
            }

        if save_frames and savePath is not None:
            if not test:
                fbase = f"{idx:06d}"
            os.makedirs(f"{savePath}/rgb_maps", exist_ok=True)
            os.makedirs(f"{savePath}/depth_maps", exist_ok=True)
            cv2.imwrite(f"{savePath}/rgb_maps/{fbase}.{img_format}", 255 * rgb_map.numpy()[..., ::-1])
            cv2.imwrite(f"{savePath}/rgb_maps/{fbase}_pose.{img_format}", pose_vis[..., ::-1])
            cv2.imwrite(f"{savePath}/depth_maps/{fbase}.{img_format}", depth_map_vis.numpy()[..., ::-1])
            if save_raw_depth:
                cv2.imwrite(f"{savePath}/depth_maps/{fbase}.tiff", depth_map.cpu().numpy())

            # os.makedirs(f"{savePath}/fwd_flows", exist_ok=True)
            # os.makedirs(f"{savePath}/bwd_flows", exist_ok=True)

            # if gt_frame["fwd_flow"] is not None:
            #     pred_fwd_flow = np.hstack([pred_fwd_flow, gt_frame["fwd_flow"]])
            #     pred_bwd_flow = np.hstack([pred_bwd_flow, gt_frame["bwd_flow"]])
            #     cv2.imwrite(f"{savePath}/fwd_flows/mask_{idx:03d}.png", 255 * gt_frame["fwd_mask"].astype(np.uint8))
            #     cv2.imwrite(f"{savePath}/bwd_flows/mask_{idx:03d}.png", 255 * gt_frame["bwd_mask"].astype(np.uint8))

            # min_fwd_flow, max_fwd_flow = np.quantile(pred_fwd_flow, 0.05), np.quantile(pred_fwd_flow, 0.95)
            # min_bwd_flow, max_bwd_flow = np.quantile(pred_bwd_flow, 0.05), np.quantile(pred_bwd_flow, 0.95)

            # pred_fwd_flow = 255 * (pred_fwd_flow - min_fwd_flow) / (max_fwd_flow - min_fwd_flow)
            # pred_bwd_flow = 255 * (pred_bwd_flow - min_bwd_flow) / (max_bwd_flow - min_bwd_flow)

            # cv2.imwrite(f"{savePath}/fwd_flows/flow_x{idx:03d}.{img_format}", pred_fwd_flow[..., 0])
            # cv2.imwrite(f"{savePath}/fwd_flows/flow_y{idx:03d}.{img_format}", pred_fwd_flow[..., 1])
            # cv2.imwrite(f"{savePath}/bwd_flows/flow_x{idx:03d}.{img_format}", pred_bwd_flow[..., 0])
            # cv2.imwrite(f"{savePath}/bwd_flows/flow_y{idx:03d}.{img_format}", pred_bwd_flow[..., 1])

    if save_video and savePath is not None:
        os.makedirs(savePath, exist_ok=True)

        with open(f"{savePath}/video.mp4", "wb") as f:
            imageio.mimwrite(f, np.stack(rgb_maps_tb), fps=30, quality=6, format="mp4", output_params=["-f", "mp4"])
        with open(f"{savePath}/posevideo.mp4", "wb") as f:
            imageio.mimwrite(f, np.stack(poses_vis), fps=30, quality=6, format="mp4", output_params=["-f", "mp4"])
        with open(f"{savePath}/depthvideo.mp4", "wb") as f:
            imageio.mimwrite(f, np.stack(depth_maps_tb), fps=30, quality=6, format="mp4", output_params=["-f", "mp4"])

    torch.cuda.empty_cache()
    return rgb_maps_tb, depth_maps_tb, gt_rgbs_tb, metrics