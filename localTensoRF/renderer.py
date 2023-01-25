import os
from signal import Handlers

import cv2
import imageio
import numpy as np
import torch
from tqdm.auto import tqdm
from utils.camera import get_fwd_bwd_cam2cams

# from models.tensorBase import render_from_samples
from utils.ray_utils import (get_ray_directions_blender, get_rays,
                             ndc_rays_blender)
from utils.utils import (get_pred_flow, rgb_lpips, rgb_ssim, visualize_depth, draw_poses,
                         visualize_depth_numpy)

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
    near_far = test_dataset.near_far
    W, H = test_dataset.img_wh

    if world2rf is not None:
        world2rf = world2rf.repeat(len(local_tensorfs.device_ids), 1, 1)

    if test:
        idxs = [train_dataset.all_fbases[fbase] for fbase in test_dataset.all_fbases]
        idxs = [idx for idx in idxs if start <= idx < poses_mtx.shape[0]]
    else:
        idxs = list(range(start, poses_mtx.shape[0]))
    N_rays_all = W * H
    rays_ids = torch.arange(N_rays_all, dtype=torch.long, device=poses_mtx.device)
    metrics = {}
    alpha, T, weight, alpha_new, T_new, weight_new = None, None, None, None, None, None
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

        rgb_map, depth_map, depth_map_gamma, directions, ij, a, b, c, d, e, f = local_tensorfs(
            rays_ids_img,
            view_ids,
            W,
            H,
            white_bg=args.white_bkgd,
            is_train=False,
            cam2world=poses_mtx[idx][None].repeat(max(len(local_tensorfs.device_ids), 1), 1, 1),
            world2rf=world2rf,
            blending_weights=blending_weights,
            opasity_gamma=args.opasity_gamma,
            test=test,
        )
        if idx == idxs[0]:
            alpha, T, weight, alpha_new, T_new, weight_new = a, b, c, d, e, f
                

        rgb_map = rgb_map.clamp(0.0, 1.0)
        rgb_map, depth_map = rgb_map.reshape(H, W, 3), depth_map.reshape(H, W)
        # cv2.imwrite("test.jpg", depth_map.detach().cpu().numpy()*128)
         
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

        if args.ray_type == "contract":
            depth_map_vis, _ = visualize_depth(depth_map.cpu().numpy(), [0, 5])
        else:
            depth_map_vis, _ = visualize_depth(depth_map.cpu().numpy(), near_far)

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
            # gt_frame = test_dataset.get_gt_frame(np.array([idx], dtype=np.int64))
            gt_rgb = torch.from_numpy(test_dataset.all_rgbs[test_dataset.all_fbases[fbase]])
            gt_rgbs_tb.append(torch.Tensor(gt_rgb))  # HWC
            mse = ((gt_rgb - rgb_map) ** 2).mean()
            ssim = rgb_ssim(gt_rgb.numpy(), rgb_map.numpy(), 1)
            lpips = 0 # rgb_lpips(gt_rgb.numpy(), rgb_map.numpy(), "vgg", poses_mtx.device)
            metrics[fbase] = {
                "mse": mse,
                "ssim": ssim,
                "lpips": lpips,
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
    return rgb_maps_tb, depth_maps_tb, gt_rgbs_tb, alpha, T, weight, alpha_new, T_new, weight_new, metrics


# @torch.no_grad()
# def evaluation(
#     test_dataset,
#     poses_mtx,
#     tensorf,
#     args,
#     renderer,
#     savePath=None,
#     N_vis=5,
#     prtx="",
#     N_samples=-1,
#     white_bg=False,
#     ray_type="ndc",
#     compute_extra_metrics=True,
#     device="cuda",
# ):
#     PSNRs, rgb_maps, depth_maps = [], [], []
#     ssims, l_alex, l_vgg = [], [], []
#     os.makedirs(savePath, exist_ok=True)
#     os.makedirs(savePath + "/rgbd", exist_ok=True)

#     try:
#         tqdm._instances.clear()
#     except Exception:
#         pass

#     near_far = test_dataset.near_far

#     W, H = test_dataset.img_wh
#     directions = get_ray_directions_blender(H, W, test_dataset.focal).to(
#         poses_mtx.device
#     )  # (H, W, 3)
#     all_rays = []
#     for i in range(poses_mtx.shape[0]):
#         c2w = poses_mtx[i]
#         rays_o, rays_d = get_rays(directions, c2w)  # both (h*w, 3)
#         if ray_type == "ndc":
#             rays_o, rays_d = ndc_rays_blender(
#                 H, W, test_dataset.focal[0], 1.0, rays_o, rays_d
#             )
#         all_rays += [torch.cat([rays_o, rays_d], 1)]  # (h*w, 6)
#     all_rays = torch.stack(all_rays, 0)

#     img_eval_interval = 1 if N_vis < 0 else max(all_rays.shape[0] // N_vis, 1)
#     idxs = list(range(0, all_rays.shape[0], img_eval_interval))
#     for idx, samples in tqdm(
#         enumerate(all_rays[0::img_eval_interval]), file=sys.stdout
#     ):

#         W, H = test_dataset.img_wh
#         rays = samples.view(-1, samples.shape[-1])
#         ts = test_dataset.all_ts[idx].view(-1)

#         rgb_map, _, depth_map, _, _, _, _ = renderer(
#             rays,
#             ts,
#             tensorf,
#             chunk=2048,
#             N_samples=N_samples,
#             ray_type=ray_type,
#             white_bg=white_bg,
#             device=device,
#         )
#         rgb_map = rgb_map.clamp(0.0, 1.0)

#         rgb_map, depth_map = (
#             rgb_map.reshape(H, W, 3).cpu(),
#             depth_map.reshape(H, W).cpu(),
#         )

#         with open(f"{savePath}/rgbd/{prtx}{idx:03d}.npy", "wb") as f:
#             np.save(f, depth_map)

#         depth_map, _ = visualize_depth_numpy(depth_map.numpy(), near_far)
#         if len(test_dataset.all_rgbs):
#             gt_rgb = test_dataset.all_rgbs[idxs[idx]].view(H, W, 3)
#             loss = torch.mean((rgb_map - gt_rgb) ** 2)
#             PSNRs.append(-10.0 * np.log(loss.item()) / np.log(10.0))

#             if compute_extra_metrics:
#                 ssim = rgb_ssim(rgb_map, gt_rgb, 1)
#                 l_a = rgb_lpips(
#                     gt_rgb.numpy(),
#                     rgb_map.numpy(),
#                     "alex",
#                     tensorf.device if args.multiGPU is None else tensorf.module.device,
#                 )
#                 l_v = rgb_lpips(
#                     gt_rgb.numpy(),
#                     rgb_map.numpy(),
#                     "vgg",
#                     tensorf.device if args.multiGPU is None else tensorf.module.device,
#                 )
#                 ssims.append(ssim)
#                 l_alex.append(l_a)
#                 l_vgg.append(l_v)

#         rgb_map = (rgb_map.numpy() * 255).astype("uint8")
#         # rgb_map = np.concatenate((rgb_map, depth_map), axis=1)
#         rgb_maps.append(rgb_map)
#         depth_maps.append(depth_map)
#         if os.is_dir(savePath):
#             with pmgr.open(f"{savePath}/{prtx}{idx:03d}.png", "wb") as f:
#                 imageio.imwrite(f, rgb_map, format="png")
#             rgb_map = np.concatenate((rgb_map, depth_map), axis=1)
#             with pmgr.open(f"{savePath}/rgbd/{prtx}{idx:03d}.png", "wb") as f:
#                 imageio.imwrite(f, rgb_map, format="png")

#     with pmgr.open(f"{savePath}/{prtx}video.mp4", "wb") as f:
#         imageio.mimwrite(
#             f,
#             np.stack(rgb_maps),
#             fps=30,
#             quality=10,
#             format="ffmpeg",
#             output_params=["-f", "mp4"],
#         )
#     with pmgr.open(f"{savePath}/{prtx}depthvideo.mp4", "wb") as f:
#         imageio.mimwrite(
#             f,
#             np.stack(depth_maps),
#             fps=30,
#             quality=10,
#             format="ffmpeg",
#             output_params=["-f", "mp4"],
#         )

#     if PSNRs:
#         psnr = np.mean(np.asarray(PSNRs))
#         if compute_extra_metrics:
#             ssim = np.mean(np.asarray(ssims))
#             l_a = np.mean(np.asarray(l_alex))
#             l_v = np.mean(np.asarray(l_vgg))
#             with pmgr.open(f"{savePath}/{prtx}mean.txt", "wb") as f:
#                 np.savetxt(f, np.asarray([psnr, ssim, l_a, l_v]))
#         else:
#             with pmgr.open(f"{savePath}/{prtx}mean.txt", "wb") as f:
#                 np.savetxt(f, np.asarray([psnr]))

#     return PSNRs

            #     else:
            #         N_samples = max(self.nSamples) // 2 * 2
            #         rgb = torch.zeros([rays.shape[0], N_samples, 3], device=rays.device)
            #         sigma = torch.zeros([rays.shape[0], N_samples], device=rays.device)
            #         z_vals, rgb[valid, :, :], sigma[valid, :], _ = self.tensorfs[rf_id](
            #             rays[valid],
            #             is_train=is_train,
            #             white_bg=white_bg,
            #             ray_type=self.ray_type,
            #             N_samples=N_samples,
            #             do_render=False,
            #         )
            #         all_z_val.append(z_vals)
            #         all_rgb.append(rgb)
            #         all_sigma.append(sigma)

            #         # if self.blending_mode == "samples_sigmas":
            #         #     sigma *= blending_weight_chunk[:, None]
            #         # else:
            #         #     raise NotImplementedError
            
            # if "samples" in self.blending_mode:
            #     all_sigma = torch.cat(all_sigma, dim=1)
            #     all_rgb = torch.cat(all_rgb, dim=1)
            #     all_z_val = torch.cat(all_z_val, dim=1)
            #     all_z_val, indices = torch.sort(all_z_val, dim=1)
            #     all_sigma = torch.index_select(all_sigma, 1, indices[0])
            #     all_rgb = torch.index_select(all_rgb, 1, indices[0])

            #     all_dists = torch.cat(
            #         (all_z_val[:, 1:] - all_z_val[:, :-1], torch.zeros_like(all_z_val[:, :1])),
            #         dim=-1,
            #     )
            #     rgb_map, depth_map, _ = render_from_samples(
            #         all_sigma,
            #         all_rgb,
            #         all_z_val,
            #         all_dists,
            #         white_bg,
            #         is_train,
            #         self.tensorfs[0].distance_scale
            #     )