# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2022 Anpei Chen

import os
import warnings

import cv2
import numpy as np
import torch
from tqdm.auto import tqdm

warnings.filterwarnings("ignore", category=DeprecationWarning)
import json
import sys
import time

from torch.utils.tensorboard import SummaryWriter

sys.path.append("localTensoRF")
from dataLoader.localrf_dataset import LocalRFDataset
from local_tensorfs import LocalTensorfs
from opt import config_parser
from renderer import render
from utils.utils import (get_fwd_bwd_cam2cams, smooth_poses_spline)
from utils.utils import (N_to_reso, TVLoss, draw_poses, get_pred_flow,
                         compute_depth_loss)


def save_transforms(poses_mtx, transform_path, local_tensorfs, train_dataset=None):
    if train_dataset is not None:
        fnames = train_dataset.all_image_paths
    else:
        fnames = [f"{i:06d}.jpg" for i in range(len(poses_mtx))]

    fl = local_tensorfs.focal(local_tensorfs.W).item()
    transforms = {
        "fl_x": fl,
        "fl_y": fl,
        "k1": 0.0,
        "k2": 0.0,
        "p1": 0.0,
        "p2": 0.0,
        "cx": local_tensorfs.W/2,
        "cy": local_tensorfs.H/2,
        "w": local_tensorfs.W,
        "h": local_tensorfs.H,
        "frames": [],
    }
    for pose_mtx, fname in zip(poses_mtx, fnames):
        pose = np.eye(4, dtype=np.float32)
        pose[:3, :] = pose_mtx
        frame_data = {
            "file_path": f"images/{fname}",
            "sharpness": 75.0,
            "transform_matrix": pose.tolist(),
        }
        transforms["frames"].append(frame_data)

    with open(transform_path, "w") as outfile:
        json.dump(transforms, outfile, indent=2)


@torch.no_grad()
def render_frames(
    args, poses_mtx, local_tensorfs, logfolder, test_dataset, train_dataset
):
    save_transforms(poses_mtx.cpu(), f"{logfolder}/transforms.json", local_tensorfs, train_dataset)
    t_w2rf = torch.stack(list(local_tensorfs.world2rf), dim=0).detach().cpu()
    RF_mtx_inv = torch.cat([torch.stack(len(t_w2rf) * [torch.eye(3)]), t_w2rf.clone()[..., None]], axis=-1)
    save_transforms(RF_mtx_inv.cpu(), f"{logfolder}/transforms_rf.json", local_tensorfs)
    
    W, H = train_dataset.img_wh

    if args.render_test:
        render(
            test_dataset,
            poses_mtx,
            local_tensorfs,
            args,
            W=W, H=H,
            savePath=f"{logfolder}/test",
            save_frames=True,
            save_video=False,
            add_frame_to_list=False,
            test=True,
            train_dataset=train_dataset,
            img_format="png",
            start=0
        )

    if args.render_path:
        c2ws = smooth_poses_spline(poses_mtx, median_prefilter=True)
        os.makedirs(f"{logfolder}/smooth_spline", exist_ok=True)
        save_transforms(c2ws.cpu(), f"{logfolder}/smooth_spline/transforms.json", local_tensorfs)
        render(
            test_dataset,
            c2ws,
            local_tensorfs,
            args,
            W=int(W / 1.5), H=int(H / 1.5),
            savePath=f"{logfolder}/smooth_spline",
            train_dataset=train_dataset,
            img_format="jpg",
            save_frames=True,
            save_video=not args.skip_saving_video, # skip_saving_video=True if set; save_video=False to save RAM --> not args.skip_saving_video
            add_frame_to_list=False,
            floater_thresh=0.5,
        )

    if args.render_from_file != "":
        with open(args.render_from_file, 'r') as f:
            transforms = json.load(f)
        c2ws = [transform["transform_matrix"] for transform in transforms["frames"]]
        c2ws = torch.tensor(c2ws).to(args.device)
        c2ws = c2ws[..., :3, :]

        if args.with_preprocessed_poses:
            raw2ours = torch.inverse(torch.from_numpy(train_dataset.first_pose)).to(c2ws)
            for c2w in c2ws:
               c2w[:3, :3] = raw2ours[:3, :3] @ c2w[:3, :3]
               c2w[:3, 3] = raw2ours[:3, :3] @ c2w[:3, 3]
               c2w[:3, 3] = raw2ours[:3, 3] + c2w[:3, 3]
            c2ws[:, :3, 3] *= train_dataset.pose_scale

        save_path = f"{logfolder}/{os.path.splitext(os.path.basename(args.render_from_file))[0]}"
        os.makedirs(save_path, exist_ok=True)
        render(
            test_dataset,
            c2ws,
            local_tensorfs,
            args,
            W=W, H=H,
            savePath=save_path,
            train_dataset=train_dataset,
            img_format="jpg",
            save_frames=True,
            save_video=not args.skip_saving_video, # skip_saving_video=True if set; save_video=False to save RAM --> not args.skip_saving_video
            add_frame_to_list=False,
            floater_thresh=0.5,
        )

@torch.no_grad()
def render_test(args):
    # init dataset
    train_dataset = LocalRFDataset(
        f"{args.datadir}",
        split="train",
        downsampling=args.downsampling,
        test_frame_every=args.test_frame_every,
        n_init_frames=args.n_init_frames,
        with_preprocessed_poses=args.with_preprocessed_poses,
        subsequence=args.subsequence,
        frame_step=args.frame_step,
    )
    test_dataset = LocalRFDataset(
        f"{args.datadir}",
        split="test",
        load_depth=args.loss_depth_weight_inital > 0,
        load_flow=args.loss_flow_weight_inital > 0,
        downsampling=args.downsampling,
        test_frame_every=args.test_frame_every,
        with_preprocessed_poses=args.with_preprocessed_poses,
        subsequence=args.subsequence,
        frame_step=args.frame_step,
    )

    if args.ckpt is None:
        logfolder = f"{args.logdir}"
        ckpt_path = f"{logfolder}/checkpoints.th"
    else:
        ckpt_path = args.ckpt

    if not os.path.isfile(ckpt_path):
        print("Backing up to intermediate checkpoints")
        ckpt_path = f"{logfolder}/checkpoints_tmp.th"
        if not os.path.isfile(ckpt_path):
            print("the ckpt path does not exists!!")
            return  

    with open(ckpt_path, "rb") as f:
        ckpt = torch.load(f, map_location=args.device)
    kwargs = ckpt["kwargs"]
    if args.with_preprocessed_poses:
        kwargs["camera_prior"] = {
            "rel_poses": torch.from_numpy(train_dataset.rel_poses).to(args.device),
            "transforms": train_dataset.transforms
        }
    else:
        kwargs["camera_prior"] = None
    kwargs["device"] = args.device
    local_tensorfs = LocalTensorfs(**kwargs)
    local_tensorfs.load(ckpt["state_dict"])
    local_tensorfs = local_tensorfs.to(args.device)

    logfolder = os.path.dirname(ckpt_path)
    render_frames(
        args,
        local_tensorfs.get_cam2world(),
        local_tensorfs,
        logfolder,
        test_dataset=test_dataset,
        train_dataset=train_dataset
    )


def reconstruction(args):
    # Apply speedup factors
    args.n_iters_per_frame = int(args.n_iters_per_frame / args.refinement_speedup_factor)
    args.n_iters_reg = int(args.n_iters_reg / args.refinement_speedup_factor)
    args.upsamp_list = [int(upsamp / args.refinement_speedup_factor) for upsamp in args.upsamp_list]
    args.update_AlphaMask_list = [int(update_AlphaMask / args.refinement_speedup_factor) 
                                  for update_AlphaMask in args.update_AlphaMask_list]
    
    args.add_frames_every = int(args.add_frames_every / args.prog_speedup_factor)
    args.lr_R_init = args.lr_R_init * args.prog_speedup_factor
    args.lr_t_init = args.lr_t_init * args.prog_speedup_factor
    args.loss_flow_weight_inital = args.loss_flow_weight_inital * args.prog_speedup_factor
    args.L1_weight = args.L1_weight * args.prog_speedup_factor
    args.TV_weight_density = args.TV_weight_density * args.prog_speedup_factor
    args.TV_weight_app = args.TV_weight_app * args.prog_speedup_factor
    
    # init dataset
    train_dataset = LocalRFDataset(
        f"{args.datadir}",
        split="train",
        downsampling=args.downsampling,
        test_frame_every=args.test_frame_every,
        load_depth=args.loss_depth_weight_inital > 0,
        load_flow=args.loss_flow_weight_inital > 0,
        with_preprocessed_poses=args.with_preprocessed_poses,
        n_init_frames=args.n_init_frames,
        subsequence=args.subsequence,
        frame_step=args.frame_step,
    )
    test_dataset = LocalRFDataset(
        f"{args.datadir}",
        split="test",
        load_depth=args.loss_depth_weight_inital > 0,
        load_flow=args.loss_flow_weight_inital > 0,
        downsampling=args.downsampling,
        test_frame_every=args.test_frame_every,
        with_preprocessed_poses=args.with_preprocessed_poses,
        subsequence=args.subsequence,
        frame_step=args.frame_step,
    )
    near_far = train_dataset.near_far

    # init resolution
    upsamp_list = args.upsamp_list
    n_lamb_sigma = args.n_lamb_sigma
    n_lamb_sh = args.n_lamb_sh

    logfolder = f"{args.logdir}"

    # init log file
    os.makedirs(logfolder, exist_ok=True)
    writer = SummaryWriter(log_dir=logfolder)

    # init parameters
    aabb = train_dataset.scene_bbox.to(args.device)
    reso_cur = N_to_reso(args.N_voxel_init, aabb)

    # TODO: Add midpoint loading
    # if args.ckpt is not None:
    #     ckpt = torch.load(args.ckpt, map_location=args.device)
    #     kwargs = ckpt["kwargs"]
    #     kwargs.update({"device": args.device})
    #     tensorf = eval(args.model_name)(**kwargs)
    #     tensorf.load(ckpt)
    # else:

    print("lr decay", args.lr_decay_target_ratio)

    # linear in logrithmic space
    N_voxel_list = (
        torch.round(
            torch.exp(
                torch.linspace(
                    np.log(args.N_voxel_init),
                    np.log(args.N_voxel_final),
                    len(upsamp_list) + 1,
                )
            )
        ).long()
    ).tolist()[1:]
    N_voxel_list = {
        usamp_idx: round(N_voxel**(1/3))**3 for usamp_idx, N_voxel in zip(upsamp_list, N_voxel_list)
    }

    if args.with_preprocessed_poses:
        camera_prior = {
            "rel_poses": torch.from_numpy(train_dataset.rel_poses).to(args.device),
            "transforms": train_dataset.transforms
        }
    else:
        camera_prior = None


    local_tensorfs = LocalTensorfs(
        camera_prior=camera_prior,
        fov=args.fov,
        n_init_frames=min(args.n_init_frames, train_dataset.num_images),
        n_overlap=args.n_overlap,
        WH=train_dataset.img_wh,
        n_iters_per_frame=args.n_iters_per_frame,
        n_iters_reg=args.n_iters_reg,
        lr_R_init=args.lr_R_init,
        lr_t_init=args.lr_t_init,
        lr_i_init=args.lr_i_init,
        lr_exposure_init=args.lr_exposure_init,
        rf_lr_init=args.lr_init,
        rf_lr_basis=args.lr_basis,
        lr_decay_target_ratio=args.lr_decay_target_ratio,
        N_voxel_list=N_voxel_list,
        update_AlphaMask_list=args.update_AlphaMask_list,
        lr_upsample_reset=args.lr_upsample_reset,
        device=args.device,
        alphaMask_thres=args.alpha_mask_thre,
        shadingMode=args.shadingMode,
        aabb=aabb,
        gridSize=reso_cur,
        density_n_comp=n_lamb_sigma,
        appearance_n_comp=n_lamb_sh,
        app_dim=args.data_dim_color,
        near_far=near_far,
        density_shift=args.density_shift,
        distance_scale=args.distance_scale,
        rayMarch_weight_thres=args.rm_weight_mask_thre,
        pos_pe=args.pos_pe,
        view_pe=args.view_pe,
        fea_pe=args.fea_pe,
        featureC=args.featureC,
        step_ratio=args.step_ratio,
        fea2denseAct=args.fea2denseAct,
    )
    local_tensorfs = local_tensorfs.to(args.device)

    torch.cuda.empty_cache()

    tvreg = TVLoss()
    W, H = train_dataset.img_wh

    training = True
    n_added_frames = 0
    last_add_iter = 0
    iteration = 0
    metrics = {}
    start_time = time.time()
    while training:
        optimize_poses = args.lr_R_init > 0 or args.lr_t_init > 0
        data_blob = train_dataset.sample(args.batch_size, local_tensorfs.is_refining, optimize_poses)
        view_ids = torch.from_numpy(data_blob["view_ids"]).to(args.device)
        rgb_train = torch.from_numpy(data_blob["rgbs"]).to(args.device)
        loss_weights = torch.from_numpy(data_blob["loss_weights"]).to(args.device)
        train_test_poses = data_blob["train_test_poses"]
        ray_idx = torch.from_numpy(data_blob["idx"]).to(args.device)
        reg_loss_weight = local_tensorfs.lr_factor ** (local_tensorfs.rf_iter[-1])

        rgb_map, depth_map, directions, ij = local_tensorfs(
            ray_idx,
            view_ids,
            W,
            H,
            is_train=True,
            test_id=train_test_poses,
        )

        # loss
        loss = 0.25 * ((torch.abs(rgb_map - rgb_train)) * loss_weights) / loss_weights.mean()
               
        loss = loss.mean()
        total_loss = loss
        writer.add_scalar("train/rgb_loss", loss, global_step=iteration)

        ## Regularization
        # Get rendered rays schedule
        if local_tensorfs.regularize and args.loss_flow_weight_inital > 0 or args.loss_depth_weight_inital > 0:
            depth_map = depth_map.view(view_ids.shape[0], -1)
            loss_weights = loss_weights.view(view_ids.shape[0], -1)
            depth_map = depth_map.view(view_ids.shape[0], -1)

            writer.add_scalar("train/reg_loss_weights", reg_loss_weight, global_step=iteration)

        # Optical flow
        if local_tensorfs.regularize and args.loss_flow_weight_inital > 0:
            if args.fov == 360:
                raise NotImplementedError
            starting_frame_id = max(train_dataset.active_frames_bounds[0] - 1, 0)
            cam2world = local_tensorfs.get_cam2world(starting_id=starting_frame_id)
            directions = directions.view(view_ids.shape[0], -1, 3)
            ij = ij.view(view_ids.shape[0], -1, 2)
            fwd_flow = torch.from_numpy(data_blob["fwd_flow"]).to(args.device).view(view_ids.shape[0], -1, 2)
            fwd_mask = torch.from_numpy(data_blob["fwd_mask"]).to(args.device).view(view_ids.shape[0], -1)
            fwd_mask[view_ids == len(cam2world) - 1] = 0
            bwd_flow = torch.from_numpy(data_blob["bwd_flow"]).to(args.device).view(view_ids.shape[0], -1, 2)
            bwd_mask = torch.from_numpy(data_blob["bwd_mask"]).to(args.device).view(view_ids.shape[0], -1)
            fwd_cam2cams, bwd_cam2cams = get_fwd_bwd_cam2cams(cam2world, view_ids - starting_frame_id)
                       
            pts = directions * depth_map[..., None]
            pred_fwd_flow = get_pred_flow(
                pts, ij, fwd_cam2cams, local_tensorfs.focal(W), local_tensorfs.center(W, H))
            pred_bwd_flow = get_pred_flow(
                pts, ij, bwd_cam2cams, local_tensorfs.focal(W), local_tensorfs.center(W, H))
            flow_loss_arr =  torch.sum(torch.abs(pred_bwd_flow - bwd_flow), dim=-1) * bwd_mask
            flow_loss_arr += torch.sum(torch.abs(pred_fwd_flow - fwd_flow), dim=-1) * fwd_mask
            flow_loss_arr[flow_loss_arr > torch.quantile(flow_loss_arr, 0.9, dim=1)[..., None]] = 0

            flow_loss = (flow_loss_arr).mean() * args.loss_flow_weight_inital * reg_loss_weight / ((W + H) / 2)
            total_loss = total_loss + flow_loss
            writer.add_scalar("train/flow_loss", flow_loss, global_step=iteration)

        # Monocular Depth 
        if local_tensorfs.regularize and args.loss_depth_weight_inital > 0:
            if args.fov == 360:
                raise NotImplementedError
            invdepths = torch.from_numpy(data_blob["invdepths"]).to(args.device)
            invdepths = invdepths.view(view_ids.shape[0], -1)
            _, _, depth_loss_arr = compute_depth_loss(1 / depth_map.clamp(1e-6), invdepths)
            depth_loss_arr[depth_loss_arr > torch.quantile(depth_loss_arr, 0.8, dim=1)[..., None]] = 0

            depth_loss = (depth_loss_arr).mean() * args.loss_depth_weight_inital * reg_loss_weight
            total_loss = total_loss + depth_loss 
            writer.add_scalar("train/depth_loss", depth_loss, global_step=iteration)

        if  local_tensorfs.regularize:
            loss_tv, l1_loss = local_tensorfs.get_reg_loss(tvreg, args.TV_weight_density, args.TV_weight_app, args.L1_weight)
            total_loss = total_loss + loss_tv + l1_loss
            writer.add_scalar("train/loss_tv", loss_tv, global_step=iteration)
            writer.add_scalar("train/l1_loss", l1_loss, global_step=iteration)

        # Optimizes
        if train_test_poses:
            can_add_rf = False
            if optimize_poses:
                local_tensorfs.optimizer_step_poses_only(total_loss)
        else:
            can_add_rf = local_tensorfs.optimizer_step(total_loss, optimize_poses)
            training |= train_dataset.active_frames_bounds[1] != train_dataset.num_images

        ## Progressive optimization
        if not local_tensorfs.is_refining:
            should_refine = (not train_dataset.has_left_frames() or (
                n_added_frames > args.n_overlap and (
                    local_tensorfs.get_dist_to_last_rf().cpu().item() > args.max_drift
                    or (train_dataset.active_frames_bounds[1] - train_dataset.active_frames_bounds[0]) >= args.n_max_frames
                )))
            if should_refine and (iteration - last_add_iter) >= args.add_frames_every:
                local_tensorfs.is_refining = True

            should_add_frame = train_dataset.has_left_frames()
            should_add_frame &= (iteration - last_add_iter + 1) % args.add_frames_every == 0

            should_add_frame &= not should_refine
            should_add_frame &= not local_tensorfs.is_refining
            # Add supervising frames
            if should_add_frame:
                local_tensorfs.append_frame()
                train_dataset.activate_frames()
                n_added_frames += 1
                last_add_iter = iteration

        # Add new RF
        if can_add_rf:
            if train_dataset.has_left_frames():
                local_tensorfs.append_rf(n_added_frames)
                n_added_frames = 0
                last_add_rf_iter = iteration

                # Remove supervising frames
                training_frames = (local_tensorfs.blending_weights[:, -1] > 0)
                train_dataset.deactivate_frames(
                    np.argmax(training_frames.cpu().numpy(), axis=0))
            else:
                training = False
        ## Log
        loss = loss.detach().item()

        writer.add_scalar(
            "train/density_app_plane_lr",
            local_tensorfs.rf_optimizer.param_groups[0]["lr"],
            global_step=iteration,
        )
        writer.add_scalar(
            "train/basis_mat_lr",
            local_tensorfs.rf_optimizer.param_groups[4]["lr"],
            global_step=iteration,
        )

        writer.add_scalar(
            "train/lr_r",
            local_tensorfs.r_optimizers[-1].param_groups[0]["lr"],
            global_step=iteration,
        )
        writer.add_scalar(
            "train/lr_t",
            local_tensorfs.t_optimizers[-1].param_groups[0]["lr"],
            global_step=iteration,
        )

        writer.add_scalar(
            "train/focal", local_tensorfs.focal(W), global_step=iteration
        )
        writer.add_scalar(
            "train/center0", local_tensorfs.center(W, H)[0].item(), global_step=iteration
        )
        writer.add_scalar(
            "train/center1", local_tensorfs.center(W, H)[1].item(), global_step=iteration
        )

        writer.add_scalar(
            "active_frames_bounds/0", train_dataset.active_frames_bounds[0], global_step=iteration
        )
        writer.add_scalar(
            "active_frames_bounds/1", train_dataset.active_frames_bounds[1], global_step=iteration
        )

        for index, blending_weights in enumerate(
            torch.permute(local_tensorfs.blending_weights, [1, 0])
        ):
            active_cam_indices = torch.nonzero(blending_weights)
            writer.add_scalar(
                f"tensorf_bounds/rf{index}_b0", active_cam_indices[0], global_step=iteration
            )
            writer.add_scalar(
                f"tensorf_bounds/rf{index}_b1", active_cam_indices[-1], global_step=iteration
            )

        # Print the current values of the losses.
        if iteration % args.progress_refresh_rate == 0:
            # All poses visualization
            poses_mtx = local_tensorfs.get_cam2world().detach().cpu()
            t_w2rf = torch.stack(list(local_tensorfs.world2rf), dim=0).detach().cpu()
            RF_mtx_inv = torch.cat([torch.stack(len(t_w2rf) * [torch.eye(3)]), -t_w2rf.clone()[..., None]], axis=-1)

            all_poses = torch.cat([poses_mtx,  RF_mtx_inv], dim=0)
            colours = ["C1"] * poses_mtx.shape[0] + ["C2"] * RF_mtx_inv.shape[0]
            img = draw_poses(all_poses, colours)
            writer.add_image("poses/all", (np.transpose(img, (2, 0, 1)) / 255.0).astype(np.float32), iteration)

            # Get runtime 
            ips = min(args.progress_refresh_rate, iteration + 1) / (time.time() - start_time)
            writer.add_scalar(f"train/iter_per_sec", ips, global_step=iteration)
            print(f"Iteration {iteration:06d}: {ips:.2f} it/s")
            start_time = time.time()

        if (iteration % args.vis_every == args.vis_every - 1):
            poses_mtx = local_tensorfs.get_cam2world().detach()
            rgb_maps_tb, depth_maps_tb, gt_rgbs_tb, fwd_flow_cmp_tb, bwd_flow_cmp_tb, depth_err_tb, loc_metrics = render(
                test_dataset,
                poses_mtx,
                local_tensorfs,
                args,
                W=W // 2, H=H // 2,
                savePath=logfolder,
                save_frames=True,
                img_format="jpg",
                test=True,
                train_dataset=train_dataset,
                start=train_dataset.active_frames_bounds[0],
                add_frame_to_list= not args.skip_TB_images, # skip_TB_images=True if set; add_frame_to_list=False to save RAM --> not args.skip_TB_images
            )

            if len(loc_metrics.values()):
                metrics.update(loc_metrics)
                mses = [metric["mse"] for metric in metrics.values()]
                writer.add_scalar(
                    f"test/PSNR", -10.0 * np.log(np.array(mses).mean()) / np.log(10.0), 
                    global_step=iteration
                )
                loc_mses = [metric["mse"] for metric in loc_metrics.values()]
                writer.add_scalar(
                    f"test/local_PSNR", -10.0 * np.log(np.array(loc_mses).mean()) / np.log(10.0), 
                    global_step=iteration
                )
                ssim = [metric["ssim"] for metric in metrics.values()]
                writer.add_scalar(
                    f"test/ssim", np.array(ssim).mean(), 
                    global_step=iteration
                )
                loc_ssim = [metric["ssim"] for metric in loc_metrics.values()]
                writer.add_scalar(
                    f"test/local_ssim", np.array(loc_ssim).mean(), 
                    global_step=iteration
                )

                if not args.skip_TB_images: # default False if not set, will be True if set. 
                    writer.add_images(
                        "test/rgb_maps",
                        torch.stack(rgb_maps_tb, 0),
                        global_step=iteration,
                        dataformats="NHWC",
                    )
                    writer.add_images(
                        "test/depth_map",
                        torch.stack(depth_maps_tb, 0),
                        global_step=iteration,
                        dataformats="NHWC",
                    )
                    writer.add_images(
                        "test/gt_maps",
                        torch.stack(gt_rgbs_tb, 0),
                        global_step=iteration,
                        dataformats="NHWC",
                    )
                    
                    if len(fwd_flow_cmp_tb) > 0:
                        writer.add_images(
                            "test/fwd_flow_cmp",
                            torch.stack(fwd_flow_cmp_tb, 0)[..., None],
                            global_step=iteration,
                            dataformats="NHWC",
                        )
                        
                        writer.add_images(
                            "test/bwd_flow_cmp",
                            torch.stack(bwd_flow_cmp_tb, 0)[..., None],
                            global_step=iteration,
                            dataformats="NHWC",
                        )
                    
                    if len(depth_err_tb) > 0:
                        writer.add_images(
                            "test/depth_cmp",
                            torch.stack(depth_err_tb, 0)[..., None],
                            global_step=iteration,
                            dataformats="NHWC",
                        )

                    # Clear all TensorBoard's lists
                    for list_tb in [rgb_maps_tb, depth_maps_tb, gt_rgbs_tb, fwd_flow_cmp_tb, bwd_flow_cmp_tb, depth_err_tb]:
                        list_tb.clear()

            with open(f"{logfolder}/checkpoints_tmp.th", "wb") as f:
                local_tensorfs.save(f)

        iteration += 1

    with open(f"{logfolder}/checkpoints.th", "wb") as f:
        local_tensorfs.save(f)

    poses_mtx = local_tensorfs.get_cam2world().detach()
    render_frames(args, poses_mtx, local_tensorfs, logfolder, test_dataset=test_dataset, train_dataset=train_dataset)


if __name__ == "__main__":

    torch.set_default_dtype(torch.float32)
    torch.manual_seed(20211202)
    np.random.seed(20211202)

    args = config_parser()
    print(args)

    if args.render_only:
        render_test(args)
    else:
        reconstruction(args)
