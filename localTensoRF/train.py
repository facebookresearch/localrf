import os
import warnings

import cv2
import numpy as np
import torch
from tqdm.auto import tqdm

warnings.filterwarnings("ignore", category=DeprecationWarning)
import json
import sys

from torch.utils.tensorboard import SummaryWriter

sys.path.append("localTensoRF")
from dataLoader.localrf_dataset import LocalRFDataset
from local_tensorfs import LocalTensorfs
from opt import config_parser
from renderer import render
from utils.utils import (get_fwd_bwd_cam2cams, smooth_poses_spline)
from utils.utils import (N_to_reso, TVLoss, draw_poses, get_pred_flow,
                         convert_sdf_samples_to_ply, compute_depth_loss)


def get_device(args):
    return torch.device(f"cuda:{args.multiGPU[0]}" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def export_mesh(args):

    ckpt = None
    with open(args.ckpt, "rb") as f:
        ckpt = torch.load(f, map_location=get_device(args))
    kwargs = ckpt["kwargs"]
    kwargs.update({"device": get_device(args)})
    tensorf = eval(args.model_name)(**kwargs)
    tensorf.load(ckpt)

    alpha, _ = tensorf.getDenseAlpha()
    convert_sdf_samples_to_ply(
        alpha.cpu(), f"{args.ckpt[:-3]}.ply", bbox=tensorf.aabb.cpu(), level=0.005
    )


def save_transforms(poses_mtx, transform_path):
    transforms = {"frames": []}
    for (save_index, pose_mtx) in enumerate(poses_mtx):
        pose = np.eye(4, dtype=np.float32)
        pose[:3, :] = pose_mtx
        frame_data = {
            "file_path": f"images/{save_index:04d}.jpg",
            "sharpness": 75.0,
            "transform_matrix": pose.tolist(),
        }
        transforms["frames"].append(frame_data)

    with open(transform_path, "w") as outfile:
        json.dump(transforms, outfile, indent=2)


@torch.no_grad()
def render_frames(
    args, poses_mtx, local_tensorfs, logfolder, test_dataset=None, train_dataset=None
):
    save_transforms(poses_mtx.cpu(), f"{logfolder}/transforms.json")
    t_w2rf = torch.stack(list(local_tensorfs.module.world2rf), dim=0).detach().cpu()
    RF_mtx_inv = torch.cat([torch.stack(len(t_w2rf) * [torch.eye(3)]), t_w2rf.clone()[..., None]], axis=-1)
    save_transforms(RF_mtx_inv.cpu(), f"{logfolder}/transforms_rf.json")

    if args.render_test:
        _, _, _, _ = render(
            test_dataset,
            poses_mtx,
            local_tensorfs,
            args,
            savePath=logfolder,
            save_frames=True,
            test=True,
            train_dataset=train_dataset,
            img_format="png",
            start=0
        )

    # ############# TMP
    # n_views = 40
    # scale = 22
    # c2ws = poses_mtx[25][None].repeat([n_views, 1, 1])
    # baseline = torch.norm(poses_mtx[25, :3, 3] - poses_mtx[24, :3, 3])
    # t = (c2ws[0, :3, :3]) @ torch.tensor([1, 0, 0]).to(c2ws)
    # for i in range(n_views):
    #     c2ws[i, :3, 3] += t * baseline * scale * (n_views - i - 20) / n_views
    # all_poses = torch.cat([poses_mtx.cpu(),  c2ws.cpu()], dim=0)
    # colours = ["C1"] * poses_mtx.shape[0] + ["C2"] * c2ws.shape[0]
    # cv2.imwrite(f"{logfolder}/extra_t.png", draw_poses(all_poses, colours)[..., ::-1])
    # os.makedirs(f"{logfolder}/extra_t", exist_ok=True)
    # _, _, _, _ = render(
    #         test_dataset,
    #         c2ws,
    #         local_tensorfs,
    #         args,
    #         savePath=f"{logfolder}/extra_t",
    #         img_format="jpg",
    #         save_frames=True,
    #         save_video=False,
    #     )

    # # ###
    # n_views = 21
    # c2ws = poses_mtx[55][None].repeat([n_views, 1, 1])
    # R = torch.Tensor(
    #     [   [0.9975510,  0.0000000,  0.0699428,],
    #         [0.0000000,  1.0000000,  0.0000000,],
    #         [-0.0699428,  0.0000000,  0.9975510,],]).to(c2ws)
    # R = (c2ws[0, :3, :3]) @ R @ torch.linalg.inv(c2ws[0, :3, :3])
    # for i in range(n_views):
    #     c2ws[i, :3, :3] = torch.linalg.matrix_power(R, n_views // 2 - i) @ c2ws[i, :3, :3]
    # all_poses = torch.cat([poses_mtx.cpu(),  c2ws.cpu()], dim=0)
    # colours = ["C1"] * poses_mtx.shape[0] + ["C2"] * c2ws.shape[0]
    # cv2.imwrite(f"{logfolder}/extra_r.png", draw_poses(all_poses, colours)[..., ::-1])
    # os.makedirs(f"{logfolder}/extra_r", exist_ok=True)
    # _, _, _, _ = render(
    #         test_dataset,
    #         c2ws,
    #         local_tensorfs,
    #         args,
    #         savePath=f"{logfolder}/extra_r",
    #         img_format="jpg",
    #         save_frames=True,
    #         save_video=False,
    #     )
    # # ############

    if args.render_path:
        c2ws = smooth_poses_spline(poses_mtx, st=0.3, sr=4)
        all_poses = torch.cat([poses_mtx.cpu(),  c2ws.cpu()], dim=0)
        colours = ["C1"] * poses_mtx.shape[0] + ["C2"] * c2ws.shape[0]
        cv2.imwrite(f"{logfolder}/smooth_spline_1.png", draw_poses(all_poses, colours)[..., ::-1])
        os.makedirs(f"{logfolder}/smooth_spline_1", exist_ok=True)
        save_transforms(c2ws.cpu(), f"{logfolder}/smooth_spline_1/transforms.json")
        _, _, _, _ = render(
            test_dataset,
            c2ws,
            local_tensorfs,
            args,
            savePath=f"{logfolder}/smooth_spline_1",
            train_dataset=train_dataset,
            img_format="jpg",
            save_frames=True,
            save_video=True,
        )

        c2ws = smooth_poses_spline(poses_mtx, st=2, sr=15)
        all_poses = torch.cat([poses_mtx.cpu(),  c2ws.cpu()], dim=0)
        colours = ["C1"] * poses_mtx.shape[0] + ["C2"] * c2ws.shape[0]
        cv2.imwrite(f"{logfolder}/smooth_spline_2.png", draw_poses(all_poses, colours)[..., ::-1])
        os.makedirs(f"{logfolder}/smooth_spline_2", exist_ok=True)
        save_transforms(c2ws.cpu(), f"{logfolder}/smooth_spline_2/transforms.json")
        _, _, _, _ = render(
            test_dataset,
            c2ws,
            local_tensorfs,
            args,
            savePath=f"{logfolder}/smooth_spline_2",
            train_dataset=train_dataset,
            img_format="jpg",
            save_frames=True,
            save_video=True,
        )

@torch.no_grad()
def render_test(args):
    # init dataset
    train_dataset = LocalRFDataset(
        f"{args.datadir}",
        split="train",
        downsampling=args.downsampling,
        n_init_frames=args.n_init_frames,
        subsequence=args.subsequence,
        with_GT_poses=args.with_GT_poses,
    )
    test_dataset = LocalRFDataset(
        f"{args.datadir}",
        split="test",
        downsampling=args.downsampling,
        subsequence=args.subsequence,
        with_GT_poses=args.with_GT_poses,
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
        ckpt = torch.load(f, map_location=get_device(args))
    kwargs = ckpt["kwargs"]
    if args.with_GT_poses:
        kwargs["camera_prior"] = {
            "rel_poses": torch.from_numpy(train_dataset.rel_poses).to(get_device(args)),
            "transforms": train_dataset.transforms
        }
    else:
        kwargs["camera_prior"] = None
    kwargs["device"] = get_device(args)
    local_tensorfs = LocalTensorfs(**kwargs)
    local_tensorfs.load(ckpt["state_dict"])
    local_tensorfs = torch.nn.DataParallel(local_tensorfs, device_ids=args.multiGPU)

    logfolder = os.path.dirname(ckpt_path)
    render_frames(
        args,
        local_tensorfs.module.get_cam2world(),
        local_tensorfs,
        logfolder,
        test_dataset=test_dataset,
        train_dataset=train_dataset
    )


def reconstruction(args):
    # init dataset
    train_dataset = LocalRFDataset(
        f"{args.datadir}",
        split="train",
        downsampling=args.downsampling,
        load_depth=args.loss_depth_weight_inital > 0,
        load_flow=args.loss_flow_weight_inital > 0,
        with_GT_poses=args.with_GT_poses,
        n_init_frames=args.n_init_frames,
        subsequence=args.subsequence,
    )
    test_dataset = LocalRFDataset(
        f"{args.datadir}",
        split="test",
        downsampling=args.downsampling,
        with_GT_poses=args.with_GT_poses,
        subsequence=args.subsequence,
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
    aabb = train_dataset.scene_bbox.to(get_device(args))
    reso_cur = N_to_reso(args.N_voxel_init, aabb)

    # TODO: Add midpoint loading
    # if args.ckpt is not None:
    #     ckpt = torch.load(args.ckpt, map_location=get_device(args))
    #     kwargs = ckpt["kwargs"]
    #     kwargs.update({"device": get_device(args)})
    #     tensorf = eval(args.model_name)(**kwargs)
    #     tensorf.load(ckpt)
    # else:

    # if args.lr_decay_iters > 0:
    #     lr_factor = args.lr_decay_target_ratio ** (1 / args.lr_decay_iters)
    # else:
    #     args.lr_decay_iters = args.n_iters
    #     lr_factor = args.lr_decay_target_ratio ** (1 / args.n_iters_per_frame)

    print("lr decay", args.lr_decay_target_ratio, args.lr_decay_iters)

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

    if args.with_GT_poses:
        camera_prior = {
            "rel_poses": torch.from_numpy(train_dataset.rel_poses).to(get_device(args)),
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
        device=get_device(args),
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
        pos_pe=args.pos_pe,
        view_pe=args.view_pe,
        fea_pe=args.fea_pe,
        featureC=args.featureC,
        step_ratio=args.step_ratio,
        fea2denseAct=args.fea2denseAct,
    )
    local_tensorfs = torch.nn.DataParallel(local_tensorfs, device_ids=args.multiGPU)

    torch.cuda.empty_cache()
    PSNRs, PSNRs_test = [], [0]

    L1_reg_weight = args.L1_weight_inital
    print("initial L1_reg_weight", L1_reg_weight)
    TV_weight_density, TV_weight_app = args.TV_weight_density, args.TV_weight_app
    tvreg = TVLoss()
    print(f"initial TV_weight density: {TV_weight_density} appearance: {TV_weight_app}")
    W, H = train_dataset.img_wh

    pbar = tqdm()
    training = True
    n_added_frames = 0
    last_add_iter = 0
    last_add_rf_iter = 0
    iteration = 0
    metrics = {}
    while training:
        data_blob = train_dataset.sample(args.batch_size, local_tensorfs.module.is_refining)
        view_ids = torch.from_numpy(data_blob["view_ids"]).to(get_device(args))
        rgb_train = torch.from_numpy(data_blob["rgbs"]).to(get_device(args))
        laplacian = torch.from_numpy(data_blob["laplacian"]).to(get_device(args))
        motion_mask = torch.from_numpy(data_blob["motion_mask"]).to(get_device(args))
        train_test_poses = data_blob["train_test_poses"]
        bg_mask = (motion_mask == 0).float()
        ray_idx = torch.from_numpy(data_blob["idx"]).to(get_device(args))

        rgb_map, depth_map, directions, ij, blending_weights = local_tensorfs(
            ray_idx,
            view_ids,
            W,
            H,
            is_train=True,
            test_id=train_test_poses,
        )

        # loss
        rgb_loss_weights = laplacian * bg_mask

        loss = 2 * (((rgb_map - rgb_train) ** 2) * rgb_loss_weights) / rgb_loss_weights.mean()
        loss[loss > torch.quantile(loss, 0.95)] = 0
        loss = loss.mean()
        total_loss = loss
        writer.add_scalar("train/rgb_loss", loss, global_step=iteration)

        ## Regularization
        # Get rendered rays schedule
        if local_tensorfs.module.regularize and args.loss_flow_weight_inital > 0 or args.loss_depth_weight_inital > 0:
            depth_map = depth_map.view(view_ids.shape[0], -1)
            bg_mask = bg_mask.view(view_ids.shape[0], -1)
            depth_map = depth_map.view(view_ids.shape[0], -1)

            # rf_ids = torch.argmax(blending_weights, axis=-1)
            # iterations = torch.Tensor(local_tensorfs.module.rf_iter).to(get_device(args))
            # loss_weights = torch.clamp(local_tensorfs.module.lr_factor ** (iterations[rf_ids]), min=0).view(view_ids.shape[0], -1)
            loss_weight = local_tensorfs.module.lr_factor ** (local_tensorfs.module.rf_iter[-1])
            writer.add_scalar("train/reg_loss_weights", loss_weight, global_step=iteration)

        # Optical flow
        if local_tensorfs.module.regularize and args.loss_flow_weight_inital > 0:
            cam2world = local_tensorfs.module.get_cam2world()
            directions = directions.view(view_ids.shape[0], -1, 3)
            ij = ij.view(view_ids.shape[0], -1, 2)
            fwd_flow = torch.from_numpy(data_blob["fwd_flow"]).to(get_device(args)).view(view_ids.shape[0], -1, 2)
            fwd_mask = torch.from_numpy(data_blob["fwd_mask"]).to(get_device(args)).view(view_ids.shape[0], -1)
            fwd_mask[view_ids == len(cam2world) - 1] = 0
            bwd_flow = torch.from_numpy(data_blob["bwd_flow"]).to(get_device(args)).view(view_ids.shape[0], -1, 2)
            bwd_mask = torch.from_numpy(data_blob["bwd_mask"]).to(get_device(args)).view(view_ids.shape[0], -1)
            fwd_cam2cams, bwd_cam2cams = get_fwd_bwd_cam2cams(cam2world, view_ids)
                       
            pts = directions * depth_map[..., None]
            center = [W / 2, H / 2]
            pred_fwd_flow = get_pred_flow(pts, ij, fwd_cam2cams, local_tensorfs.module.focal(W), center)
            pred_bwd_flow = get_pred_flow(pts, ij, bwd_cam2cams, local_tensorfs.module.focal(W), center)
            flow_loss_arr =  torch.sum(torch.abs(pred_bwd_flow - bwd_flow), dim=-1) * bwd_mask
            flow_loss_arr += torch.sum(torch.abs(pred_fwd_flow - fwd_flow), dim=-1) * fwd_mask
            flow_loss_arr[flow_loss_arr > torch.quantile(flow_loss_arr, 0.95)] = 0

            flow_loss = (flow_loss_arr).mean() * args.loss_flow_weight_inital * loss_weight / ((W + H) / 2)
            total_loss = total_loss + flow_loss
            writer.add_scalar("train/flow_loss", flow_loss, global_step=iteration)

        # Monocular Depth 
        if local_tensorfs.module.regularize and args.loss_depth_weight_inital > 0:
            invdepths = torch.from_numpy(data_blob["invdepths"]).to(get_device(args))
            invdepths = invdepths.view(view_ids.shape[0], -1)
            depth_loss_arr = compute_depth_loss(1 / (torch.clamp(depth_map, 1e-6)), invdepths)
            depth_loss_arr[depth_loss_arr > torch.quantile(depth_loss_arr, 0.95)] = 0

            depth_loss = (depth_loss_arr).mean() * args.loss_depth_weight_inital * loss_weight
            total_loss = total_loss + depth_loss 
            writer.add_scalar("train/depth_loss", depth_loss, global_step=iteration)

        if  local_tensorfs.module.regularize:
            loss_tv, l1_loss = local_tensorfs.module.get_reg_loss(tvreg, args.TV_weight_density, args.TV_weight_app, args.L1_weight_inital)
            total_loss = total_loss + loss_tv + l1_loss
            writer.add_scalar("train/loss_tv", loss_tv, global_step=iteration)
            writer.add_scalar("train/l1_loss", l1_loss, global_step=iteration)

        # Optimizes
        optimize_poses = args.lr_R_init > 0 or args.lr_t_init > 0
        if train_test_poses:
            can_add_rf = False
            if optimize_poses:
                local_tensorfs.module.optimizer_step_poses_only(total_loss)
        else:
            can_add_rf = local_tensorfs.module.optimizer_step(total_loss, optimize_poses)
            training |= train_dataset.active_frames_bounds[1] != train_dataset.num_images

        if iteration % 5000:
            torch.cuda.empty_cache()

        ## Progressive optimization
        should_refine = (not train_dataset.has_left_frames() or (
            n_added_frames > args.n_overlap and (
                local_tensorfs.module.get_dist_to_last_rf().cpu().item() > args.max_drift
                # or n_added_frames >= args.n_max_frames
            )))
        if should_refine and (iteration - last_add_iter) >= args.add_frames_every:
            local_tensorfs.module.is_refining = True

        should_add_frame = train_dataset.has_left_frames()
        should_add_frame &= (iteration - last_add_iter + 1) % args.add_frames_every == 0
        if last_add_rf_iter != 0:
            should_add_frame &= (iteration - last_add_rf_iter) > (args.add_frames_every * args.n_overlap)

        should_add_frame &= not should_refine
        should_add_frame &= not local_tensorfs.module.is_refining
        # Add supervising frames
        if should_add_frame:
            local_tensorfs.module.append_frame()
            train_dataset.activate_frames()
            n_added_frames += 1
            last_add_iter = iteration
            torch.cuda.empty_cache()

        # Add new RF
        if can_add_rf:
            if train_dataset.has_left_frames():
                local_tensorfs.module.append_rf(n_added_frames)
                n_added_frames = 0
                last_add_iter = iteration
                last_add_rf_iter = iteration

                # Remove supervising frames
                training_frames = (local_tensorfs.module.blending_weights[:, -1] > 0)
                train_dataset.deactivate_frames(
                    np.argmax(training_frames.cpu().numpy(), axis=0))
            else:
                training = False
        ## Log
        loss = loss.detach().item()

        writer.add_scalar(
            "train/density_app_plane_lr",
            local_tensorfs.module.rf_optimizers[-1].param_groups[0]["lr"],
            global_step=iteration,
        )
        writer.add_scalar(
            "train/basis_mat_lr",
            local_tensorfs.module.rf_optimizers[-1].param_groups[4]["lr"],
            global_step=iteration,
        )

        writer.add_scalar(
            "train/lr_r",
            local_tensorfs.module.r_optimizers[-1].param_groups[0]["lr"],
            global_step=iteration,
        )
        writer.add_scalar(
            "train/lr_t",
            local_tensorfs.module.t_optimizers[-1].param_groups[0]["lr"],
            global_step=iteration,
        )

        writer.add_scalar(
            "train/lr_intrinsics",
            local_tensorfs.module.intrinsic_optimizer.param_groups[0]["lr"],
            global_step=iteration,
        )

        writer.add_scalar(
            "focal", local_tensorfs.module.focal(W), global_step=iteration
        )

        writer.add_scalar(
            "active_frames_bounds/0", train_dataset.active_frames_bounds[0], global_step=iteration
        )
        writer.add_scalar(
            "active_frames_bounds/1", train_dataset.active_frames_bounds[1], global_step=iteration
        )

        for index, blending_weights in enumerate(
            torch.permute(local_tensorfs.module.blending_weights, [1, 0])
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
            pbar.set_description(
                f"Iteration {iteration:05d}:"
                + f" loss = {loss:.6f}"
            )

            # All poses visualization
            poses_mtx = local_tensorfs.module.get_cam2world().detach().cpu()
            t_w2rf = torch.stack(list(local_tensorfs.module.world2rf), dim=0).detach().cpu()
            RF_mtx_inv = torch.cat([torch.stack(len(t_w2rf) * [torch.eye(3)]), -t_w2rf.clone()[..., None]], axis=-1)

            all_poses = torch.cat([poses_mtx,  RF_mtx_inv], dim=0)
            colours = ["C1"] * poses_mtx.shape[0] + ["C2"] * RF_mtx_inv.shape[0]
            img = draw_poses(all_poses, colours)
            writer.add_image("poses/all", (np.transpose(img, (2, 0, 1)) / 255.0).astype(np.float32), iteration)

        # if iteration % args.vis_every == 0:
        #     poses_mtx = local_tensorfs.module.get_cam2world().detach()
        #     vis_logfolder = f"{logfolder}/prog_vis/{iteration}"
        #     os.makedirs(vis_logfolder, exist_ok=True)
        #     save_transforms(poses_mtx.cpu(), f"{vis_logfolder}/transforms.json")
        #     t_w2rf = torch.stack(list(local_tensorfs.module.world2rf), dim=0).detach().cpu()
        #     RF_mtx_inv = torch.cat([torch.stack(len(t_w2rf) * [torch.eye(3)]), t_w2rf.clone()[..., None]], axis=-1)
        #     save_transforms(RF_mtx_inv.cpu(), f"{vis_logfolder}/transforms_rf.json")
            
        #     rgb_maps_tb, depth_maps_tb, gt_rgbs_tb, loc_metrics = render(
        #         test_dataset,
        #         poses_mtx,
        #         local_tensorfs,
        #         args,
        #         savePath=vis_logfolder,
        #         save_frames=True,
        #         test=True,
        #         train_dataset=train_dataset,
        #         img_format="jpg",
        #         start=train_dataset.active_frames_bounds[0],
        #         save_raw_depth=True,
        #     )

        if (
            iteration % args.vis_every == args.vis_every - 1
        ):
            poses_mtx = local_tensorfs.module.get_cam2world().detach()
            rgb_maps_tb, depth_maps_tb, gt_rgbs_tb, loc_metrics = render(
                test_dataset,
                poses_mtx,
                local_tensorfs,
                args,
                savePath=logfolder,
                save_frames=True,
                test=True,
                train_dataset=train_dataset,
                img_format="jpg",
                start=train_dataset.active_frames_bounds[0]
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

            with open(f"{logfolder}/checkpoints_tmp.th", "wb") as f:
                local_tensorfs.module.save(f)
            torch.cuda.empty_cache()

        pbar.update(1)
        iteration += 1

    with open(f"{logfolder}/checkpoints.th", "wb") as f:
        local_tensorfs.module.save(f)
    pbar.close()

    poses_mtx = local_tensorfs.module.get_cam2world().detach()
    render_frames(args, poses_mtx, local_tensorfs, logfolder, test_dataset=test_dataset, train_dataset=train_dataset)


if __name__ == "__main__":

    torch.set_default_dtype(torch.float32)
    torch.manual_seed(20211202)
    np.random.seed(20211202)

    args = config_parser()
    print(args)

    if args.export_mesh:
        export_mesh(args)

    if args.render_only and (args.render_test or args.render_path):
        render_test(args)
    else:
        reconstruction(args)
