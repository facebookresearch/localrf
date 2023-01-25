import os
import warnings

import imageio
import cv2
import numpy as np
import torch
from mpl_toolkits.mplot3d import Axes3D  # noqa
from tqdm.auto import tqdm

warnings.filterwarnings("ignore", category=DeprecationWarning)
import json
import sys

from torch.utils.tensorboard import SummaryWriter

sys.path.append("localTensoRF")
from dataLoader import dataset_dict
from local_tensorfs import LocalTensorfs
from opt import config_parser
from renderer import render
from utils.camera import (average_poses, cam2world, get_lerp_poses, inverse_pose, get_cam2cams, get_fwd_bwd_cam2cams, pts2px,
                          get_novel_view_poses, compute_render_path, rotation_distance,
                          smooth_poses, upsample_poses_2x, smooth_poses_spline)
from utils.utils import (N_to_reso, TVLoss, cal_n_samples, draw_poses, get_pred_flow,
                         convert_sdf_samples_to_ply, compute_tv_norm,
                         compute_depth_loss)


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


def save_transforms(poses_mtx, transform_path, id_transform=False, index=None):
    transforms = {"frames": []}
    for (i, pose_mtx) in enumerate(poses_mtx):
        pose = np.eye(4, dtype=np.float32)
        if not id_transform:
            pose[:3, :] = pose_mtx
        if index is None:
            save_index = i
        else:
            save_index = index
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
    RF_mtx = torch.stack(list(local_tensorfs.module.world2rf), dim=0).detach().cpu()
    RF_mtx_inv = RF_mtx.clone()
    RF_mtx_inv[:, :3, :3] = torch.transpose(RF_mtx[:, :3, :3], 1, 2)
    RF_mtx_inv[:, :3, 3] = -torch.bmm(RF_mtx_inv[:, :3, :3], RF_mtx[:, :3, 3:])[..., 0]
    save_transforms(RF_mtx_inv.cpu(), f"{logfolder}/transforms_rf.json")
    save_transforms(poses_mtx.cpu(), f"{logfolder}/transforms_id.json", True)

    # rgb_maps_tb, depth_maps_tb, gt_rgbs_tb, alpha, T, weight, alpha_new, T_new, weight_new, loc_mses = render(
    #     test_dataset,
    #     poses_mtx,
    #     local_tensorfs,
    #     args,
    #     savePath=logfolder,
    #     save_frames=True,
    #     test=True,
    #     train_dataset=train_dataset,
    #     img_format="png",
    #     start=0
    # )

    if args.render_path:

        # # TODO f
        # c2ws = torch.from_numpy(compute_render_path(poses_mtx.cpu().numpy())).to(get_device(args)).float()
        # print(c2ws.shape)
        # frame_indices = torch.zeros(
        #     [c2ws.shape[0]], dtype=torch.long, device=get_device(args)
        # )
        # _, _, _, _, _, _, _, _, _, _ = render(
        #     test_dataset,
        #     c2ws,
        #     local_tensorfs,
        #     args,
        #     frame_indices=frame_indices,
        #     savePath=f"{logfolder}/360",
        #     save_video=True,
        # )
        c2ws = smooth_poses_spline(poses_mtx, st=0.3, sr=4)
        all_poses = torch.cat([poses_mtx.cpu(),  c2ws.cpu()], dim=0)
        colours = ["C1"] * poses_mtx.shape[0] + ["C2"] * c2ws.shape[0]
        cv2.imwrite(f"{logfolder}/smooth_spline.png", draw_poses(all_poses, colours)[..., ::-1])
        os.makedirs(f"{logfolder}/smooth_spline", exist_ok=True)
        save_transforms(c2ws.cpu(), f"{logfolder}/smooth_spline/transforms.json", index=0)
        _, _, _, _, _, _, _, _, _, _ = render(
            test_dataset,
            c2ws,
            local_tensorfs,
            args,
            savePath=f"{logfolder}/smooth_spline",
            img_format="jpg",
            save_frames=True,
            save_video=True,
        )

        c2ws = smooth_poses_spline(poses_mtx, st=2, sr=15)
        all_poses = torch.cat([poses_mtx.cpu(),  c2ws.cpu()], dim=0)
        colours = ["C1"] * poses_mtx.shape[0] + ["C2"] * c2ws.shape[0]
        cv2.imwrite(f"{logfolder}/smooth_spline1.png", draw_poses(all_poses, colours)[..., ::-1])
        os.makedirs(f"{logfolder}/smooth_spline1", exist_ok=True)
        save_transforms(c2ws.cpu(), f"{logfolder}/smooth_spline1/transforms.json", index=0)
        _, _, _, _, _, _, _, _, _, _ = render(
            test_dataset,
            c2ws,
            local_tensorfs,
            args,
            savePath=f"{logfolder}/smooth_spline1",
            img_format="jpg",
            save_frames=True,
            save_video=True,
        )

    if args.render_train:
        _, _, _, _, _, _, _, _, _, _ = render(
            test_dataset,
            poses_mtx,
            local_tensorfs,
            args,
            savePath=f"{logfolder}/imgs_train_all",
            save_video=True,
        )

        c2ws = smooth_poses(poses_mtx)
        all_poses = torch.cat([poses_mtx.cpu(),  c2ws.cpu()], dim=0)
        colours = ["C1"] * poses_mtx.shape[0] + ["C2"] * c2ws.shape[0]
        cv2.imwrite(f"{logfolder}/smooth.png", draw_poses(all_poses, colours)[..., ::-1])
        _, _, _, _, _, _, _, _, _, _ = render(
            test_dataset,
            c2ws,
            local_tensorfs,
            args,
            savePath=f"{logfolder}/smooth",
            save_video=True,
        )
        save_transforms(c2ws.cpu(), f"{logfolder}/smooth/transforms.json", index=0)

        poses_mtx[:, 0] = -poses_mtx[:, 0]
        # from barf
        avg_pose = torch.from_numpy(average_poses(poses_mtx.cpu().detach().numpy())).to(
            get_device(args)
        )
        avg_pose[0] = -avg_pose[0]

        # c2ws = get_lerp_poses(poses_mtx, N=35)
        # c2ws[:, 0] = -c2ws[:, 0]
        # frame_indices = torch.zeros(
        #     [c2ws.shape[0]], dtype=torch.long, device=get_device(args)
        # )
        # _, _, _, _, _, _, _, _, _ = render(
        #     test_dataset,
        #     c2ws,
        #     local_tensorfs,
        #     args,
        #     frame_indices=frame_indices,
        #     savePath=f"{logfolder}/lerp",
        #     N_vis=-1,
        #     save_video=True,
        # )
        # save_transforms(c2ws.cpu(), f"{logfolder}/lerp/transforms.json", index=0)

        c2ws = get_novel_view_poses(avg_pose, N=20, scale=0.2).to(get_device(args))
        frame_indices = torch.zeros(
            [c2ws.shape[0]], dtype=torch.long, device=get_device(args)
        )
        _, _, _, _, _, _, _, _, _, _ = render(
            test_dataset,
            c2ws,
            local_tensorfs,
            args,
            frame_indices=frame_indices,
            savePath=f"{logfolder}/path",
            save_video=True,
        )
        save_transforms(c2ws.cpu(), f"{logfolder}/path/transforms.json", index=0)

        c2ws = get_novel_view_poses(avg_pose, N=20, scale=1).to(get_device(args))
        frame_indices = torch.zeros(
            [c2ws.shape[0]], dtype=torch.long, device=get_device(args)
        )
        _, _, _, _, _, _, _, _, _, _ = render(
            test_dataset,
            c2ws,
            local_tensorfs,
            args,
            frame_indices=frame_indices,
            savePath=f"{logfolder}/path_wide",
            save_video=True,
        )
        save_transforms(c2ws.cpu(), f"{logfolder}/path_wide/transforms.json", index=0)

        poses_mtx[:, 0] = -poses_mtx[:, 0]

@torch.no_grad()
def render_test(args):
    # init dataset
    dataset = dataset_dict[args.dataset_name]
    train_dataset = dataset(
        f"{args.basedir}/sequenced/{args.datadir}",
        split="train",
        resize_h=args.resize_h,
        ray_type=args.ray_type,
        n_init_frames=args.n_init_frames,
        subsequence=args.subsequence,
        with_GT_poses=args.with_GT_poses,
    )
    test_dataset = dataset(
        f"{args.basedir}/sequenced/{args.datadir}",
        split="test",
        resize_h=args.resize_h,
        ray_type=args.ray_type,
        subsequence=args.subsequence,
        with_GT_poses=args.with_GT_poses,
    )

    if args.ckpt is None:
        logfolder = f"{args.basedir}/logs/{args.expname}/{args.datadir}"
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
    kwargs["lr_poses_init"] = 0
    kwargs["new_rf_init"] = 0
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


@torch.no_grad()
def render_multiple_rf(args):
    local_tensorfs_list = []
    all_poses_mtx = []
    for expname in args.expname_list:
        logfolder = f"{args.basedir}/logs/{args.expname}/{args.datadir}"
        ckpt_path = f"{logfolder}/checkpoints.th"

        if not os.path.isfile(ckpt_path):
            print(f'the ckpt path "{ckpt_path}" does not exists!!')
            return

        with open(ckpt_path, "rb") as f:
            ckpt = torch.load(f, map_location=get_device(args))
        kwargs = ckpt["kwargs"]
        kwargs["lr_poses_init"] = 0
        kwargs["camera_prior"] = None
        kwargs["device"] = get_device(args)
        local_tensorfs = LocalTensorfs(**kwargs)
        local_tensorfs.load_state_dict(ckpt["state_dict"])
        local_tensorfs = torch.nn.DataParallel(local_tensorfs, device_ids=args.multiGPU)
        local_tensorfs_list.append(local_tensorfs)
        all_poses_mtx.append(local_tensorfs.module.get_cam2world())

    overlap = all_poses_mtx[0].shape[0] // 2
    world2rfs = [torch.FloatTensor(np.eye(4)[:3, :4]).to(get_device(args))]
    rf_scales = [1]
    all_poses_mtx_canonical = [all_poses_mtx[0]]
    local_tensorfs = local_tensorfs_list[0]
    for rf_index in range(1, len(local_tensorfs_list)):
        last_poses = all_poses_mtx_canonical[rf_index - 1][-overlap:].clone()
        first_poses = all_poses_mtx[rf_index][:overlap].clone()
        last_scale = torch.norm(last_poses[-1, :, 3] - last_poses[0, :, 3])
        first_scale = torch.norm(first_poses[-1, :, 3] - first_poses[0, :, 3])
        rf_scale = last_scale / first_scale
        rf_scales.append(rf_scale.cpu().item())

        last_poses[:, 0] = -last_poses[:, 0]
        last_avg_pose = torch.from_numpy(
            average_poses(last_poses.cpu().detach().numpy())
        ).to(get_device(args))
        last_avg_pose[0] = -last_avg_pose[0]

        first_poses[:, 0] = -first_poses[:, 0]
        first_avg_pose = torch.from_numpy(
            average_poses(first_poses.cpu().detach().numpy())
        ).to(get_device(args))
        first_avg_pose[0] = -first_avg_pose[0]

        rf_pose = torch.zeros([3, 4], device=get_device(args))
        rf_pose[:3, :3] = torch.matmul(
            last_avg_pose[:3, :3], torch.linalg.inv(first_avg_pose[:3, :3])
        )
        rf_pose[:3, 3] = last_avg_pose[:3, 3]
        rf_pose[:3, 3] -= (
            torch.matmul(last_avg_pose[:3, :3], first_avg_pose[:3, 3]) * rf_scale
        )
        poses_mtx_canonical = all_poses_mtx[rf_index][overlap:].clone()
        poses_mtx_canonical[:, :3, 3] = torch.matmul(
            rf_pose[:3, :3], poses_mtx_canonical[:, :3, 3].T
        ).T
        poses_mtx_canonical[:, :3, 3] *= rf_scale
        poses_mtx_canonical[:, :3, 3] += rf_pose[:3, 3]

        poses_mtx_canonical[:, :3, :3] = torch.matmul(
            rf_pose[:3, :3], poses_mtx_canonical[:, :3, :3].T
        ).T

        all_poses_mtx_canonical.append(poses_mtx_canonical)

        rf_poses_inv = torch.zeros_like(rf_pose)
        rf_poses_inv[:3, :3] = rf_pose[:3, :3].T
        rf_poses_inv[:3, 3] = -rf_pose[:3, :3].T @ rf_pose[:3, 3]
        rf_poses_inv /= rf_scale
        world2rfs.append(rf_poses_inv)
        local_tensorfs.module.tensorfs.append(
            local_tensorfs_list[rf_index].module.tensorfs[0]
        )

    all_poses_mtx_canonical = torch.cat(all_poses_mtx_canonical)
    world2rfs = torch.stack(world2rfs)

    outfolder = f"{args.basedir}/logs/{args.expname}/{args.datadir}"
    os.makedirs(outfolder, exist_ok=True)

    c2ws_canonical = smooth_poses(all_poses_mtx_canonical)
    # c2ws_canonical = upsample_poses_2x(c2ws_canonical)
    h_n_p_rf = overlap
    n_frames = len(c2ws_canonical)

    all_blending_weights = []
    for rf_index in range(len(local_tensorfs_list)):
        blending_weights = torch.zeros(n_frames).to(get_device(args))
        start = rf_index * (h_n_p_rf + 1)
        stop = rf_index * (h_n_p_rf + 1) + h_n_p_rf
        blending_weights[start:stop] = (0.5 + torch.arange(h_n_p_rf)) / h_n_p_rf

        start = (rf_index + 1) * (h_n_p_rf + 1)
        stop = (rf_index + 1) * (h_n_p_rf + 1) + h_n_p_rf
        blending_weights[start:stop] = 1 - (0.5 + torch.arange(h_n_p_rf)) / h_n_p_rf

        blending_weights[(rf_index) * (h_n_p_rf + 1) + h_n_p_rf] = 1
        if rf_index == 0:
            blending_weights[:h_n_p_rf] = 1
        if rf_index == len(local_tensorfs_list) - 1:
            blending_weights[-h_n_p_rf:] = 1

        all_blending_weights.append(blending_weights)

    all_blending_weights = torch.stack(all_blending_weights, dim=-1)

    # init dataset
    dataset = dataset_dict[args.dataset_name]
    train_dataset = dataset(
        f"{args.basedir}/sequenced/{args.datadir}",
        split="train",
        resize_h=args.resize_h,
        ray_type=args.ray_type,
        n_init_frames=args.n_init_frames,
        subsequence=args.subsequence,
    )
    test_dataset = dataset(
        f"{args.basedir}/sequenced/{args.datadir}",
        split="test",
        resize_h=args.resize_h,
        ray_type=args.ray_type,
        subsequence=args.subsequence,
    )

    frame_indices = torch.zeros(
        [c2ws_canonical.shape[0]], dtype=torch.long, device=get_device(args)
    )
    render(
        test_dataset,
        c2ws_canonical,
        local_tensorfs,
        args,
        frame_indices=frame_indices,
        savePath=outfolder,
        N_vis=-1,
        save_video=True,
        world2rfs=world2rfs,
        all_blending_weights=all_blending_weights,
    )


@torch.no_grad()
def evaluate_camera_alignment(pose_aligned, pose_GT):
    # measure errors in rotation and translation
    # pose_aligned: [N, 3, 4]
    # pose_GT:      [N, 3, 4]
    R_aligned, t_aligned = pose_aligned.split([3, 1], dim=-1)  # [N, 3, 3], [N, 3, 1]
    R_GT, t_GT = pose_GT.split([3, 1], dim=-1)  # [N, 3, 3], [N, 3, 1]
    R_error = rotation_distance(R_aligned, R_GT)
    t_error = (t_aligned - t_GT)[..., 0].norm(dim=-1)
    return R_error, t_error



class SimpleCustomBatch:
    def __init__(self, data_blob):
        self.data_blob = torch.stack({"rgbs": data_blob})

    def pin_memory(self):
        for key in self.data_blob:
            self.data_blob[key] = self.data_blob[key].pin_memory()
        return self


def collate_wrapper(batch):
    return SimpleCustomBatch(batch)


def reconstruction(args):
    # init dataset
    dataset = dataset_dict[args.dataset_name]
    train_dataset = dataset(
        f"{args.basedir}/sequenced/{args.datadir}",
        split="train",
        resize_h=args.resize_h,
        load_depth=args.loss_depth_weight_inital > 0,
        load_flow=args.loss_flow_weight_inital > 0,
        with_GT_poses=args.with_GT_poses,
        ray_type=args.ray_type,
        n_init_frames=args.n_init_frames,
        subsequence=args.subsequence,
    )
    test_dataset = dataset(
        f"{args.basedir}/sequenced/{args.datadir}",
        split="test",
        resize_h=args.resize_h,
        with_GT_poses=args.with_GT_poses,
        ray_type=args.ray_type,
        subsequence=args.subsequence,
    )
    white_bg = train_dataset.white_bg
    near_far = train_dataset.near_far

    # init resolution
    upsamp_list = args.upsamp_list
    n_lamb_sigma = args.n_lamb_sigma
    n_lamb_sh = args.n_lamb_sh

    logfolder = f"{args.basedir}/logs/{args.expname}/{args.datadir}"

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

    if args.lr_decay_iters > 0:
        rf_lr_factor = args.lr_decay_target_ratio ** (1 / args.lr_decay_iters)
    else:
        args.lr_decay_iters = args.n_iters
        rf_lr_factor = args.lr_decay_target_ratio ** (1 / args.n_iters)

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
        lr_poses_init=args.lr_poses_init,
        camera_prior=camera_prior,
        fov=args.fov,
        pose_representation=args.pose_representation,
        ray_type=args.ray_type,
        n_init_frames=min(args.n_init_frames, train_dataset.num_images),
        n_overlap=args.n_overlap,
        blending_mode=args.blending_mode,
        WH=train_dataset.img_wh,
        new_rf_init=args.new_rf_init,
        n_iters=args.n_iters,
        rf_lr_init=args.lr_init,
        rf_lr_basis=args.lr_basis,
        lr_decay_target_ratio=args.lr_decay_target_ratio,
        rf_lr_factor=rf_lr_factor,
        N_voxel_list=N_voxel_list,
        lr_upsample_reset=args.lr_upsample_reset,
        optimize_focal=args.optimize_focal,
        device=get_device(args),
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

    # TODO implement
    # if args.optimize_focal:
    #     lr_focal = 3e-3
    #     lr_focal_end = 1e-5  # 5:X, 10:X
    #     optimizer_focal = torch.optim.Adam([local_tensorfs.module.focal], lr=lr_focal)
    #     gamma = (lr_focal_end / lr_focal) ** (1.0 / (args.n_iters))
    #     scheduler_focal = torch.optim.lr_scheduler.ExponentialLR(
    #         optimizer_focal, gamma=gamma
    #     )

    Ortho_reg_weight = args.Ortho_weight
    print("initial Ortho_reg_weight", Ortho_reg_weight)

    L1_reg_weight = args.L1_weight_inital
    print("initial L1_reg_weight", L1_reg_weight)
    TV_weight_density, TV_weight_app = args.TV_weight_density, args.TV_weight_app
    # tvreg = torch.nn.DataParallel(TVLoss(), device_ids=args.multiGPU)
    tvreg = TVLoss()
    # (local_tensorfs, device_ids=args.multiGPU)
    print(f"initial TV_weight density: {TV_weight_density} appearance: {TV_weight_app}")
    W, H = train_dataset.img_wh

    # poses_render = []
    # alphas_graph = []
    pbar = tqdm()
    training = True
    n_added_frames = 0
    last_add_iter = 0
    last_add_rf_iter = 0
    iteration = 0
    metrics = {}
    # def print_trace(p):
    #     print(p.key_averages().table(sort_by="self_cpu_time_total", row_limit=20))
    # with torch.profiler.profile(
    #     activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA,],
    #     schedule=torch.profiler.schedule(wait=1, warmup=1, active=2),
    #     on_trace_ready=print_trace,
    # ) as p:
    while training:
        # print(p.key_averages().table(sort_by="self_cpu_time_total", row_limit=20))
        data_blob = train_dataset.sample(args.batch_size)
        view_ids = torch.from_numpy(data_blob["view_ids"]).to(get_device(args))
        rgb_train = torch.from_numpy(data_blob["rgbs"]).to(get_device(args))
        # rgb_train = rgb_train.view(view_ids.shape[0], -1, 3)
        laplacian = torch.from_numpy(data_blob["laplacian"]).to(get_device(args))
        # laplacian = laplacian.view(view_ids.shape[0], -1, 1)
        motion_mask = torch.from_numpy(data_blob["motion_mask"]).to(get_device(args))
        # motion_mask = motion_mask.view(view_ids.shape[0], -1, 1)
        train_test_poses = data_blob["train_test_poses"]
        n_test_frames = 0
        bg_mask = (motion_mask == 0).float()
        ray_idx = torch.from_numpy(data_blob["idx"]).to(get_device(args))

        rgb_map, depth_map, depth_map_gamma, directions, ij, blending_weights, loss_dist = local_tensorfs(
            ray_idx,
            view_ids,
            W,
            H,
            white_bg=white_bg,
            is_train=True,
            opasity_gamma=args.opasity_gamma,
            distortion_loss_weight=args.distortion_loss_weight,
        )

        # rgb_map = rgb_map.view(view_ids.shape[0], -1, 3)

        # loss
        # loss = torch.mean(((rgb_map - rgb_train) ** 2))


        lap_weights = laplacian * bg_mask
        # lap_weights = laplacian - laplacian.mean()
        # lap_weights = lap_weights / torch.sqrt(lap_weights.var())
        # lap_weights = torch.sigmoid(lap_weights)

        # if n_test_frames:
        #     rgb_map_test_poses = rgb_map[-n_test_frames:].clone()
        #     rgb_train_test_poses = rgb_train[-n_test_frames:].clone()
        #     lap_weights_test_poses = lap_weights[-n_test_frames:].clone()

        #     loss_test_poses = 0.5 * torch.mean((torch.abs(rgb_map_test_poses - rgb_train_test_poses)) * lap_weights_test_poses) / lap_weights_test_poses.mean()
        #     total_loss_test_poses = loss_test_poses
        #     writer.add_scalar("train_test_poses/photo_loss", loss_test_poses, global_step=iteration)

        #     rgb_map = rgb_map[:-n_test_frames].clone()
        #     rgb_train = rgb_train[:-n_test_frames].clone()
        #     lap_weights = lap_weights[:-n_test_frames].clone()

        loss = 2 * (((rgb_map - rgb_train) ** 2) * lap_weights) / lap_weights.mean()
        # loss[loss > torch.quantile(loss, 0.95)] = 0
        loss = loss.mean()
        # loss = 0.5 * torch.mean((torch.abs(rgb_map - rgb_train)) * lap_weights) / lap_weights.mean()
        total_loss = loss
        writer.add_scalar("train/photo_loss", loss, global_step=iteration)
        loss_dist = loss_dist.mean()

        ## Regularization
        # # novel view patches
        # if args.patch_size > 1 and args.patch_batch_size > 0 and args.patch_d_tv_weight > 0:
        #     # get median spacing of currently supervising frames to bound the novel view renders
        #     cam2world = torch.cat(list(local_tensorfs.module.cam2world), dim=0).detach().clone()
        #     training_cam2world = cam2world[train_dataset.active_frames_bounds[0]:]
        #     frame2frame = training_cam2world[1:] - training_cam2world[:-1]
        #     med_shift = torch.median(frame2frame.abs(), dim=0)[0]
        #     patch_tv_loss = 0
        #     for _ in range(args.patch_batch_size):
        #         view_id = np.random.randint(train_dataset.active_frames_bounds[0], train_dataset.active_frames_bounds[1])
        #         i_start = np.random.randint(0, W - args.patch_size)
        #         j_start = np.random.randint(0, H - args.patch_size)
        #         i, j = torch.meshgrid(
        #             torch.arange(i_start, i_start + args.patch_size, device=get_device(args)), 
        #             torch.arange(j_start, j_start + args.patch_size, device=get_device(args)), 
        #             indexing='xy')

        #         ray_idx = i + j * W + view_id * W * H
                
        #         pose_shift = torch.rand_like(med_shift) * 2 - 1
        #         pose_shift = pose_shift * med_shift
        #         cam2world_jitter = cam2world[view_id] + pose_shift
        #         cam2world_jitter = local_tensorfs.module.get_mtx([cam2world_jitter[None]]).detach()

        #         patch_rgb_map, patch_depth_map, patch_depth_map_gamma, _, _, _, patch_loss_dist = local_tensorfs(
        #             ray_idx.view(-1),
        #             W,
        #             H,
        #             white_bg=white_bg,
        #             is_train=True,
        #             cam2world=cam2world_jitter.repeat(len(local_tensorfs.device_ids), 1, 1),
        #             distortion_loss_weight=args.distortion_loss_weight,
        #         )

        #         patch_depth_map = patch_depth_map.view(1, args.patch_size, args.patch_size)
        #         loss_dist += patch_loss_dist.mean() * args.patch_size**2 / args.batch_size
        #         patch_tv_loss += compute_tv_norm(1/(patch_depth_map+1e-3)).mean() * args.patch_d_tv_weight / args.patch_batch_size

        #     total_loss = total_loss + patch_tv_loss
        #     writer.add_scalar("train/patch_tv_loss", patch_tv_loss, global_step=iteration)

        # Get rendered rays schedule
        if args.loss_flow_weight_inital > 0 or args.loss_depth_weight_inital > 0:
            depth_map = depth_map.view(view_ids.shape[0], -1)
            bg_mask = bg_mask.view(view_ids.shape[0], -1)
            depth_map_gamma = depth_map_gamma.view(view_ids.shape[0], -1)

            rf_ids = torch.argmax(blending_weights, axis=-1)
            iterations = torch.Tensor(local_tensorfs.module.rf_iter).to(get_device(args))
            loss_weights = torch.clamp(rf_lr_factor ** (iterations[rf_ids]), min=0).view(view_ids.shape[0], -1)
            writer.add_scalar("train/loss_weights", loss_weights.mean(), global_step=iteration)
            # if n_test_frames:
            #     loss_weights_test_poses = loss_weights[-n_test_frames:].clone()
            #     loss_weights = loss_weights[:-n_test_frames].clone()

        # Optical flow
        if args.loss_flow_weight_inital > 0:
            cam2world = local_tensorfs.module.get_cam2world()
            directions = directions.view(view_ids.shape[0], -1, 3)
            ij = ij.view(view_ids.shape[0], -1, 2)
            fwd_flow = torch.from_numpy(data_blob["fwd_flow"]).to(get_device(args)).view(view_ids.shape[0], -1, 2)
            fwd_mask = torch.from_numpy(data_blob["fwd_mask"]).to(get_device(args)).view(view_ids.shape[0], -1)
            fwd_mask[view_ids == len(cam2world) - 1] = 0
            bwd_flow = torch.from_numpy(data_blob["bwd_flow"]).to(get_device(args)).view(view_ids.shape[0], -1, 2)
            bwd_mask = torch.from_numpy(data_blob["bwd_mask"]).to(get_device(args)).view(view_ids.shape[0], -1)
            fwd_cam2cams, bwd_cam2cams = get_fwd_bwd_cam2cams(cam2world, view_ids)
            
            # pts = directions * depth_map[..., None]
            # pred_fwd_flow = get_pred_flow(pts, ij, fwd_cam2cams, local_tensorfs.module.focal, center)
            # pred_bwd_flow = get_pred_flow(pts, ij, bwd_cam2cams, local_tensorfs.module.focal, center)
            # flow_loss =  torch.sum(torch.abs(pred_fwd_flow - fwd_flow), dim=-1) * fwd_mask# / torch.clamp(torch.norm(fwd_flow, dim=-1), min=1)
            # flow_loss += torch.sum(torch.abs(pred_bwd_flow - bwd_flow), dim=-1) * bwd_mask# / torch.clamp(torch.norm(bwd_flow, dim=-1), min=1)
            
            pts = directions * depth_map_gamma[..., None]
            center = local_tensorfs.module.center(W, H)
            pred_fwd_flow = get_pred_flow(pts, ij, fwd_cam2cams, local_tensorfs.module.focal(W, H), center)
            pred_bwd_flow = get_pred_flow(pts, ij, bwd_cam2cams, local_tensorfs.module.focal(W, H), center)
            flow_loss_arr =  torch.sum(torch.abs(pred_bwd_flow - bwd_flow), dim=-1) * bwd_mask
            flow_loss_arr += torch.sum(torch.abs(pred_fwd_flow - fwd_flow), dim=-1) * fwd_mask
            flow_loss_arr[flow_loss_arr > torch.quantile(flow_loss_arr, 0.95)] = 0
            # flow_loss_arr[flow_loss_arr > 100] = 0

            # if n_test_frames:
            #     flow_loss_arr_test_poses = flow_loss_arr[-n_test_frames:].clone()
            #     flow_loss_test_poses = (flow_loss_arr_test_poses * loss_weights_test_poses).mean() * args.loss_flow_weight_inital / ((W + H) / 2)
            #     total_loss_test_poses = total_loss_test_poses + flow_loss_test_poses
            #     writer.add_scalar("train_test_poses/flow_loss", flow_loss_test_poses, global_step=iteration)

            #     flow_loss_arr = flow_loss_arr[:-n_test_frames].clone()

            flow_loss_arr = flow_loss_arr * loss_weights
            flow_loss = (flow_loss_arr).mean() * args.loss_flow_weight_inital / ((W + H) / 2)
            total_loss = total_loss + flow_loss
            writer.add_scalar("train/flow_loss", flow_loss, global_step=iteration)



        # Monocular Depth 
        if args.loss_depth_weight_inital > 0:
            # depth_loss =  compute_depth_loss(1 / (torch.clamp(depth_map, 1e-6)), invdepths)
            invdepths = torch.from_numpy(data_blob["invdepths"]).to(get_device(args))
            invdepths = invdepths.view(view_ids.shape[0], -1)
            depth_loss_arr = compute_depth_loss(1 / (torch.clamp(depth_map_gamma, 1e-6)), invdepths)
            depth_loss_arr[depth_loss_arr > torch.quantile(depth_loss_arr, 0.95)] = 0

            # if n_test_frames:
            #     depth_loss_arr_test_poses = depth_loss_arr[-n_test_frames:].clone()
            #     depth_loss_test_poses = (depth_loss_arr_test_poses  * loss_weights_test_poses).mean() * args.loss_depth_weight_inital
            #     total_loss_test_poses = total_loss_test_poses + depth_loss_test_poses
            #     writer.add_scalar("train_test_poses/depth_loss", depth_loss_test_poses, global_step=iteration)

            #     depth_loss_arr = depth_loss_arr[:-n_test_frames].clone()

            depth_loss_arr = depth_loss_arr * loss_weights
            depth_loss = (depth_loss_arr).mean() * args.loss_depth_weight_inital
            total_loss = total_loss + depth_loss 
            writer.add_scalar("train/depth_loss", depth_loss, global_step=iteration)



        total_loss = total_loss + loss_dist
        writer.add_scalar("train/loss_dist", loss_dist, global_step=iteration)

        loss_tv, l1_loss = local_tensorfs.module.get_reg_loss(tvreg, args.TV_weight_density, args.TV_weight_app, args.L1_weight_inital)
        total_loss = total_loss + loss_tv + l1_loss
        writer.add_scalar("train/loss_tv", loss_tv, global_step=iteration)
        writer.add_scalar("train/l1_loss", l1_loss, global_step=iteration)

        # if args.optimize_poses and args.pose_consistancy_loss:
        #     local_motion = poses_mtx[1:, :, 3] - poses_mtx[0:-1, :, 3]
        #     # local_motion_grad = local_motion[1:] - local_motion[0:-1]

        #     weight = 10 * max(400 - iteration, 0)  # 1e2
        #     pose_consistancy_loss = torch.mean(local_motion**2) * weight
        #     # pose_consistancy_loss = torch.mean(torch.abs(local_motion)) * weight
        #     total_loss = total_loss + pose_consistancy_loss
        #     writer.add_scalar(
        #         "train/pose_consistancy_loss",
        #         pose_consistancy_loss,
        #         global_step=iteration,
        #     )
        # if Ortho_reg_weight > 0:
        #     loss_reg = tensorf.module.vector_comp_diffs()
        #     total_loss = total_loss + Ortho_reg_weight * loss_reg
        #     writer.add_scalar(
        #         "train/reg", loss_reg.detach().item(), global_step=iteration
        #     )

        # Optimizes
        if train_test_poses:
            should_add_rf = False
            if args.optimize_poses:
                local_tensorfs.module.optimizer_step_poses_only(total_loss)
        else:
            training, should_add_rf = local_tensorfs.module.optimizer_step(total_loss, args.optimize_poses)
            training |= train_dataset.active_frames_bounds[1] != train_dataset.num_images

        if iteration % 1000:
            torch.cuda.empty_cache()

        ## Progressive optimization
        

        # should_add_rf = local_tensorfs.module.get_dist_to_last_rf().cpu().item() > args.max_drift
        # should_add_rf |= (n_added_frames + 1) % args.n_max_frames == 0
        # should_add_rf |= local_tensorfs.module.rf_iter[-1] >= args.upsamp_list[0]
        # should_add_rf &= train_dataset.active_frames_bounds[1] + args.n_overlap < train_dataset.num_images
        # should_add_rf &= n_added_frames > args.n_overlap
        # n_training_rf = np.sum(np.array(local_tensorfs.module.rf_iter) < args.n_iters)
        # block_add_rf = should_add_rf and n_training_rf >= args.n_max_training_rfs # Don't add more RF if several are already being optimized
        # block_add_rf |= should_add_rf and (local_tensorfs.module.rf_iter[-1] < args.n_iters - 1) #(args.n_iters + last_up) // 2)

        if n_added_frames > args.n_overlap and (
                local_tensorfs.module.get_dist_to_last_rf().cpu().item() > args.max_drift or
                n_added_frames >= args.n_max_frames):
            local_tensorfs.module.is_refining = True

        should_add_frame = train_dataset.has_left_frames()
        should_add_frame &= (iteration - last_add_iter + 1) % args.add_frames_every == 0
        if last_add_rf_iter != 0:
            should_add_frame &= (iteration - last_add_rf_iter) > (args.add_frames_every * 5)

        should_add_frame &= not local_tensorfs.module.is_refining
        # Add supervising frames
        if should_add_frame:
            local_tensorfs.module.append_frame(reset_rf_opt=args.add_frame_reset_rf_opt)
            train_dataset.activate_frames()
            n_added_frames += 1
            last_add_iter = iteration
            torch.cuda.empty_cache()

        # Add new RF
        if should_add_rf:
            local_tensorfs.module.is_refining = False
            local_tensorfs.module.append_rf(n_added_frames)
            n_added_frames = 0
            last_add_iter = iteration
            last_add_rf_iter = iteration

            # Remove supervising frames
            training_frames = local_tensorfs.module.get_training_frames()
            train_dataset.deactivate_frames(np.argmax(training_frames, axis=0))

        ## Log
        loss = loss.detach().item()
        # PSNRs.append(-10.0 * np.log(loss) / np.log(10.0))
        # writer.add_scalar("train/PSNR", PSNRs[-1], global_step=iteration)
        # writer.add_scalar("train/mse", loss, global_step=iteration)


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
            "train/lr_pose",
            local_tensorfs.module.pose_optimizers[-1].param_groups[0]["lr"],
            global_step=iteration,
        )

        if args.optimize_focal:
            writer.add_scalar(
                "train/lr_intrinsics",
                local_tensorfs.module.intrinsic_optimizer.param_groups[0]["lr"],
                global_step=iteration,
            )

        writer.add_scalar(
            "fov/x", local_tensorfs.module.fov2[0].item() * 180 / np.pi, global_step=iteration
        )
        writer.add_scalar(
            "fov/y", local_tensorfs.module.fov2[1].item() * 180 / np.pi, global_step=iteration
        )
        writer.add_scalar(
            "intrinsics/focal_0", local_tensorfs.module.focal(W, H)[0].item(), global_step=iteration
        )
        writer.add_scalar(
            "intrinsics/focal_1", local_tensorfs.module.focal(W, H)[1].item(), global_step=iteration
        )
        writer.add_scalar(
            "intrinsics/center_0", local_tensorfs.module.center(W, H)[0].item(), global_step=iteration
        )
        writer.add_scalar(
            "intrinsics/center_1", local_tensorfs.module.center(W, H)[1].item(), global_step=iteration
        )
        writer.add_scalar(
            "intrinsics/focal_offset0", local_tensorfs.module.focal_offset[0].item(), global_step=iteration
        )
        writer.add_scalar(
            "intrinsics/focal_offset1", local_tensorfs.module.focal_offset[1].item(), global_step=iteration
        )
        writer.add_scalar(
            "intrinsics/center_offset0", local_tensorfs.module.center_offset[0].item(), global_step=iteration
        )
        writer.add_scalar(
            "intrinsics/center_offset1", local_tensorfs.module.center_offset[1].item(), global_step=iteration
        )
        writer.add_scalar(
            "intrinsics/dist_coefs0", local_tensorfs.module.dist_coefs[0].item(), global_step=iteration
        )
        writer.add_scalar(
            "intrinsics/dist_coefs1", local_tensorfs.module.dist_coefs[1].item(), global_step=iteration
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
                # + f" train_psnr = {float(np.mean(PSNRs)):.2f}"
                # + f" test_psnr = {float(np.mean(PSNRs_test)):.2f}"
                + f" mse = {loss:.6f}"
            )
            # PSNRs = []

            # All poses visualization
            poses_mtx = local_tensorfs.module.get_cam2world().detach().cpu()
            RF_mtx = torch.stack(list(local_tensorfs.module.world2rf), dim=0).detach().cpu()
            RF_mtx_inv = RF_mtx.clone()
            RF_mtx_inv[:, :3, :3] = torch.transpose(RF_mtx[:, :3, :3], 1, 2)
            RF_mtx_inv[:, :3, 3] = -torch.bmm(RF_mtx_inv[:, :3, :3], RF_mtx[:, :3, 3:])[..., 0]

            all_poses = torch.cat([poses_mtx,  RF_mtx_inv], dim=0)
            colours = ["C1"] * poses_mtx.shape[0] + ["C2"] * RF_mtx_inv.shape[0]
            img = draw_poses(all_poses, colours)
            # current_img = (test_dataset.all_rgbs[train_dataset.active_frames_bounds[1] - 1]*255).astype(np.uint8)
            # current_img = cv2.resize(current_img, (int(current_img.shape[1] * img.shape[0] / current_img.shape[0]), img.shape[0]))
            # img = np.hstack([img, current_img])
            cv2.imwrite(f"{logfolder}/poses.png", img[:, :, ::-1])
            writer.add_image("poses/all", (np.transpose(img, (2, 0, 1)) / 255.0).astype(np.float32), iteration)
            # poses_render.append(img)

            # Per RF poses visualization
            cam2rfs = local_tensorfs.module.get_cam2rf(
                local_tensorfs.module.world2rf, 
                local_tensorfs.module.get_cam2world(), 
                range(len(local_tensorfs.module.world2rf)),
            )
            for key in cam2rfs:
                valid = torch.nonzero(local_tensorfs.module.blending_weights[:, key])[:, 0].detach().cpu()
                img = draw_poses(cam2rfs[key].detach().cpu()[valid, :, :], ["C1"] * valid.shape[0])
                cv2.imwrite(f"{logfolder}/poses_rf{key}.png", img[:, :, ::-1])
                writer.add_image(f"poses/rf{key}", (np.transpose(img, (2, 0, 1)) / 255.0).astype(np.float32), iteration)

        if iteration % args.vis_every == 0:
            poses_mtx = local_tensorfs.module.get_cam2world().detach()
            vis_logfolder = f"{logfolder}/prog_vis/{iteration}"
            os.makedirs(vis_logfolder, exist_ok=True)
            save_transforms(poses_mtx.cpu(), f"{vis_logfolder}/transforms.json")
            RF_mtx = torch.stack(list(local_tensorfs.module.world2rf), dim=0).detach().cpu()
            RF_mtx_inv = RF_mtx.clone()
            RF_mtx_inv[:, :3, :3] = torch.transpose(RF_mtx[:, :3, :3], 1, 2)
            RF_mtx_inv[:, :3, 3] = -torch.bmm(RF_mtx_inv[:, :3, :3], RF_mtx[:, :3, 3:])[..., 0]
            save_transforms(RF_mtx_inv.cpu(), f"{vis_logfolder}/transforms_rf.json")
            
            rgb_maps_tb, depth_maps_tb, gt_rgbs_tb, alpha, T, weight, alpha_new, T_new, weight_new, loc_metrics = render(
                test_dataset,
                poses_mtx,
                local_tensorfs,
                args,
                savePath=vis_logfolder,
                save_frames=True,
                test=True,
                train_dataset=train_dataset,
                img_format="jpg",
                start=train_dataset.active_frames_bounds[0],
                save_raw_depth=True,
            )

        if (
            iteration % args.vis_train_every == args.vis_train_every - 1
        ):
            poses_mtx = local_tensorfs.module.get_cam2world().detach()
            rgb_maps_tb, depth_maps_tb, gt_rgbs_tb, alpha, T, weight, alpha_new, T_new, weight_new, loc_metrics = render(
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
            # import matplotlib.pyplot as plt
            
            # fig = plt.figure()
            # plt.plot(np.linspace(0, 1, len(alpha)), alpha.detach().cpu().numpy(), label="alpha")
            # plt.plot(np.linspace(0, 1, len(alpha)), T[:-1].detach().cpu().numpy(), label="T")
            # plt.plot(np.linspace(0, 1, len(alpha)), weight.detach().cpu().numpy(), label="weight")
            # plt.plot(np.linspace(0, 1, len(alpha)), alpha_new.detach().cpu().numpy(), label="alpha_new")
            # plt.plot(np.linspace(0, 1, len(alpha)), T_new[:-1].detach().cpu().numpy(), label="T_new")
            # plt.plot(np.linspace(0, 1, len(alpha)), weight_new.detach().cpu().numpy(), label="weight_new")
            # plt.legend(loc='upper center')
            # fig.canvas.draw()
            # img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
            # img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            # cv2.imwrite(f"{logfolder}/alphas.png", img[..., ::-1])
            # alphas_graph.append(img)

            metrics.update(loc_metrics)
            mses = [metric["mse"] for metric in metrics.values()]
            writer.add_scalar(
                f"test/PSNR", -10.0 * np.log(np.array(mses).mean()) / np.log(10.0), 
                global_step=iteration
            )
            loc_mses = [metric["mse"] for metric in loc_metrics.values()]
            writer.add_scalar(
                f"test/loc_PSNR", -10.0 * np.log(np.array(loc_mses).mean()) / np.log(10.0), 
                global_step=iteration
            )
            ssim = [metric["ssim"] for metric in metrics.values()]
            writer.add_scalar(
                f"test/ssim", np.array(ssim).mean(), 
                global_step=iteration
            )
            loc_ssim = [metric["ssim"] for metric in loc_metrics.values()]
            writer.add_scalar(
                f"test/loc_ssim", np.array(loc_ssim).mean(), 
                global_step=iteration
            )
            lpips = [metric["lpips"] for metric in metrics.values()]
            writer.add_scalar(
                f"test/lpips", np.array(lpips).mean(), 
                global_step=iteration
            )
            loc_lpips = [metric["lpips"] for metric in loc_metrics.values()]
            writer.add_scalar(
                f"test/loc_lpips", np.array(loc_lpips).mean(), 
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
        # p.step()

    with open(f"{logfolder}/checkpoints.th", "wb") as f:
        local_tensorfs.module.save(f)
    pbar.close()

    # with open(f"{logfolder}/poses_video.mp4", "wb") as f:
    #     imageio.mimwrite(f, np.stack(poses_render), fps=30, quality=6, format="mp4", output_params=["-f", "mp4"])
    # with open(f"{logfolder}/alphas_video.mp4", "wb") as f:
    #     imageio.mimwrite(f, np.stack(alphas_graph), fps=30, quality=6, format="mp4", output_params=["-f", "mp4"])

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

    if args.render_multiple_rf:
        render_multiple_rf(args)
    else:
        if args.render_only and (args.render_test or args.render_path):
            render_test(args)
        else:
            reconstruction(args)
