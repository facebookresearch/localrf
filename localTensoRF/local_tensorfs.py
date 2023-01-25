import copy
from http.client import UnimplementedFileMode
import math

import numpy as np
import torch

# from models.tensorBase import render_from_samples
from models.tensoRF import TensorVMSplit

from utils.camera import lie, sixD_to_mtx
from utils.ray_utils import get_ray_directions_lean, get_rays_lean, ndc_rays_blender2
from utils.utils import cal_n_samples, N_to_reso
from torch_efficient_distloss.eff_distloss import eff_distloss_native

# from .torch_efficient_distloss import flatten_eff_distloss


def ids2pixel_view(W, H, ids):
    """
    Regress pixel coordinates from ray indices
    """
    col = ids % W
    row = (ids // W) % H
    view_ids = ids // (W * H)
    return col, row, view_ids

def ids2pixel(W, H, ids):
    """
    Regress pixel coordinates from ray indices
    """
    col = ids % W
    row = (ids // W) % H
    return col, row

class LocalTensorfs(torch.nn.Module):
    """
    Self calibrating local tensorfs.
    """

    def __init__(
        self,
        lr_poses_init,
        camera_prior,
        fov,
        pose_representation,
        ray_type,
        n_init_frames,
        n_overlap,
        blending_mode,
        WH,
        new_rf_init,
        n_iters,
        rf_lr_init,
        rf_lr_basis,
        lr_decay_target_ratio,
        rf_lr_factor,
        N_voxel_list,
        lr_upsample_reset,
        optimize_focal,
        device,
        shadingMode,
        **tensorf_args,
    ):

        super(LocalTensorfs, self).__init__()

        self.camera_prior = camera_prior
        self.fov = fov
        self.pose_representation = pose_representation
        self.ray_type = ray_type
        self.n_init_frames = n_init_frames
        self.n_overlap = n_overlap
        self.blending_mode = blending_mode
        self.W, self.H = WH
        self.new_rf_init = new_rf_init
        self.n_iters = n_iters
        self.rf_lr_init, self.rf_lr_basis, self.lr_decay_target_ratio = rf_lr_init, rf_lr_basis, lr_decay_target_ratio
        self.rf_lr_factor = rf_lr_factor
        self.N_voxel_list = N_voxel_list
        self.lr_upsample_reset = lr_upsample_reset
        self.optimize_focal = optimize_focal
        self.device = device
        self.shadingMode = shadingMode
        self.tensorf_args = tensorf_args
        self.lr_pose = lr_poses_init
        self.is_refining = False

        # Setup pose and camera parameters
        self.cam2world, self.app_embedings = torch.nn.ParameterList(), torch.nn.ParameterList()
        self.pose_optimizers, self.pose_linked_rf = [], [] #, self.pose_schedulers, self.pose_iter, self.n_iters_pose = [], [], [], []
        self.append_frame(n_init_frames)

        if self.camera_prior is not None:
            focal = torch.Tensor([self.camera_prior["transforms"]["fl_x"], self.camera_prior["transforms"]["fl_y"]]).to(self.device)
            focal *= self.W / self.camera_prior["transforms"]["w"]
        else:
            fov2 = fov * math.pi / 180
            f = self.W / math.tan(fov2 / 2) / 2
            focal = torch.Tensor([f, f]).to(self.device)
        fov2 = 2 * torch.atan(torch.Tensor(WH).to(self.device) / focal / 2)
        self.fov2 = torch.nn.Parameter(fov2)

        self.focal_offset = torch.nn.Parameter(torch.zeros_like(focal))
        self.center_offset = torch.nn.Parameter(torch.zeros_like(focal))
        self.dist_coefs = torch.nn.Parameter(torch.zeros_like(focal))

        if self.camera_prior is not None:
            with torch.no_grad():
                self.focal_offset[1] = (focal[1] - focal[0]) / (self.W + self.H) * 2
        # def focal(self, W, H):
        #     # focal = torch.Tensor([W, H]).to(self.fov2) / torch.tan(self.fov2 / 2) / 2
        #     focal = W / torch.tan(self.fov2[0] / 2) / 2
        #     focal = focal.expand([2])
        #     focal = focal + self.focal_offset * (W + H) / 2

        # self.dist_coefs = torch.nn.Parameter(torch.zeros([4], device=device))
        if optimize_focal:
            self.intrinsics_grad_vars = [
                {"params": self.fov2, "lr": self.lr_pose * 1e-1},
                {"params": self.focal_offset, "lr": self.lr_pose * 1e-3},
                {"params": self.center_offset, "lr": self.lr_pose * 1e-3},
                {"params": self.dist_coefs, "lr": self.lr_pose * 1e-2},
            ]
            self.intrinsic_optimizer = torch.optim.Adam(self.intrinsics_grad_vars, betas=(0.9, 0.99))
            self.intrinsic_optimizer_init = torch.optim.Adam(copy.deepcopy(self.intrinsics_grad_vars), betas=(0.9, 0.99))


        # Setup radiance fields
        self.tensorfs = torch.nn.ParameterList()
        self.rf_optimizers, self.rf_iter = [], []
        self.world2rf = torch.nn.ParameterList()
        self.blending_weights = torch.nn.Parameter(
            torch.ones([self.n_init_frames, 1], device=self.device, requires_grad=False), 
            requires_grad=False,
        )
        self.append_rf()
        grad_vars = self.tensorfs[-1].get_optparam_groups(
            self.rf_lr_init, self.rf_lr_basis
        )
        self.default_rf_opt = (torch.optim.Adam(grad_vars, betas=(0.9, 0.99)))

    def append_rf(self, n_added_frames=1):
        if len(self.tensorfs) > 0:
            n_overlap = min(n_added_frames, self.n_overlap, self.blending_weights.shape[0] - 1)
            weights_overlap = 1 / n_overlap + torch.arange(
                0, 1, 1 / n_overlap
            )
            self.blending_weights.requires_grad = False
            self.blending_weights[-n_overlap :, -1] = 1 - weights_overlap
            new_blending_weights = torch.zeros_like(self.blending_weights[:, 0:1])
            new_blending_weights[-n_overlap :, 0] = weights_overlap
            self.blending_weights = torch.nn.Parameter(
                torch.cat([self.blending_weights, new_blending_weights], dim=1),
                requires_grad=False,
            )
            world2rf = self.get_mtx(self.cam2world[-1][-1:].detach().clone())[0]
            world2rf[:3, :3] = torch.eye(3, device=world2rf.device)
            world2rf[:3, 3] = -world2rf[:3, 3]
        else:
            world2rf = torch.eye(3, 4, device=self.device)

        self.tensorfs.append(TensorVMSplit(self.device, shadingMode=self.shadingMode, **self.tensorf_args))

        self.world2rf.append(world2rf.detach())
        
        self.rf_iter.append(0)

        if self.new_rf_init and len(self.tensorfs) > 1:
            keys = list(self.N_voxel_list.keys())
            valid_upsampling_iter = np.array(keys) <= self.rf_iter[-2]
            key_idx = valid_upsampling_iter.nonzero()[0]
            if len(key_idx) > 1:
                # key = keys[key_idx[-min(3, len(key_idx))]]
                key = keys[key_idx[-2]]
                n_voxels = self.N_voxel_list[key]
                reso_cur = N_to_reso(n_voxels, self.tensorfs[-1].aabb)
                self.tensorfs[-1].upsample_volume_grid(reso_cur)
                self.rf_iter[-1] = key + 1

            self.tensorfs[-1].init_from_prev(
                self.tensorfs[-2], 
                self.world2rf[-1][:3, 3].detach() - self.world2rf[-2][:3, 3].detach()
            )

        grad_vars = self.tensorfs[-1].get_optparam_groups(
            self.rf_lr_init, self.rf_lr_basis
        )
        self.rf_optimizers.append(torch.optim.Adam(grad_vars, betas=(0.9, 0.99)))

        print("************ blending_weights", self.blending_weights)

    def get_training_frames(self):
        training_tensorfs = np.array(self.rf_iter) < self.n_iters
        training_weights = (
            self.blending_weights.detach().cpu().numpy()[:, training_tensorfs]
        )
        return (training_weights > 0).any(axis=1)
    
    def append_frame(self, n_frames=1, reset_rf_opt=False):
        if len(self.cam2world) == 0:
            self.cam2world.append(self.init_poses(n_frames))
            if self.shadingMode == "MLP_Fea_late_view_late_emb":
                self.app_embedings.append(torch.randn([1, 16], device=self.device).repeat(n_frames, 1))
            self.pose_linked_rf.append(0)
            # self.n_iters_pose.append(self.n_iters)
        else:
            new_cams = self.get_mtx(self.cam2world[-1][-1:].repeat(n_frames, 1, 1))
            poses6d = []
            for new_cam in new_cams:
                pose = torch.zeros(3, 3).to(self.device)
                pose[:, 0] = new_cam[:3, 0]
                pose[:, 1] = new_cam[:3, 1]
                pose[:, 2] = new_cam[:3, 3]
                poses6d.append(pose)

            poses6d = torch.stack(poses6d, 0)
            self.cam2world.append(poses6d)
            if self.shadingMode == "MLP_Fea_late_view_late_emb":
                self.app_embedings.append(self.app_embedings[-1][-1:].repeat(n_frames, 1))

            self.blending_weights = torch.nn.Parameter(
                torch.cat([self.blending_weights, self.blending_weights[-1:, :]], dim=0),
                requires_grad=False,
            )

            rf_ind = torch.nonzero(self.blending_weights[-1, :])[0]
            self.pose_linked_rf.append(rf_ind)

            if reset_rf_opt:
                for param_group, param_group_init in zip(self.rf_optimizers[rf_ind].param_groups, self.default_rf_opt.param_groups):
                    param_group["lr"] = param_group_init["lr"]
                
                upsampling_iterations = np.array([-1] + list(self.N_voxel_list.keys()))
                last_upsampling_iteration = upsampling_iterations[upsampling_iterations < self.rf_iter[rf_ind]].max()
                self.rf_iter[rf_ind] = last_upsampling_iteration + 1

                
            # Prevent optimizing poses after the current RF is done optimizing
            # rf_ind = torch.nonzero(self.blending_weights[-1, :])[0]
            # self.n_iters_pose.append(max(self.n_iters - self.rf_iter[rf_ind], 1)) #  - self.rf_iter[rf_ind]

        if self.camera_prior is not None:
            rel_poses = self.camera_prior["rel_poses"]
            poses6d = []
            cam2world = self.get_cam2world()
            for idx in range(len(cam2world) - n_frames, len(cam2world)):
                prev_c2w = cam2world[max(idx - 1, 0)].clone()
                cam2world[idx] = prev_c2w.clone()
                cam2world[idx][:3, :3] = prev_c2w[:3, :3] @ rel_poses[idx][:3, :3]
                cam2world[idx][:3, 3] += prev_c2w[:3, :3] @ rel_poses[idx][:3, 3]

                pose = torch.zeros(3, 3).to(self.device)
                pose[:, 0] = cam2world[idx, :3, 0]
                pose[:, 1] = cam2world[idx, :3, 1]
                pose[:, 2] = cam2world[idx, :3, 3]
                poses6d.append(pose)

            poses6d = torch.stack(poses6d, 0)
            self.cam2world[-1].data = poses6d

        if self.shadingMode == "MLP_Fea_late_view_late_emb":
            self.pose_optimizers.append(torch.optim.Adam([self.cam2world[-1], self.app_embedings[-1]], betas=(0.9, 0.99), lr=self.lr_pose)) # , betas=(0.9, 0.99)
        else:
            self.pose_optimizers.append(torch.optim.Adam([self.cam2world[-1]], betas=(0.9, 0.99), lr=self.lr_pose)) # , betas=(0.9, 0.99)
        # lr_pose_end = 1e-5  # 5:X, 10:X
        # self.pose_optimizers.append(torch.optim.Adam([self.cam2world[-1]], lr=lr_pose))
        # gamma = (lr_pose_end / lr_pose) ** (1.0 / (self.n_iters_pose[-1]))
        # self.pose_schedulers.append(
        #     torch.optim.lr_scheduler.ExponentialLR(
        #         self.pose_optimizers[-1], gamma=gamma
        #     )
        # )

    def optimizer_step_poses_only(self, loss):
        for idx in range(len(self.pose_optimizers)):
            iteration = self.rf_iter[self.pose_linked_rf[idx]]
            if iteration < self.n_iters - 10:
                for param_group in self.pose_optimizers[idx].param_groups:
                    param_group["lr"] = self.lr_pose * max((self.rf_lr_factor ** (iteration)), 0)
                self.pose_optimizers[idx].zero_grad()
        
        loss.backward()

        # Optimize poses
        for idx in range(len(self.cam2world)):
            iteration = self.rf_iter[self.pose_linked_rf[idx]]
            if iteration < self.n_iters - 10:
                self.pose_optimizers[idx].step()
                
    def optimizer_step(self, loss, optimize_poses):
        if optimize_poses:
            # Poses
            for idx in range(len(self.pose_optimizers)):
                iteration = self.rf_iter[self.pose_linked_rf[idx]]
                if iteration < self.n_iters - 10:
                    for param_group in self.pose_optimizers[idx].param_groups:
                        param_group["lr"] = self.lr_pose * max((self.rf_lr_factor ** (iteration)), 0)
                    self.pose_optimizers[idx].zero_grad()
            
            # Intrinsics
            if self.optimize_focal and self.rf_iter[0] < self.n_iters - 10:
                warmup_iters = 200
                loss_scaler = (self.rf_iter[0] / warmup_iters) if self.rf_iter[0] < warmup_iters else (self.rf_lr_factor ** (self.rf_iter[0]))
                loss_scaler = max(loss_scaler, 0)
                # loss_scaler = self.rf_lr_factor ** (self.rf_iter[0])
                for param_group, param_group_init in zip(self.intrinsic_optimizer.param_groups, self.intrinsic_optimizer_init.param_groups):
                    param_group["lr"] = param_group_init["lr"] * loss_scaler
                self.intrinsic_optimizer.zero_grad()

        # tensorfs
        for optimizer, iteration in zip(self.rf_optimizers, self.rf_iter):
            if iteration < self.n_iters:
                optimizer.zero_grad()

        loss.backward()

        # Optimize RFs
        should_add_rf = False
        for idx in range(len(self.tensorfs)):
            if self.rf_iter[idx] < self.n_iters:
                self.rf_optimizers[idx].step()
                for param_group in self.rf_optimizers[idx].param_groups:
                    param_group["lr"] = param_group["lr"] * self.rf_lr_factor

                # Increase RF resolution
                if self.rf_iter[idx] in self.N_voxel_list:
                    n_voxels = self.N_voxel_list[self.rf_iter[idx]]
                    reso_cur = N_to_reso(n_voxels, self.tensorfs[idx].aabb)
                    self.tensorfs[idx].upsample_volume_grid(reso_cur)

                    if self.lr_upsample_reset:
                        print("reset lr to initial")
                        grad_vars = self.tensorfs[idx].get_optparam_groups(
                            self.rf_lr_init, self.rf_lr_basis
                        )
                        self.rf_optimizers[idx] = torch.optim.Adam(
                            grad_vars, betas=(0.9, 0.99)
                        )

                if self.rf_iter[idx] == self.n_iters - 1:
                    should_add_rf = True

                if self.is_refining:
                    self.rf_iter[idx] += 1

        if optimize_poses:
            # Optimize poses
            for idx in range(len(self.cam2world)):
                iteration = self.rf_iter[self.pose_linked_rf[idx]]
                if iteration < self.n_iters:
                    self.pose_optimizers[idx].step()
            
            if self.optimize_focal and self.rf_iter[0] < self.n_iters:
                self.intrinsic_optimizer.step()

        # Stop when no more radiance field is optimizing
        return (np.array(self.rf_iter) < self.n_iters).any(), should_add_rf

    def init_poses(self, n_poses):
        poses = torch.zeros([n_poses, 3, 3], device=self.device)
        poses[:, 0, 0] = 1
        poses[:, 1, 1] = 1

        return poses

    def get_mtx(self, poses):
        return sixD_to_mtx(poses, 1)

    def get_cam2world(self, view_ids=None):
        cam2world = torch.cat(list(self.cam2world), dim=0)
        if view_ids is None:
            return self.get_mtx(cam2world)
        else:
            return self.get_mtx(cam2world[view_ids])

    def get_kwargs(self):
        kwargs = {
            "fov": self.fov,
            "pose_representation": self.pose_representation,
            "ray_type": self.ray_type,
            "n_init_frames": self.n_init_frames,
            "n_overlap": self.n_overlap,
            "blending_mode": self.blending_mode,
            "WH": (self.W, self.H),
            "n_iters": self.n_iters,
            "new_rf_init": self.new_rf_init,
            "rf_lr_init": self.rf_lr_init,
            "rf_lr_basis": self.rf_lr_basis,
            "lr_decay_target_ratio": self.lr_decay_target_ratio,
            "rf_lr_factor": self.rf_lr_factor,
            "N_voxel_list": self.N_voxel_list,
            "lr_upsample_reset": self.lr_upsample_reset,
            "optimize_focal": self.optimize_focal,
        }
        kwargs.update(self.tensorfs[0].get_kwargs())

        return kwargs

    def save(self, path):
        kwargs = self.get_kwargs()
        ckpt = {"kwargs": kwargs, "state_dict": self.state_dict()}
        torch.save(ckpt, path)

    def load(self, state_dict):
        # TODO A bit hacky?
        import re
        for key in state_dict:
            if re.fullmatch(r"cam2world.[1-9][0-9]*", key):
                self.append_frame(state_dict[key].shape[0])
            if re.fullmatch(r"tensorfs.[1-9][0-9]*.density_plane.0", key):
                self.tensorf_args["gridSize"] = [state_dict[key].shape[2], state_dict[key].shape[3], state_dict[f"{key[:-15]}density_line.0"].shape[2]]
                self.append_rf()

        self.blending_weights = torch.nn.Parameter(
            torch.ones_like(state_dict["blending_weights"]), requires_grad=False
        )

        self.load_state_dict(state_dict)

    def get_cam2rf(self, world2rf, cam2world, active_rf_ids):
        cam2rfs = {}
        for rf_id in active_rf_ids:
            cam2rf = cam2world.clone()
            cam2rf[:, :3, 3] += world2rf[rf_id][:3, 3][None]

            cam2rfs[rf_id] = cam2rf
        return cam2rfs

    def get_dist_to_last_rf(self):
        cam2rf = self.get_cam2rf(self.world2rf, self.get_cam2world()[-1][None], [-1])[-1]
        return torch.norm(cam2rf[0, :3, 3])

    def get_reg_loss(self, tvreg, TV_weight_density, TV_weight_app, L1_weight_inital):
        tv_loss = 0
        l1_loss = 0
        for tensorf, iteration in zip(self.tensorfs, self.rf_iter):
            if iteration < self.n_iters:
                if TV_weight_density > 0:
                    tv_weight = TV_weight_density * (self.rf_lr_factor ** iteration)
                    tv_loss += tensorf.TV_loss_density(tvreg).mean() * tv_weight
                    
                    tv_weight = TV_weight_app * self.rf_lr_factor ** iteration
                    tv_loss += tensorf.TV_loss_app(tvreg).mean() * tv_weight
        
                if L1_weight_inital > 0:
                    l1_loss += tensorf.density_L1() * L1_weight_inital # * (self.rf_lr_factor ** iteration)
        return tv_loss, l1_loss

    def focal(self, W, H):
        # focal = torch.Tensor([W, H]).to(self.fov2) / torch.tan(self.fov2 / 2) / 2
        focal = W / torch.tan(self.fov2[0] / 2) / 2
        focal = focal.expand([2])
        focal = focal + self.focal_offset * (W + H) / 2
        return focal

    def center(self, W, H):
        center = torch.Tensor([W / 2, H / 2]).to(self.center_offset)
        center = center + self.center_offset * (W + H) / 2
        return center

    def forward(
        self,
        ray_ids,
        view_ids,
        W,
        H,
        white_bg=True,
        is_train=True,
        cam2world=None,
        world2rf=None,
        blending_weights=None,
        opasity_gamma=1.0,
        distortion_loss_weight=0,
        chunk=4096,
        test=False,
    ):
        focal = self.focal(W, H)
        i, j = ids2pixel(W, H, ray_ids)
        directions = get_ray_directions_lean(i, j, focal, self.center(W, H), self.dist_coefs)

        if blending_weights is None:
            blending_weights = self.blending_weights[view_ids].clone()
        if cam2world is None:
            cam2world = self.get_cam2world(view_ids)
        elif cam2world.shape[0] == 1:
            view_ids *= 0
        if world2rf is None:
            world2rf = self.world2rf

        # Binarize weights for blending so we query a single RF per ray
        if is_train:# and self.blending_mode == "image":
            non_training_rf = torch.tensor(self.rf_iter).to(ray_ids) >= self.n_iters
            blending_weights = blending_weights.detach().clone()
            blending_weights[:, non_training_rf] = 0
            blending_weights[blending_weights > 0] = torch.clamp(torch.rand_like(blending_weights[blending_weights > 0]), 1e-6)
            blending_weights /= torch.sum(blending_weights, dim=1, keepdim=True)
            blending_weights = torch.nan_to_num(blending_weights, 0, 0, 0)

            blending_weights[blending_weights > 0.5] = 1
            blending_weights[blending_weights < 0.5] = 0

        # TODO Remove
        # blending_weights[blending_weights > 0.5] = 1
        # blending_weights[blending_weights < 0.5] = 0

        active_rf_ids = torch.nonzero(torch.sum(blending_weights, dim=0))[:, 0].tolist()
        ij = torch.stack([i, j], dim=-1)
        if len(active_rf_ids) == 0:
            print("****** No good RF")
            return torch.ones([ray_ids.shape[0], 3]), torch.ones_like(ray_ids).float(), torch.ones_like(ray_ids).float(), directions, ij, blending_weights, 0

        cam2rfs = self.get_cam2rf(world2rf, cam2world, active_rf_ids)

        if self.shadingMode == "MLP_Fea_late_view_late_emb":
            app_embedings = torch.cat(list(self.app_embedings), 0)
            app_embedings = app_embedings[view_ids]
        else:
            app_embedings = None

        # view_ids_expanded = view_ids.repeat_interleave(ray_ids.shape[0] // view_ids.shape[0])
        for key in cam2rfs:
            cam2rfs[key] = cam2rfs[key].repeat_interleave(ray_ids.shape[0] // view_ids.shape[0], dim=0)
        blending_weights_expanded = blending_weights.repeat_interleave(ray_ids.shape[0] // view_ids.shape[0], dim=0)
        rgbs, depth_maps, depth_maps_gamma = [], [], []
        N_rays_all = ray_ids.shape[0]
        chunk = chunk // len(active_rf_ids)
        dist_loss = torch.zeros(1, device=ray_ids.device)
        for chunk_idx in range(N_rays_all // chunk + int(N_rays_all % chunk > 0)):
            if chunk_idx != 0:
                torch.cuda.empty_cache()
            if self.shadingMode == "MLP_Fea_late_view_late_emb":
                app_embedings_chunk = app_embedings[chunk_idx * chunk : (chunk_idx + 1) * chunk]
            directions_chunk = directions[chunk_idx * chunk : (chunk_idx + 1) * chunk]
            # view_ids_chunk = view_ids_expanded[chunk_idx * chunk : (chunk_idx + 1) * chunk]
            blending_weights_chunk = blending_weights_expanded[
                chunk_idx * chunk : (chunk_idx + 1) * chunk
            ]

            all_z_val, all_rgb, all_sigma = [], [], []
            rgb_map, depth_map, depth_map_gamma = 0, 0, 0
            for rf_id in active_rf_ids:
                blending_weight_chunk = blending_weights_chunk[:, rf_id]
                cam2rf = cam2rfs[rf_id][chunk_idx * chunk : (chunk_idx + 1) * chunk]
                # cam2rf = cam2rf[view_ids_chunk]

                rays_o, rays_d = get_rays_lean(directions_chunk, cam2rf)
                if self.ray_type == "ndc":
                    rays_o, rays_d = ndc_rays_blender2(H, W, focal, 1.0, rays_o, rays_d)
                rays = torch.cat([rays_o, rays_d], -1).view(-1, 6)

                valid = blending_weight_chunk > 0
                # if self.blending_mode == "image":
                rgb_map_t = torch.zeros_like(rays[:, :3])
                depth_map_t = torch.zeros_like(rays[:, 0])
                depth_map_gamma_t = torch.zeros_like(rays[:, 0])
                rgb_map_t[valid], depth_map_t[valid], depth_map_gamma_t[valid], distances, dists, alpha, T, weight, alpha_gamma, T_gamma, weight_gamma = self.tensorfs[rf_id](
                    rays[valid],
                    is_train=is_train,
                    white_bg=white_bg,
                    ray_type=self.ray_type,
                    N_samples=-1,
                    opasity_gamma=opasity_gamma,
                    app_embedings=app_embedings_chunk[valid] if self.shadingMode == "MLP_Fea_late_view_late_emb" else None,
                )

                rgb_map = rgb_map + rgb_map_t * blending_weight_chunk[..., None]
                depth_map = depth_map + depth_map_t * blending_weight_chunk
                depth_map_gamma = depth_map_gamma + depth_map_gamma_t * blending_weight_chunk

                if is_train and distortion_loss_weight > 0:
                    dist_loss += eff_distloss_native(weight[:, :-1], distances[:, :-1], dists[:, :-1]) * (
                        distortion_loss_weight * (self.rf_lr_factor ** self.rf_iter[rf_id]))

            rgbs.append(rgb_map)
            depth_maps.append(depth_map)
            depth_maps_gamma.append(depth_map_gamma)

        rgbs, depth_maps, depth_maps_gamma = torch.cat(rgbs), torch.cat(depth_maps), torch.cat(depth_maps_gamma)

        if is_train:
            return rgbs, depth_maps, depth_maps_gamma, directions, ij, blending_weights, dist_loss
        else:
            sample = len(alpha) // 2
            return rgbs, depth_maps, depth_maps_gamma, directions, ij, alpha[sample], T[sample], weight[sample], alpha_gamma[sample], T_gamma[sample], weight_gamma[sample]