from cProfile import label
import time

import numpy as np
import torch
import torch.nn
import torch.nn.functional as F

from utils.ray_utils import contract, uncontract

from .sh import eval_sh_bases


def positional_encoding(positions, freqs):

    freq_bands = (2 ** torch.arange(freqs, device=positions.device).float())  # (F,)
    pts = (positions[..., None] * freq_bands).reshape(
        positions.shape[:-1] + (freqs * positions.shape[-1],)
    )  # (..., DF)
    pts = torch.cat([torch.sin(pts), torch.cos(pts)], dim=-1)
    return pts

def alpha2weights(alpha):
    alpha[:, -1] = 1
    T = torch.cumprod(
        torch.cat(
            [torch.ones(alpha.shape[0], 1, device=alpha.device), 1.0 - alpha + 1e-10], -1
        ),
        -1,
    )
    weights = alpha * T[:, :-1]  # [N_rays, N_samples]
    return weights, T #[:, -1:]

# def raw2alpha(sigma, dist):
#     # sigma, dist  [N_rays, N_samples]
#     alpha = 1.0 - torch.exp(-sigma * dist)

#     weights, T = alpha2weights(alpha)
#     return alpha, weights, T


def SHRender(xyz_sampled, viewdirs, features):
    sh_mult = eval_sh_bases(2, viewdirs)[:, None]
    rgb_sh = features.view(-1, 3, sh_mult.shape[-1])
    rgb = torch.relu(torch.sum(sh_mult * rgb_sh, dim=-1) + 0.5)
    return rgb


def RGBRender(xyz_sampled, viewdirs, features):

    rgb = features
    return rgb


class MLPRender_Fea(torch.nn.Module):
    def __init__(self, inChanel, viewpe=6, feape=6, featureC=128):
        super(MLPRender_Fea, self).__init__()

        self.in_mlpC = 2 * viewpe * 3 + 2 * feape * inChanel + 3 + inChanel
        self.viewpe = viewpe
        self.feape = feape
        layer1 = torch.nn.Linear(self.in_mlpC, featureC)
        layer2 = torch.nn.Linear(featureC, featureC)
        layer3 = torch.nn.Linear(featureC, 3)

        self.mlp = torch.nn.Sequential(
            layer1,
            torch.nn.ReLU(inplace=True),
            layer2,
            torch.nn.ReLU(inplace=True),
            layer3,
        )
        torch.nn.init.constant_(self.mlp[-1].bias, 0)

    def forward(self, pts, viewdirs, features):
        indata = [features, viewdirs]
        if self.feape > 0:
            indata += [positional_encoding(features, self.feape)]
        if self.viewpe > 0:
            indata += [positional_encoding(viewdirs, self.viewpe)]
        mlp_in = torch.cat(indata, dim=-1)
        rgb = self.mlp(mlp_in)
        rgb = torch.sigmoid(rgb)

        return rgb


class MLPRender_Fea_late_view(torch.nn.Module):
    def __init__(self, inChanel, viewpe=6, feape=6, featureC=128):
        super(MLPRender_Fea_late_view, self).__init__()

        self.in_mlpC = 2 * feape * inChanel + inChanel
        self.in_view = 2 * viewpe * 3 + 3
        self.viewpe = viewpe
        self.feape = feape
        layer1 = torch.nn.Linear(self.in_mlpC, featureC)
        layer2 = torch.nn.Linear(featureC, featureC)
        layer3 = torch.nn.Linear(featureC + self.in_view, 3)

        self.mlp = torch.nn.Sequential(
            layer1, torch.nn.ReLU(inplace=True), layer2, torch.nn.ReLU(inplace=True)
        )
        self.mlp_view = torch.nn.Sequential(layer3)
        torch.nn.init.constant_(self.mlp_view[-1].bias, 0)

    def forward(self, pts, viewdirs, features):
        indata = [features]
        if self.feape > 0:
            indata += [positional_encoding(features, self.feape)]
        indata_view = [viewdirs]
        if self.viewpe > 0:
            indata_view += [positional_encoding(viewdirs, self.viewpe)]
        mlp_in = torch.cat(indata, dim=-1)
        inter_features = self.mlp(mlp_in)
        mlp_view_in = torch.cat([inter_features] + indata_view, dim=-1)
        rgb = self.mlp_view(mlp_view_in)
        rgb = torch.sigmoid(rgb)

        return rgb

class MLPRender_Fea_late_view_late_emb(torch.nn.Module):
    def __init__(self, inChanel, viewpe=6, feape=6, featureC=128, emb_n=16):
        super(MLPRender_Fea_late_view_late_emb, self).__init__()

        self.in_mlpC = 2 * feape * inChanel + inChanel
        self.in_view = 2 * viewpe * 3 + 3
        self.in_emb  = emb_n
        self.viewpe = viewpe
        self.feape = feape
        layer1 = torch.nn.Linear(self.in_mlpC, featureC)
        layer2 = torch.nn.Linear(featureC, featureC)
        layer3 = torch.nn.Linear(featureC + self.in_view + self.in_emb, 3)

        self.mlp = torch.nn.Sequential(
            layer1, torch.nn.ReLU(inplace=True), layer2, torch.nn.ReLU(inplace=True)
        )
        self.mlp_view = torch.nn.Sequential(layer3)
        torch.nn.init.constant_(self.mlp_view[-1].bias, 0)

    def forward(self, pts, viewdirs, features, embedding):
        indata = [features]
        if self.feape > 0:
            indata += [positional_encoding(features, self.feape)]
        indata_view = [viewdirs]
        if self.viewpe > 0:
            indata_view += [positional_encoding(viewdirs, self.viewpe)]
        mlp_in = torch.cat(indata, dim=-1)
        inter_features = self.mlp(mlp_in)
        mlp_view_in = torch.cat([inter_features] + indata_view + [embedding], dim=-1)
        rgb = self.mlp_view(mlp_view_in)
        rgb = torch.sigmoid(rgb)

        return rgb

class MLPRender_Fea_woView(torch.nn.Module):
    def __init__(self, inChanel, viewpe=6, feape=6, featureC=128):
        super(MLPRender_Fea_woView, self).__init__()

        self.in_mlpC = 2 * feape * inChanel + inChanel
        self.viewpe = viewpe
        self.feape = feape
        layer1 = torch.nn.Linear(self.in_mlpC, featureC)
        layer2 = torch.nn.Linear(featureC, featureC)
        layer3 = torch.nn.Linear(featureC, 3)

        self.mlp = torch.nn.Sequential(
            layer1,
            torch.nn.ReLU(inplace=True),
            layer2,
            torch.nn.ReLU(inplace=True),
            layer3,
        )
        torch.nn.init.constant_(self.mlp[-1].bias, 0)

    def forward(self, pts, viewdirs, features):
        indata = [features]
        if self.feape > 0:
            indata += [positional_encoding(features, self.feape)]
        mlp_in = torch.cat(indata, dim=-1)
        rgb = self.mlp(mlp_in)
        rgb = torch.sigmoid(rgb)

        return rgb


class MLPRender_PE(torch.nn.Module):
    def __init__(self, inChanel, viewpe=6, pospe=6, featureC=128):
        super(MLPRender_PE, self).__init__()

        self.in_mlpC = (3 + 2 * viewpe * 3) + (3 + 2 * pospe * 3) + inChanel  #
        self.viewpe = viewpe
        self.pospe = pospe
        layer1 = torch.nn.Linear(self.in_mlpC, featureC)
        layer2 = torch.nn.Linear(featureC, featureC)
        layer3 = torch.nn.Linear(featureC, 3)

        self.mlp = torch.nn.Sequential(
            layer1,
            torch.nn.ReLU(inplace=True),
            layer2,
            torch.nn.ReLU(inplace=True),
            layer3,
        )
        torch.nn.init.constant_(self.mlp[-1].bias, 0)

    def forward(self, pts, viewdirs, features):
        indata = [features, viewdirs]
        if self.pospe > 0:
            indata += [positional_encoding(pts, self.pospe)]
        if self.viewpe > 0:
            indata += [positional_encoding(viewdirs, self.viewpe)]
        mlp_in = torch.cat(indata, dim=-1)
        rgb = self.mlp(mlp_in)
        rgb = torch.sigmoid(rgb)

        return rgb


class MLPRender(torch.nn.Module):
    def __init__(self, inChanel, viewpe=6, featureC=128):
        super(MLPRender, self).__init__()

        self.in_mlpC = (3 + 2 * viewpe * 3) + inChanel
        self.viewpe = viewpe

        layer1 = torch.nn.Linear(self.in_mlpC, featureC)
        layer2 = torch.nn.Linear(featureC, featureC)
        layer3 = torch.nn.Linear(featureC, 3)

        self.mlp = torch.nn.Sequential(
            layer1,
            torch.nn.ReLU(inplace=True),
            layer2,
            torch.nn.ReLU(inplace=True),
            layer3,
        )
        torch.nn.init.constant_(self.mlp[-1].bias, 0)

    def forward(self, pts, viewdirs, features):
        indata = [features, viewdirs]
        if self.viewpe > 0:
            indata += [positional_encoding(viewdirs, self.viewpe)]
        mlp_in = torch.cat(indata, dim=-1)
        rgb = self.mlp(mlp_in)
        rgb = torch.sigmoid(rgb)

        return rgb

# def render_from_samples(
#     sigma, rgb, z_vals, dists, white_bg, is_train, distance_scale=25, blending_weights=1
# ):

#     # Can be redundant to compute weights here
#     alpha, weight, bg_weight = raw2alpha(
#         sigma, dists * distance_scale, blending_weights
#     )

#     acc_map = torch.sum(weight, -1)
#     rgb_map = torch.sum(weight[..., None] * rgb, -2)

#     if white_bg or (is_train and torch.rand((1,)) < 0.5):
#         rgb_map = rgb_map + (1.0 - acc_map[..., None])

#     rgb_map = rgb_map.clamp(0, 1)

#     depth_map = torch.sum(weight * z_vals, -1)

#     return rgb_map, depth_map, weight


class TensorBase(torch.nn.Module):
    def __init__(
        self,
        device,
        aabb,
        gridSize,
        density_n_comp=8,
        appearance_n_comp=24,
        app_dim=27,
        shadingMode="MLP_PE",
        alphaMask=None,
        near_far=[2.0, 6.0],
        density_shift=-10,
        distance_scale=25,
        rayMarch_weight_thres=0.0001,
        pos_pe=6,
        view_pe=6,
        fea_pe=6,
        featureC=128,
        step_ratio=2.0,
        fea2denseAct="softplus",
    ):
        super(TensorBase, self).__init__()

        self.density_n_comp = density_n_comp.copy()
        self.app_n_comp = appearance_n_comp.copy()
        self.app_dim = app_dim
        self.aabb = torch.nn.Parameter(aabb, requires_grad=False)
        self.alphaMask = alphaMask
        self.device = device

        self.density_shift = density_shift
        self.distance_scale = distance_scale
        self.rayMarch_weight_thres = rayMarch_weight_thres
        self.fea2denseAct = fea2denseAct

        self.near_far = near_far.copy()
        self.step_ratio = step_ratio

        self.update_stepSize(gridSize.copy())

        self.matMode = [[0, 1], [0, 2], [1, 2]]
        self.vecMode = [2, 1, 0]
        self.comp_w = [1, 1, 1]

        self.init_svd_volume(gridSize, device)

        self.shadingMode, self.pos_pe, self.view_pe, self.fea_pe, self.featureC = (
            shadingMode,
            pos_pe,
            view_pe,
            fea_pe,
            featureC,
        )
        self.init_render_func(shadingMode, pos_pe, view_pe, fea_pe, featureC, device)

    def init_render_func(self, shadingMode, pos_pe, view_pe, fea_pe, featureC, device):
        if shadingMode == "MLP_PE":
            self.renderModule = MLPRender_PE(
                self.app_dim, view_pe, pos_pe, featureC
            ).to(device)
        elif shadingMode == "MLP_Fea":
            self.renderModule = MLPRender_Fea(
                self.app_dim, view_pe, fea_pe, featureC
            ).to(device)
        elif shadingMode == "MLP_Fea_late_view":
            self.renderModule = MLPRender_Fea_late_view(
                self.app_dim, view_pe, fea_pe, featureC
            ).to(device)
        elif shadingMode == "MLP_Fea_late_view_late_emb":
            self.renderModule = MLPRender_Fea_late_view_late_emb(
                self.app_dim, view_pe, fea_pe, featureC
            ).to(device)
        elif shadingMode == "MLP_Fea_woView":
            self.renderModule = MLPRender_Fea_woView(
                self.app_dim, view_pe, fea_pe, featureC
            ).to(device)
        elif shadingMode == "MLP":
            self.renderModule = MLPRender(self.app_dim, view_pe, featureC).to(device)
        elif shadingMode == "SH":
            self.renderModule = SHRender
        elif shadingMode == "RGB":
            assert self.app_dim == 3
            self.renderModule = RGBRender
        else:
            print("Unrecognized shading module")
            exit()
        print("pos_pe", pos_pe, "view_pe", view_pe, "fea_pe", fea_pe)
        print(self.renderModule)

    def update_stepSize(self, gridSize):
        print("aabb", self.aabb.view(-1))
        print("grid size", gridSize)
        self.aabbSize = self.aabb[1] - self.aabb[0]
        self.invaabbSize = torch.nn.Parameter(2.0 / self.aabbSize, requires_grad=False)
        self.gridSize = torch.LongTensor(gridSize).to(self.device)
        self.units = self.aabbSize / (self.gridSize - 1)
        self.stepSize = torch.mean(self.units) * self.step_ratio
        self.aabbDiag = torch.sqrt(torch.sum(torch.square(self.aabbSize)))
        self.nSamples = int((self.aabbDiag / self.stepSize).item()) + 1
        print("sampling step size: ", self.stepSize)
        print("sampling number: ", self.nSamples)

    def init_svd_volume(self, res, device):
        pass

    def compute_features(self, xyz_sampled):
        pass

    def compute_densityfeature(self, xyz_sampled):
        pass

    def compute_appfeature(self, xyz_sampled):
        pass

    def normalize_coord(self, xyz_sampled):
        return (
            xyz_sampled - self.aabb[0]
        ) * self.invaabbSize - 1

    def get_optparam_groups(self, lr_init_spatial=0.02, lr_init_network=0.001):
        pass

    def get_kwargs(self):
        return {
            "aabb": self.aabb,
            "gridSize": self.gridSize.tolist(),
            "density_n_comp": self.density_n_comp,
            "appearance_n_comp": self.app_n_comp,
            "app_dim": self.app_dim,
            "density_shift": self.density_shift,
            "distance_scale": self.distance_scale,
            "rayMarch_weight_thres": self.rayMarch_weight_thres,
            "fea2denseAct": self.fea2denseAct,
            "near_far": self.near_far,
            "step_ratio": self.step_ratio,
            "shadingMode": self.shadingMode,
            "pos_pe": self.pos_pe,
            "view_pe": self.view_pe,
            "fea_pe": self.fea_pe,
            "featureC": self.featureC,
        }

    def save(self, se3_poses, path):
        kwargs = self.get_kwargs()
        kwargs["se3_poses"] = se3_poses
        ckpt = {"kwargs": kwargs, "state_dict": self.state_dict()}
        if self.alphaMask is not None:
            alpha_volume = self.alphaMask.alpha_volume.bool().cpu().numpy()
            ckpt.update({"alphaMask.shape": alpha_volume.shape})
            ckpt.update({"alphaMask.mask": np.packbits(alpha_volume.reshape(-1))})
            ckpt.update({"alphaMask.aabb": self.alphaMask.aabb.cpu()})
        torch.save(ckpt, path)

    def sample_ray_ndc(self, rays_o, rays_d, is_train=True, N_samples=-1):
        N_samples = N_samples if N_samples > 0 else self.nSamples
        near, far = self.near_far
        interpx = torch.linspace(near, far, N_samples).unsqueeze(0).to(rays_o)
        if is_train:
            interpx += torch.rand_like(interpx).to(rays_o) * ((far - near) / N_samples)

        rays_pts = rays_o[..., None, :] + rays_d[..., None, :] * interpx[..., None]
        mask_outbbox = (
            (self.aabb.to(rays_pts.get_device())[0] > rays_pts)
            | (rays_pts > self.aabb.to(rays_pts.get_device())[1])
        ).any(dim=-1)
        return rays_pts, interpx, ~mask_outbbox

    def sample_ray(self, rays_o, rays_d, is_train=True, N_samples=-1):
        N_samples = N_samples if N_samples > 0 else self.nSamples
        stepsize = self.stepSize
        near, far = self.near_far
        vec = torch.where(rays_d == 0, torch.full_like(rays_d, 1e-6), rays_d)
        rate_a = (self.aabb[1] - rays_o) / vec
        rate_b = (self.aabb[0] - rays_o) / vec
        t_min = torch.minimum(rate_a, rate_b).amax(-1).clamp(min=near, max=far)

        rng = torch.arange(N_samples)[None].float()
        if is_train:
            rng = rng.repeat(rays_d.shape[-2], 1)
            rng += torch.rand_like(rng[:, [0]])
        step = stepsize * rng.to(rays_o.device)
        interpx = t_min[..., None] + step

        rays_pts = rays_o[..., None, :] + rays_d[..., None, :] * interpx[..., None]
        mask_outbbox = ((self.aabb[0] > rays_pts) | (rays_pts > self.aabb[1])).any(
            dim=-1
        )

        return rays_pts, interpx, ~mask_outbbox

    def sample_ray_contracted(self, rays_o, rays_d, is_train=True, N_samples=-1):
        N_samples = N_samples if N_samples > 0 else self.nSamples
        N_samples = N_samples // 4 # TODO f

        t_vals = (
            torch.linspace(0.0, N_samples - 1, N_samples, device=rays_o.device)[None] / N_samples
        )

        interpx = t_vals.clone()

        if is_train:
            interpx += torch.rand_like(t_vals) / N_samples
            t_vals += torch.rand_like(t_vals) / N_samples

        near, far = [1, 1e3]
        interpx = torch.cat(
            [interpx, 1.0 / (1.0 / near * (1.0 - t_vals) + 1.0 / far * t_vals)], dim=1
        )
        interpx += 5e-2
        rays_pts = rays_o[..., None, :] + rays_d[..., None, :] * interpx[..., None]

        rays_pts = contract(rays_pts)

        # rays_o expected to be in contracted space
        # dist_from_cam = torch.norm(rays_pts - rays_o[..., None, :], dim=-1)

        mask_outbbox = torch.zeros_like(rays_pts[..., 0]) > 0
        return rays_pts, interpx, ~mask_outbbox#, dist_from_cam

    # def sample_ray_contracted(self, rays_o, rays_d, is_train=True, N_samples=-1):
    #     N_samples = N_samples if N_samples > 0 else self.nSamples
    #     N_samples = N_samples // 2
    #     near, far = self.near_far

    #     interpx = torch.arange(N_samples, device=rays_o.device, dtype=torch.float32)[None]

    #     if is_train:
    #         interpx += torch.rand_like(interpx)

    #     interpx = near + (interpx / N_samples) * (far - near)
    #     uncontract(interpx)

    #     rays_pts = rays_o[..., None, :] + rays_d[..., None, :] * interpx[..., None]

    #     contract(rays_pts)

    #     return rays_pts*1.1, interpx, torch.ones_like(rays_pts[..., 0]) > 0

    def shrink(self, new_aabb, voxel_size):
        pass

    @torch.no_grad()
    def filtering_rays(
        self, all_rays, all_rgbs, N_samples=256, chunk=10240 * 5, bbox_only=False
    ):
        print("========> filtering rays ...")
        tt = time.time()

        N = torch.tensor(all_rays.shape[:-1]).prod()

        mask_filtered = []
        idx_chunks = torch.split(torch.arange(N), chunk)
        for idx_chunk in idx_chunks:
            rays_chunk = all_rays[idx_chunk].to(self.device)

            rays_o, rays_d = rays_chunk[..., :3], rays_chunk[..., 3:6]
            if bbox_only:
                vec = torch.where(rays_d == 0, torch.full_like(rays_d, 1e-6), rays_d)
                rate_a = (self.aabb[1] - rays_o) / vec
                rate_b = (self.aabb[0] - rays_o) / vec
                t_min = torch.minimum(rate_a, rate_b).amax(
                    -1
                )  # .clamp(min=near, max=far)
                t_max = torch.maximum(rate_a, rate_b).amin(
                    -1
                )  # .clamp(min=near, max=far)
                mask_inbbox = t_max > t_min

            else:
                xyz_sampled, _, _ = self.sample_ray(
                    rays_o, rays_d, N_samples=N_samples, is_train=False
                )
                mask_inbbox = (
                    self.alphaMask.sample_alpha(xyz_sampled).view(
                        xyz_sampled.shape[:-1]
                    )
                    > 0
                ).any(-1)

            mask_filtered.append(mask_inbbox.cpu())

        mask_filtered = torch.cat(mask_filtered).view(all_rgbs.shape[:-1])

        print(
            f"Ray filtering done! takes {time.time()-tt} s. ray mask ratio: {torch.sum(mask_filtered) / N}"
        )
        return all_rays[mask_filtered], all_rgbs[mask_filtered]

    def feature2density(self, density_features):
        if self.fea2denseAct == "softplus":
            return F.softplus(density_features + self.density_shift)
        elif self.fea2denseAct == "relu":
            return F.relu(density_features)

    def forward(
        self,
        rays_chunk,
        white_bg=True,
        is_train=False,
        ray_type="ndc",
        N_samples=-1,
        opasity_gamma=1.0,
        app_embedings=None,
    ):

        # sample points
        viewdirs = rays_chunk[:, 3:6]
        viewdirs_norm = torch.norm(viewdirs, dim=-1, keepdim=True)
        if ray_type == "ndc":
            xyz_sampled, z_vals, ray_valid = self.sample_ray_ndc(
                rays_chunk[:, :3], viewdirs, is_train=is_train, N_samples=N_samples
            )
            dists = torch.cat(
                (z_vals[:, 1:] - z_vals[:, :-1], torch.zeros_like(z_vals[:, :1])),
                dim=-1,
            )
            dists = dists * viewdirs_norm
        elif ray_type == "contract":
            xyz_sampled, z_vals, ray_valid = self.sample_ray_contracted(
                rays_chunk[:, :3], viewdirs, is_train=is_train, N_samples=N_samples
            )
            # dists = torch.cat(
            #     (dist_from_cam[:, 1:] - dist_from_cam[:, :-1], torch.zeros_like(dist_from_cam[:, :1])),
            #     dim=-1,
            # )
            dists = torch.cat(
                (z_vals[:, 1:] - z_vals[:, :-1], torch.zeros_like(z_vals[:, :1])),
                dim=-1,
            )
            distances = z_vals * viewdirs_norm
            dists = dists * viewdirs_norm
        else:
            raise NotImplementedError
            xyz_sampled, z_vals, ray_valid = self.sample_ray(
                rays_chunk[:, :3], viewdirs, is_train=is_train, N_samples=N_samples
            )
            dists = torch.cat(
                (z_vals[:, 1:] - z_vals[:, :-1], torch.zeros_like(z_vals[:, :1])),
                dim=-1,
            )
        viewdirs = viewdirs / viewdirs_norm
        viewdirs = viewdirs.view(-1, 1, 3).expand(xyz_sampled.shape)

        sigma = torch.zeros(xyz_sampled.shape[:-1], device=xyz_sampled.device)
        rgb = torch.zeros((*xyz_sampled.shape[:2], 3), device=xyz_sampled.device)

        if ray_valid.any():
            xyz_sampled = self.normalize_coord(xyz_sampled)
            sigma_feature = self.compute_densityfeature(
                xyz_sampled[ray_valid],
            )

            validsigma = self.feature2density(sigma_feature)
            sigma[ray_valid] = validsigma

        alpha = 1.0 - torch.exp(-sigma * dists * self.distance_scale)
        # if is_train == False:
        #     first_large_alpha = torch.argmax((alpha > torch.quantile(alpha, 0.4, dim=-1, keepdim=True)).float(), dim=-1)
        #     alpha[torch.arange(alpha.shape[1], device=alpha.device)[None] < first_large_alpha[..., None]] = 0
            
        weight, T = alpha2weights(alpha)
        
        acc_map = torch.sum(weight, -1)
        depth_map = torch.sum(weight * z_vals, -1)

        # if not is_train:
        #     idx_map = torch.sum(weight * torch.arange(alpha.shape[1], device=alpha.device)[None], -1, keepdim=True)
        #     # first_large_alpha = torch.argmax((alpha > torch.quantile(alpha, 0.4, dim=-1, keepdim=True)).float(), dim=-1)
        #     # alpha[torch.arange(alpha.shape[1], device=alpha.device)[None] < first_large_alpha[..., None]] = 0
        #     alpha[torch.arange(alpha.shape[1], device=alpha.device)[None] < idx_map * 0.7] = 0
        #     weight, T = alpha2weights(alpha)
        
        app_mask = weight > self.rayMarch_weight_thres
        if app_mask.any():
            app_features = self.compute_appfeature(
                xyz_sampled[app_mask],
            )
            if self.shadingMode == "MLP_Fea_late_view_late_emb":
                valid_rgbs = self.renderModule(
                    xyz_sampled[app_mask], viewdirs[app_mask].detach(), app_features, app_embedings[:, None, :].repeat(1, xyz_sampled.shape[1], 1)[app_mask]
                )            
            else:
                valid_rgbs = self.renderModule(
                    xyz_sampled[app_mask], viewdirs[app_mask].detach(), app_features
                )
            rgb[app_mask] = valid_rgbs

        rgb_map = torch.sum(weight[..., None] * rgb, -2)
        if white_bg or (is_train and torch.rand((1,)) < 0.5):
            rgb_map = rgb_map + (1.0 - acc_map[..., None])
        rgb_map = rgb_map.clamp(0, 1)

        # Render depth with gamma to the alphas
        alpha_gamma = torch.nan_to_num((alpha + 1e-6) ** opasity_gamma)
        weight_gamma, T_gamma = alpha2weights(alpha_gamma)
        depth_map_gamma = torch.sum(weight_gamma * z_vals, -1)
        depth_map_gamma = depth_map_gamma + torch.clamp(1.0 - acc_map, 0, 1) * 1e3

        return rgb_map, depth_map, depth_map_gamma, distances, dists, alpha, T, weight, alpha_gamma, T_gamma, weight_gamma

        # with torch.no_grad():
        #     _, weight_for_decimation, _ = raw2alpha(sigma, dists * self.distance_scale)
        # bg_weight = T[:, -1:]
        # if do_render:
            # rgb_map, depth_map, weight = render_from_samples(
            #     sigma, rgb, z_vals, dists, white_bg, is_train, self.distance_scale
            # )
            # return rgb_map, depth_map, weight, z_vals * viewdirs_norm, dists
            #     # torch.tile(z_vals, (xyz_sampled.shape[0], 1)),
            # )  # , rgb, sigma, dists#, alpha, weight, bg_weight
        # else:
        #     return z_vals, rgb, sigma, dists

        # #################""
        # alpha = 1.0 - torch.exp(-sigma * dists * self.distance_scale)
        # alpha = alpha ** (1/2.2)
        # T = torch.cumprod(
        #     torch.cat(
        #         [torch.ones(alpha.shape[0], 1, device=alpha.device), 1.0 - alpha + 1e-10], -1
        #     ),
        #     -1,
        # )
        # weights = alpha * T[:, :-1]  # [N_rays, N_samples]
        # import matplotlib.pyplot as plt
        # y = weights[50]
        # plt.plot(np.arange(len(y)), y.detach().cpu().numpy(), label="new weights")
        # plt.legend(loc='upper center')
        # plt.savefig('test.png')
        # #################
