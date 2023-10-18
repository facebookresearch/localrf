# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2022 Anpei Chen

import os
import random

import numpy as np
import torch
import cv2
import re

from joblib import delayed, Parallel
from torch.utils.data import Dataset
from utils.utils import decode_flow
import json

def concatenate_append(old, new, dim):
    new = np.concatenate(new, 0).reshape(-1, dim)
    if old is not None:
        new = np.concatenate([old, new], 0)

    return new

class LocalRFDataset(Dataset):
    def __init__(
        self,
        datadir,
        split="train",
        frames_chunk=20,
        downsampling=-1,
        load_depth=False,
        load_flow=False,
        with_preprocessed_poses=False,
        n_init_frames=7,
        subsequence=[0, -1],
        test_frame_every=10,
        frame_step=1,
    ):
        self.root_dir = datadir
        self.split = split
        self.frames_chunk = max(frames_chunk, n_init_frames)
        self.downsampling = downsampling
        self.load_depth = load_depth
        self.load_flow = load_flow
        self.frame_step = frame_step

        if with_preprocessed_poses:
            with open(os.path.join(self.root_dir, "transforms.json"), 'r') as f:
                self.transforms = json.load(f)
            self.image_paths = [os.path.basename(frame_meta["file_path"]) for frame_meta in self.transforms["frames"]]
            self.image_paths = sorted(self.image_paths)
            poses_dict = {os.path.basename(frame_meta["file_path"]): frame_meta["transform_matrix"] for frame_meta in self.transforms["frames"]}
            poses = []
            for idx, image_path in enumerate(self.image_paths):
                pose = np.array(poses_dict[image_path], dtype=np.float32)
                poses.append(pose)

            self.first_pose = np.array(poses_dict[self.image_paths[0]], dtype=np.float32)
            self.rel_poses = []
            for idx in range(len(poses)):
                if idx == 0:
                    pose = np.eye(4, dtype=np.float32)
                else:
                    pose = np.linalg.inv(poses[idx - 1]) @ poses[idx]
                self.rel_poses.append(pose)
            self.rel_poses = np.stack(self.rel_poses, axis=0) 

            self.pose_scale = 2e-2 / np.median(np.linalg.norm(self.rel_poses[:, :3, 3], axis=-1))
            self.rel_poses[:, :3, 3] *= self.pose_scale
            self.rel_poses = self.rel_poses[::frame_step]

        else:
            self.image_paths = sorted(os.listdir(os.path.join(self.root_dir, "images")))
        if subsequence != [0, -1]:
            self.image_paths = self.image_paths[subsequence[0]:subsequence[1]]

        self.image_paths = self.image_paths[::frame_step]
        self.all_image_paths = self.image_paths

        self.test_mask = []
        self.test_paths = []
        for idx, image_path in enumerate(self.image_paths):
            fbase = os.path.splitext(image_path)[0]
            index = int(fbase) if fbase.isnumeric() else idx
            if test_frame_every > 0 and index % test_frame_every == 0:
                self.test_paths.append(image_path)
                self.test_mask.append(1)
            else:
                self.test_mask.append(0)
        self.test_mask = np.array(self.test_mask)

        if split=="test":
            self.image_paths = self.test_paths
            self.frames_chunk = len(self.image_paths)
        self.num_images = len(self.image_paths)
        self.all_fbases = {os.path.splitext(image_path)[0]: idx for idx, image_path in enumerate(self.image_paths)}

        self.white_bg = False

        self.near_far = [0.1, 1e3] # Dummi
        self.scene_bbox = 2 * torch.tensor([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]])

        self.all_rgbs = None
        self.all_invdepths = None
        self.all_fwd_flow, self.all_fwd_mask, self.all_bwd_flow, self.all_bwd_mask = None, None, None, None
        self.all_loss_weights = None

        self.active_frames_bounds = [0, 0]
        self.loaded_frames = 0
        self.activate_frames(n_init_frames)


    def activate_frames(self, n_frames=1):
        self.active_frames_bounds[1] += n_frames
        self.active_frames_bounds[1] = min(
            self.active_frames_bounds[1], self.num_images
        )

        if self.active_frames_bounds[1] > self.loaded_frames:
            self.read_meta()



    def has_left_frames(self):
        return self.active_frames_bounds[1] < self.num_images

    def deactivate_frames(self, first_frame):
        n_frames = first_frame - self.active_frames_bounds[0]
        self.active_frames_bounds[0] = first_frame

        self.all_rgbs = self.all_rgbs[n_frames * self.n_px_per_frame:] 
        if self.load_depth:
            self.all_invdepths = self.all_invdepths[n_frames * self.n_px_per_frame:]
        if self.load_flow:
            self.all_fwd_flow = self.all_fwd_flow[n_frames * self.n_px_per_frame:]
            self.all_fwd_mask = self.all_fwd_mask[n_frames * self.n_px_per_frame:]
            self.all_bwd_flow = self.all_bwd_flow[n_frames * self.n_px_per_frame:]
            self.all_bwd_mask = self.all_bwd_mask[n_frames * self.n_px_per_frame:]
        self.all_loss_weights = self.all_loss_weights[n_frames * self.n_px_per_frame:]



    def read_meta(self):
        def read_image(i):
            image_path = os.path.join(self.root_dir, "images", self.image_paths[i])
            motion_mask_path = os.path.join(self.root_dir, "masks", 
                f"{os.path.splitext(self.image_paths[i])[0]}.png")
            if not os.path.isfile(motion_mask_path):
                motion_mask_path = os.path.join(self.root_dir, "masks/all.png")


            img = cv2.imread(image_path)[..., ::-1]
            img = img.astype(np.float32) / 255
            if self.downsampling != -1:
                scale = 1 / self.downsampling
                img = cv2.resize(img, None, 
                    fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

            if self.load_depth:
                invdepth_path = os.path.join(self.root_dir, "depth", 
                    f"{os.path.splitext(self.image_paths[i])[0]}.png")
                invdepth = cv2.imread(invdepth_path, -1).astype(np.float32)
                invdepth = cv2.resize(
                    invdepth, tuple(img.shape[1::-1]), interpolation=cv2.INTER_AREA)
            else:
                invdepth = None

            if self.load_flow:
                glob_idx = self.all_image_paths.index(self.image_paths[i])
                if glob_idx+1 < len(self.all_image_paths):
                    fwd_flow_path = self.all_image_paths[glob_idx+1]
                else:
                    fwd_flow_path = self.all_image_paths[0]
                if self.frame_step != 1:
                    fwd_flow_path = os.path.join(self.root_dir, "flow_ds", 
                        f"fwd_step{self.frame_step}_{os.path.splitext(fwd_flow_path)[0]}.png")
                    bwd_flow_path = os.path.join(self.root_dir, "flow_ds", 
                        f"bwd_step{self.frame_step}_{os.path.splitext(self.image_paths[i])[0]}.png")
                else:
                    fwd_flow_path = os.path.join(self.root_dir, "flow_ds", 
                        f"fwd_{os.path.splitext(fwd_flow_path)[0]}.png")
                    bwd_flow_path = os.path.join(self.root_dir, "flow_ds", 
                        f"bwd_{os.path.splitext(self.image_paths[i])[0]}.png")
                encoded_fwd_flow = cv2.imread(fwd_flow_path, cv2.IMREAD_UNCHANGED)
                encoded_bwd_flow = cv2.imread(bwd_flow_path, cv2.IMREAD_UNCHANGED)
                flow_scale = img.shape[0] / encoded_fwd_flow.shape[0] 
                encoded_fwd_flow = cv2.resize(
                    encoded_fwd_flow, tuple(img.shape[1::-1]), interpolation=cv2.INTER_AREA)
                encoded_bwd_flow = cv2.resize(
                    encoded_bwd_flow, tuple(img.shape[1::-1]), interpolation=cv2.INTER_AREA)            
                fwd_flow, fwd_mask = decode_flow(encoded_fwd_flow)
                bwd_flow, bwd_mask = decode_flow(encoded_bwd_flow)
                fwd_flow = fwd_flow * flow_scale
                bwd_flow = bwd_flow * flow_scale
            else:
                fwd_flow, fwd_mask, bwd_flow, bwd_mask = None, None, None, None

            if os.path.isfile(motion_mask_path):
                mask = cv2.imread(motion_mask_path, cv2.IMREAD_UNCHANGED)
                if len(mask.shape) != 2:
                    mask = mask[..., 0]
                mask = cv2.resize(mask, tuple(img.shape[1::-1]), interpolation=cv2.INTER_AREA) > 0
            else:
                mask = None

            return {
                "img": img, 
                "invdepth": invdepth,
                "fwd_flow": fwd_flow,
                "fwd_mask": fwd_mask,
                "bwd_flow": bwd_flow,
                "bwd_mask": bwd_mask,
                "mask": mask,
            }

        n_frames_to_load = min(self.frames_chunk, self.num_images - self.loaded_frames)
        all_data = Parallel(n_jobs=-1, backend="threading")(
            delayed(read_image)(i) for i in range(self.loaded_frames, self.loaded_frames + n_frames_to_load) 
        )
        self.loaded_frames += n_frames_to_load
        all_rgbs = [data["img"] for data in all_data]
        all_invdepths = [data["invdepth"] for data in all_data]
        all_fwd_flow = [data["fwd_flow"] for data in all_data]
        all_fwd_mask = [data["fwd_mask"] for data in all_data]
        all_bwd_flow = [data["bwd_flow"] for data in all_data]
        all_bwd_mask = [data["bwd_mask"] for data in all_data]
        all_mask = [data["mask"] for data in all_data]

        all_laplacian = [
                np.ones_like(img[..., 0]) * cv2.Laplacian(
                            cv2.cvtColor((img*255).astype(np.uint8), cv2.COLOR_RGB2GRAY), cv2.CV_32F
                        ).var()
            for img in all_rgbs
        ]
        all_loss_weights = [laplacian if mask is None else laplacian * mask for laplacian, mask in zip(all_laplacian, all_mask)]

        self.img_wh = list(all_rgbs[0].shape[1::-1])
        self.n_px_per_frame = self.img_wh[0] * self.img_wh[1]

        if self.split != "train":
            self.all_rgbs = np.stack(all_rgbs, 0)
            if self.load_depth:
                self.all_invdepths = np.stack(all_invdepths, 0)
            if self.load_flow:
                self.all_fwd_flow = np.stack(all_fwd_flow, 0)
                self.all_fwd_mask = np.stack(all_fwd_mask, 0)
                self.all_bwd_flow = np.stack(all_bwd_flow, 0)
                self.all_bwd_mask = np.stack(all_bwd_mask, 0)
        else:
            self.all_rgbs = concatenate_append(self.all_rgbs, all_rgbs, 3)
            if self.load_depth:
                self.all_invdepths = concatenate_append(self.all_invdepths, all_invdepths, 1)
            if self.load_flow:
                self.all_fwd_flow = concatenate_append(self.all_fwd_flow, all_fwd_flow, 2)
                self.all_fwd_mask = concatenate_append(self.all_fwd_mask, all_fwd_mask, 1)
                self.all_bwd_flow = concatenate_append(self.all_bwd_flow, all_bwd_flow, 2)
                self.all_bwd_mask = concatenate_append(self.all_bwd_mask, all_bwd_mask, 1)
            self.all_loss_weights = concatenate_append(self.all_loss_weights, all_loss_weights, 1)


    def __len__(self):
        return int(1e10)

    def __getitem__(self, i):
        raise NotImplementedError
        idx = np.random.randint(self.sampling_bound[0], self.sampling_bound[1])

        return {"rgbs": self.all_rgbs[idx], "idx": idx}

    def get_frame_fbase(self, view_id):
        return list(self.all_fbases.keys())[view_id]

    def sample(self, batch_size, is_refining, optimize_poses, n_views=16):
        active_test_mask = self.test_mask[self.active_frames_bounds[0] : self.active_frames_bounds[1]]
        test_ratio = active_test_mask.mean()
        if optimize_poses:
            train_test_poses = test_ratio > random.uniform(0, 1)
        else:
            train_test_poses = False

        inclusion_mask = active_test_mask if train_test_poses else 1 - active_test_mask
        sample_map = np.arange(
            self.active_frames_bounds[0], 
            self.active_frames_bounds[1], 
            dtype=np.int64)[inclusion_mask == 1]
        
        raw_samples = np.random.randint(0, inclusion_mask.sum(), n_views, dtype=np.int64)

        # Force having the last views during coarse optimization
        if not is_refining and inclusion_mask.sum() > 4:
            raw_samples[:2] = inclusion_mask.sum() - 1
            raw_samples[2:4] = inclusion_mask.sum() - 2
            raw_samples[4] = inclusion_mask.sum() - 3
            raw_samples[5] = inclusion_mask.sum() - 4

        view_ids = sample_map[raw_samples]

        idx = np.random.randint(0, self.n_px_per_frame, batch_size, dtype=np.int64)
        idx = idx.reshape(n_views, -1)
        idx = idx + view_ids[..., None] * self.n_px_per_frame
        idx = idx.reshape(-1)

        idx_sample = idx - self.active_frames_bounds[0] * self.n_px_per_frame

        return {
            "rgbs": self.all_rgbs[idx_sample], 
            "loss_weights": self.all_loss_weights[idx_sample], 
            "invdepths": self.all_invdepths[idx_sample] if self.load_depth else None,
            "fwd_flow": self.all_fwd_flow[idx_sample] if self.load_flow else None,
            "fwd_mask": self.all_fwd_mask[idx_sample] if self.load_flow else None,
            "bwd_flow": self.all_bwd_flow[idx_sample] if self.load_flow else None,
            "bwd_mask": self.all_bwd_mask[idx_sample] if self.load_flow else None,
            "idx": idx,
            "view_ids": view_ids,
            "train_test_poses": train_test_poses,
        }