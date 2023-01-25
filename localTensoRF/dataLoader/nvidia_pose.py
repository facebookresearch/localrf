from genericpath import isfile
import os
import random

import numpy as np
import torch
import cv2
import time
import re

from joblib import delayed, Parallel
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from utils.utils import decode_flow
import json

def concatenate_append(old, new, dim):
    new = np.concatenate(new, 0).reshape(-1, dim)
    if old is not None:
        new = np.concatenate([old, new], 0)

    return new

def read_pfm(path):
    """Read pfm file.

    Args:
        path (str): path to file

    Returns:
        tuple: (data, scale)
    """
    with open(path, "rb") as file:

        color = None
        width = None
        height = None
        scale = None
        endian = None

        header = file.readline().rstrip()
        if header.decode("ascii") == "PF":
            color = True
        elif header.decode("ascii") == "Pf":
            color = False
        else:
            raise Exception("Not a PFM file: " + path)

        dim_match = re.match(r"^(\d+)\s(\d+)\s$", file.readline().decode("ascii"))
        if dim_match:
            width, height = list(map(int, dim_match.groups()))
        else:
            raise Exception("Malformed PFM header.")

        scale = float(file.readline().decode("ascii").rstrip())
        if scale < 0:
            # little-endian
            endian = "<"
            scale = -scale
        else:
            # big-endian
            endian = ">"

        data = np.fromfile(file, endian + "f")
        shape = (height, width, 3) if color else (height, width)

        data = np.reshape(data, shape)
        data = np.flipud(data)

        return data, scale

def sample_excluding(low, high, n_samples, inclusion_mask):
    sample_map = np.arange(low, high, dtype=np.int64)[inclusion_mask == 1]
    raw_samples = np.random.randint(0, inclusion_mask.sum(), n_samples, dtype=np.int64)
    return sample_map[raw_samples]

class NvidiaPoseDataset(Dataset):
    def __init__(
        self,
        datadir,
        split="train",
        frames_chunk=20,
        resize_h=-1,
        load_depth=False,
        load_flow=False,
        with_GT_poses=False,
        ray_type="ndc",
        n_init_frames=7,
        subsequence=[0, -1],
        test_skip=10
    ):
        """
        spheric_poses: whether the images are taken in a spheric inward-facing manner
                       default: False (forward-facing)
        val_num: number of val images (used for multigpu training, validate same image for all gpus)
        """

        self.test_skip = test_skip
        self.root_dir = datadir
        self.split = split
        self.frames_chunk = max(frames_chunk, n_init_frames)
        self.resize_h = resize_h
        self.load_depth = load_depth
        self.load_flow = load_flow

        if with_GT_poses:
            with open(os.path.join(self.root_dir, "transforms.json"), 'r') as f:
                self.transforms = json.load(f)
            self.image_paths = [os.path.basename(frame_meta["file_path"]) for frame_meta in self.transforms["frames"]]
            self.image_paths = sorted(self.image_paths)
            poses_dict = {os.path.basename(frame_meta["file_path"]): frame_meta["transform_matrix"] for frame_meta in self.transforms["frames"]}
            poses = []
            for idx, image_path in enumerate(self.image_paths):
                pose = np.array(poses_dict[image_path], dtype=np.float32)
                poses.append(pose)

            self.rel_poses = []
            for idx in range(len(poses)):
                if idx == 0:
                    pose = np.eye(4, dtype=np.float32)
                    # pose = poses[idx].copy() # TODO f
                else:
                    pose = np.linalg.inv(poses[idx - 1]) @ poses[idx]
                self.rel_poses.append(pose)
            self.rel_poses = np.stack(self.rel_poses, axis=0) 

            # TODO f
            # scale = 0.33
            scale = 2e-2 / np.median(np.linalg.norm(self.rel_poses[:, :3, 3], axis=-1))
            self.rel_poses[:, :3, 3] *= scale

        else:
            self.image_paths = sorted(os.listdir(os.path.join(self.root_dir, "images")))
        if subsequence != [0, -1]:
            self.image_paths = self.image_paths[subsequence[0]:subsequence[1]]
        
        self.test_mask = []
        self.test_paths = []
        for idx, image_path in enumerate(self.image_paths):
            fbase = os.path.splitext(image_path)[0]
            index = int(fbase) if fbase.isnumeric() else idx
            if index % test_skip == 0:
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

        if ray_type == "contract":
            self.near_far = [0, 2]
            self.scene_bbox = 2 * torch.tensor([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]])
        else:
            self.near_far = [0.0, 1.0]
            self.scene_bbox = torch.tensor([[-1.5, -1.67, -1.0], [1.5, 1.67, 1.0]])

        self.all_rgbs = None
        self.all_invdepths = None
        self.all_fwd_flow, self.all_fwd_mask, self.all_bwd_flow, self.all_bwd_mask = None, None, None, None
        self.laplacian, self.all_motion_mask = None, None

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
        self.laplacian = self.laplacian[n_frames * self.n_px_per_frame:]
        self.all_motion_mask = self.all_motion_mask[n_frames * self.n_px_per_frame:]



    def read_meta(self):
        def read_image(i):
            image_path = os.path.join(self.root_dir, "images", self.image_paths[i])
            motion_mask_path = os.path.join(self.root_dir, "motion_masks", 
                f"{os.path.splitext(self.image_paths[i])[0][1:]}.png")

            img = cv2.imread(image_path)[..., ::-1]
            img = img.astype(np.float32) / 255
            if self.resize_h != -1:
                scale = self.resize_h / img.shape[0]
                img = cv2.resize(img, None, 
                    fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

            if self.load_depth:
                # invdepth_path = os.path.join(self.root_dir, "depth", 
                #     f"{os.path.splitext(self.image_paths[i])[0]}.png")
                # invdepth = cv2.imread(invdepth_path, cv2.IMREAD_UNCHANGED)
                # invdepth = invdepth.astype(np.float32) / invdepth.max()
                invdepth_path = os.path.join(self.root_dir, "depth", 
                    f"{os.path.splitext(self.image_paths[i])[0]}.pfm")
                invdepth, _ = read_pfm(invdepth_path)
                invdepth = cv2.resize(
                    invdepth, tuple(img.shape[1::-1]), interpolation=cv2.INTER_AREA)
            else:
                invdepth = None

            if self.load_flow:
                fwd_flow_path = os.path.join(self.root_dir, "flow_ds", 
                    f"fwd_{os.path.splitext(self.image_paths[i])[0]}.png")
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
                motion_mask = cv2.imread(motion_mask_path, cv2.IMREAD_UNCHANGED)
                if len(motion_mask.shape) != 2:
                    motion_mask = motion_mask[..., 0]
                motion_mask = cv2.dilate(motion_mask, np.ones([5, 5]))
                motion_mask = cv2.resize(motion_mask, tuple(img.shape[1::-1]), interpolation=cv2.INTER_AREA) > 0
            else:
                motion_mask = np.zeros_like(img[..., 0])

            return {
                "img": img, 
                "invdepth": invdepth,
                "fwd_flow": fwd_flow,
                "fwd_mask": fwd_mask,
                "bwd_flow": bwd_flow,
                "bwd_mask": bwd_mask,
                "motion_mask": motion_mask,
            }

        n_frames_to_load = min(self.frames_chunk, self.num_images - self.loaded_frames)
        # self.all_rgbs = [read_image(i) for i in range(self.loaded_frames, self.loaded_frames + n_frames_to_load)]
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
        all_motion_mask = [data["motion_mask"] for data in all_data]
        # Remove the forward flow offset due to preprocessing pipeline
        all_fwd_flow = all_fwd_flow[1:] + [all_fwd_flow[0]]
        all_fwd_mask = all_fwd_mask[1:] + [all_fwd_mask[0]]
        laplacian = [
                np.ones_like(img[..., 0]) * np.sqrt(cv2.Laplacian(img[0], -1).var()) 
            for img in all_rgbs
        ]

        self.img_wh = list(all_rgbs[0].shape[1::-1])
        self.n_px_per_frame = self.img_wh[0] * self.img_wh[1]

        if self.split != "train":
            self.all_rgbs = np.stack(all_rgbs, 0)
        else:
            self.all_rgbs = concatenate_append(self.all_rgbs, all_rgbs, 3)
            if self.load_depth:
                self.all_invdepths = concatenate_append(self.all_invdepths, all_invdepths, 1)
            if self.load_flow:
                self.all_fwd_flow = concatenate_append(self.all_fwd_flow, all_fwd_flow, 2)
                self.all_fwd_mask = concatenate_append(self.all_fwd_mask, all_fwd_mask, 1)
                self.all_bwd_flow = concatenate_append(self.all_bwd_flow, all_bwd_flow, 2)
                self.all_bwd_mask = concatenate_append(self.all_bwd_mask, all_bwd_mask, 1)
            self.laplacian = concatenate_append(self.laplacian, laplacian, 1)
            self.all_motion_mask = concatenate_append(self.all_motion_mask, all_motion_mask, 1)


    def __len__(self):
        return int(1e10)

    def __getitem__(self, i):
        raise NotImplementedError
        idx = np.random.randint(self.sampling_bound[0], self.sampling_bound[1])

        return {"rgbs": self.all_rgbs[idx], "idx": idx}

    def get_frame_fbase(self, view_id):
        return list(self.all_fbases.keys())[view_id]

    # TODO: Remove
    def get_gt_frame(self, view_ids):
        W, H = self.img_wh
        idx = np.arange(W * H, dtype=np.int64)
        idx = idx + view_ids * self.n_px_per_frame

        idx_sample = idx - self.active_frames_bounds[0] * self.n_px_per_frame
        idx_sample[idx_sample < 0] = 0

        if self.load_flow:
            fwd_mask = self.all_fwd_mask[idx_sample].reshape(len(view_ids), -1)
            fwd_mask[view_ids==self.active_frames_bounds[1] - 1, ...] = 0
            fwd_mask = fwd_mask.reshape(-1, 1)
        else:
            fwd_mask = None


        frame = {
            "rgbs": self.all_rgbs[idx_sample], 
            "laplacian": self.laplacian[idx_sample], 
            "invdepths": self.all_invdepths[idx_sample] if self.load_depth else None,
            "fwd_flow": self.all_fwd_flow[idx_sample] if self.load_flow else None,
            "fwd_mask": fwd_mask,
            "bwd_flow": self.all_bwd_flow[idx_sample] if self.load_flow else None,
            "bwd_mask": self.all_bwd_mask[idx_sample] if self.load_flow else None,
        }
        for key in frame:
            if frame[key] is not None:
                frame[key] = frame[key].reshape(H, W, -1)

        return frame

    def sample(self, batch_size, n_views=16):
        # start = self.active_frames_bounds[0] * self.n_px_per_frame
        # stop = self.active_frames_bounds[1] * self.n_px_per_frame

        active_test_mask = self.test_mask[self.active_frames_bounds[0] : self.active_frames_bounds[1]]
        test_ratio = active_test_mask.mean()
        # n_test_views = int(np.ceil(n_views * test_ratio))
        train_test_poses = test_ratio > random.uniform(0, 1)


        # if n_test_views != 0:
        #     view_ids = np.zeros(n_views, dtype=np.int64)
        #     view_ids[:-n_test_views] = sample_excluding(
        #         self.active_frames_bounds[0],
        #         self.active_frames_bounds[1],
        #         n_views - n_test_views,
        #         1 - active_test_mask)
        #     view_ids[-n_test_views:] = sample_excluding(
        #         self.active_frames_bounds[0],
        #         self.active_frames_bounds[1],
        #         n_test_views,
        #         active_test_mask)
        if test_ratio != 0:
            view_ids = sample_excluding(
                self.active_frames_bounds[0],
                self.active_frames_bounds[1],
                n_views,
                active_test_mask if train_test_poses else 1 - active_test_mask,
            )
        else:
            view_ids = np.random.randint(
                self.active_frames_bounds[0],
                self.active_frames_bounds[1],
                n_views, dtype=np.int64
            )

        idx = np.random.randint(0, self.n_px_per_frame, batch_size, dtype=np.int64)
        idx = idx.reshape(n_views, -1)
        idx = idx + view_ids[..., None] * self.n_px_per_frame
        idx = idx.reshape(-1)

        idx_sample = idx - self.active_frames_bounds[0] * self.n_px_per_frame

        return {
            "rgbs": self.all_rgbs[idx_sample], 
            "laplacian": self.laplacian[idx_sample], 
            "invdepths": self.all_invdepths[idx_sample] if self.load_depth else None,
            "fwd_flow": self.all_fwd_flow[idx_sample] if self.load_flow else None,
            "fwd_mask": self.all_fwd_mask[idx_sample] if self.load_flow else None,
            "bwd_flow": self.all_bwd_flow[idx_sample] if self.load_flow else None,
            "bwd_mask": self.all_bwd_mask[idx_sample] if self.load_flow else None,
            "motion_mask": self.all_motion_mask[idx_sample],
            "idx": idx,
            "view_ids": view_ids,
            "train_test_poses": train_test_poses,
            # "n_test_frames": n_test_views,
        }