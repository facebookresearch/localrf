import json
import copy
import os
import shutil
import cv2
import numpy as np
from skimage.metrics import structural_similarity, peak_signal_noise_ratio

from localTensoRF.utils.utils import rgb_lpips, rgb_ssim

scenes = [
    # "intermediate/M60/skip_4",
    # "intermediate/Panther/skip_4",
    # "intermediate/Train/skip_4",
    # "advanced/Auditorium/skip_4",
    # "advanced/Ballroom/skip_4",
    # "advanced/Courtroom/skip_4",
    # "advanced/Museum/skip_4",
    # "train/Caterpillar/skip_4",
    # "train/Church/skip_4",
    # # "train/Courthouse/skip_4",
    # # "intermediate/Playground/skip_4",
    # # "train/Barn/skip_4",
    # # "train/Ignatius/skip_4",
    # # "train/Meetingroom/skip_4",
    # # "train/Truck/skip_4",

    # "ours/uw1/skip_0",
    "ours/uw2/skip_0",
    "ours/pg/skip_0",
    "ours/hike_07_08_gopro_4/skip_2",
    # "ours/hike_09_26_7/skip_0",
    "ours/hike_1008_2/skip_2", 
    "ours/hike_09_26_1/skip_0",
]

methods = [
    # "mipnerf360",
    # "npp",
    # "meganerf",
    # "barf_colmap",
    # "barf",
    # "scnerf_colmap",
    # "ours_colmap",
    # "ours_colmapopt",
    "ours_self",
    "ours_noprog",
    "ours_noloc",

]

for method in methods:
    all_framewise_psnrs = []
    all_framewise_ssim = []
    all_framewise_lpips = []

    msess = []
    for scene in scenes:
        with open(f"results/{method}/{scene}/metrics.json", "r") as f:
            results_dict = json.loads(f.read())

        framewise_psnrs = results_dict["all_framewise_psnrs"]
        framewise_ssim = results_dict["all_framewise_ssim"]
        framewise_lpips = results_dict["all_framewise_lpips"]

        all_framewise_psnrs += framewise_psnrs
        all_framewise_ssim += framewise_ssim
        all_framewise_lpips += framewise_lpips

    all_framewise_psnrs = np.stack(all_framewise_psnrs)
    all_framewise_mses = 10 ** (-all_framewise_psnrs / 10)
    all_framewise_ssim = np.stack(all_framewise_ssim)
    all_framewise_lpips = np.stack(all_framewise_lpips)

    avg_psnr = 10 * np.log10(1 / all_framewise_mses.mean())
    avg_ssim = 1 - np.sqrt(1 - all_framewise_ssim).mean() ** 2
    avg_lpips = all_framewise_lpips.mean()

    print(f"{method} & {avg_psnr:.2f} & {avg_ssim:.3f} & {avg_lpips:.3f} \\\\")
