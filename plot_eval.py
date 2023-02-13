import json
import numpy as np

import matplotlib.pyplot as plt

scenes = [
    # "intermediate/M60/skip_4",
    # "intermediate/Panther/skip_4",
    # "intermediate/Playground/skip_4",
    # "intermediate/Train/skip_4",
    # "advanced/Auditorium/skip_4",
    # "advanced/Ballroom/skip_4",
    # "advanced/Courtroom/skip_4",
    # "advanced/Museum/skip_4",
    # "train/Barn/skip_4",
    # "train/Caterpillar/skip_4",
    # "train/Church/skip_4",
    # "train/Courthouse/skip_4",
    # "train/Ignatius/skip_4",
    # "train/Meetingroom/skip_4",
    # "train/Truck/skip_4",

    # "ours/uw1/skip_0",
    "sequenced/ours/uw2/skip_0",
    "sequenced/ours/pg/skip_0",
    "sequenced/ours/hike_07_08_gopro_4/skip_2",
    # "ours/hike_09_26_7/skip_0",
    "sequenced/ours/hike_1008_2/skip_2", 
    "sequenced/ours/hike_09_26_1/skip_0",
]

methods = [
    "mipnerf360",
    "nerfacto",
    "barf",
    # "ours_colmap",
    "ours_self",
]

method_labels = [
    "Mip-NeRF360",
    "Nerfacto",
    "BARF",
    # "Ours + colmap",
    "Ours",
]

colors = [
    "C3",
    "C0",
    "C2",
    "C4",
]

all_methods_framewise_psnrs = {}
all_methods_framewise_ssim = {}
all_methods_framewise_lpips = {}

for method in methods:
    all_framewise_psnrs = []
    all_framewise_ssim = []
    all_framewise_lpips = []

    msess = []
    for scene in scenes:
        with open(f"results/{method}/{scene}/metrics.json", "r") as f:
            results_dict = json.loads(f.read())

        framewise_psnrs = results_dict["all_framewise_psnrs"][:107]
        framewise_ssim = results_dict["all_framewise_ssim"][:107]
        framewise_lpips = results_dict["all_framewise_lpips"][:107]

        all_framewise_psnrs += framewise_psnrs
        all_framewise_ssim += framewise_ssim
        all_framewise_lpips += framewise_lpips

    all_framewise_psnrs = np.stack(all_framewise_psnrs)
    all_framewise_ssim = np.stack(all_framewise_ssim)
    all_framewise_lpips = np.stack(all_framewise_lpips)

    all_framewise_psnrs = np.sort(all_framewise_psnrs)[::-1]
    all_framewise_ssim = np.sort(all_framewise_ssim)[::-1]
    all_framewise_lpips = np.sort(all_framewise_lpips)

    all_methods_framewise_psnrs[method] = all_framewise_psnrs
    all_methods_framewise_ssim[method] = all_framewise_ssim
    all_methods_framewise_lpips[method] = all_framewise_lpips

def plot_sorted(per_method_framewise_metrics, metric_name):
    plt.figure(figsize=(6, 3))
    for color, method, method_label in zip(colors, methods, method_labels):
        framewise_metrics = per_method_framewise_metrics[method]
        x = np.arange(len(framewise_metrics))

        if method_label == "Ours":
            plt.plot(x, framewise_metrics, label=method_label, linewidth=3.5, color=color)
        else:
            plt.plot(x, framewise_metrics, label=method_label, linewidth=2.5, color=color)
        # plt.xticks([])

    if metric_name == "LPIPS":
        plt.legend(loc='lower right', shadow=False, fontsize='x-large')   
    else:
        plt.legend(loc='upper right', shadow=False, fontsize='x-large')   
    plt.savefig(f"results/{metric_name}.pdf")
    plt.savefig(f"results/{metric_name}.png")

plot_sorted(all_methods_framewise_psnrs, "PSNR")
plot_sorted(all_methods_framewise_ssim, "SSIM")
plot_sorted(all_methods_framewise_lpips, "LPIPS")

print(len(all_methods_framewise_psnrs["mipnerf360"]) / len(all_methods_framewise_psnrs["barf"]))