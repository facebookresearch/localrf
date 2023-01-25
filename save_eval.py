import json
import copy
import os
import shutil
import cv2
import numpy as np
from skimage.metrics import structural_similarity, peak_signal_noise_ratio

from localTensoRF.utils.utils import rgb_lpips, rgb_ssim

scenes = [
    "ours/uw1/skip_0",
    "ours/uw2/skip_0",
    "ours/pg/skip_0",
    "ours/hike_07_08_gopro_4/skip_2",
    "ours/hike_09_26_7/skip_0",
    "ours/hike_1008_2/skip_2", 
    "ours/hike_09_26_1/skip_0",

    # "intermediate/M60/skip_4",
    # "intermediate/Panther/skip_4",
    # "intermediate/Train/skip_4",
    # "advanced/Auditorium/skip_4",
    # "advanced/Ballroom/skip_4",
    # "advanced/Courtroom/skip_4",
    # "advanced/Museum/skip_4",
    # "train/Caterpillar/skip_4",
    # "train/Church/skip_4",
    # # # "train/Courthouse/skip_4",
    # # # "intermediate/Playground/skip_4",
    # # # "train/Barn/skip_4",
    # # # "train/Ignatius/skip_4",
    # # # "train/Meetingroom/skip_4",
    # # # "train/Truck/skip_4",
]

methods = [
    # "npp",
    # "barf_colmap",
    # "meganerf",
    # "scnerf_colmap",
    # "ours_colmap",
    "ours_self2",
    # "ours_noprog",
    # "ours_noloc",
    # "ours_colmapopt",
    # "mipnerf360",
    # "barf",
]

colmap_methods = [
    "mipnerf360",
    "npp",
    "barf_colmap",
    "meganerf",
    "scnerf_colmap",
    "ours_colmap",
    "ours_colmapopt",
]

for method in methods:
    # all_framewise_mses = []
    # all_framewise_psnrs = []
    # all_framewise_ssim = []
    # all_framewise_lpips = []

    # all_framewise_psnrs_dict = {}
    # all_framewise_ssim_dict = {}
    # all_framewise_lpips_dict = {}

    # scenewise_psnrs = {}
    # scenewise_ssim = {}
    # scenewise_lpips = {}

    msess = []
    for scene in scenes:
        print(f"****** Processing {scene} for {method}")
        gt_dir = f"data/sequenced/{scene}"
        res_dir = f"data/logs_eval/{method}"

        all_gt_paths = sorted(os.listdir(f"{gt_dir}/images"))
        all_gt_paths = [gt_path for gt_path in all_gt_paths if (int(os.path.splitext(gt_path)[0])%10) == 0]

        if method in colmap_methods:
            with open(f"{gt_dir}/transforms.json", 'r') as f:
                in_json = json.loads(f.read())
            gt_paths = [os.path.basename(frame['file_path']) for frame in in_json["frames"]]
            gt_paths = [gt_path for gt_path in gt_paths if (int(os.path.splitext(gt_path)[0])%10) == 0]
            gt_paths = sorted(gt_paths)
        else:
            gt_paths = all_gt_paths

        framewise_mses = []
        framewise_psnrs = []
        framewise_ssim = []
        framewise_lpips = []
        for idx, gt_path in enumerate(gt_paths):
            gt_rgb = cv2.imread(f"{gt_dir}/images/{gt_path}").astype(np.float32)[..., ::-1] / 255

            if method == "mipnerf360":
                rgb = cv2.imread(f"{res_dir}/{scene}/render/test_preds_step_250000/color_{idx:03d}.png")
            elif method == "npp":
                rgb = cv2.imread(f"{res_dir}/{scene}/render_test_250000/{gt_path}")
            elif "barf" in method:
                rgb = cv2.imread(f"{res_dir}/None/{scene}/test_view/rgb_{idx}.png")
            elif method == "meganerf":
                rgb = cv2.imread(f"{res_dir}/{scene}/val/outputs/{gt_path}")
            elif method == "scnerf_colmap":
                rgb = cv2.imread(f"{res_dir}/{scene}/render_test/{gt_path}")
            elif method == "ours_self":
                rgb = cv2.imread(f"data/logs/selffast/{scene}/rgb_maps/{os.path.splitext(gt_path)[0]}.png")
                if rgb is None:
                    rgb = cv2.imread(f"data/logs/selffast/{scene}/rgb_maps/{gt_path}")
            elif method == "ours_self2":
                rgb = cv2.imread(f"data/logs/self2/{scene}/rgb_maps/{os.path.splitext(gt_path)[0]}.png")
                if rgb is None:
                    rgb = cv2.imread(f"data/logs/self2/{scene}/rgb_maps/{gt_path}")
            elif method == "ours_colmap":
                rgb = cv2.imread(f"data/logs/colmapfast/{scene}/rgb_maps/{os.path.splitext(gt_path)[0]}.png")
                if rgb is None:
                    rgb = cv2.imread(f"data/logs/colmapfast/{scene}/rgb_maps/{gt_path}")
            elif method == "ours_colmapopt":
                rgb = cv2.imread(f"data/logs/colmapoptfast/{scene}/rgb_maps/{os.path.splitext(gt_path)[0]}.png")
                if rgb is None:
                    rgb = cv2.imread(f"data/logs/colmapoptfast/{scene}/rgb_maps/{gt_path}")
            elif method == "ours_noprog":
                rgb = cv2.imread(f"data/logs/noprogfast/{scene}/rgb_maps/{os.path.splitext(gt_path)[0]}.png")
                if rgb is None:
                    rgb = cv2.imread(f"data/logs/noprogfast/{scene}/rgb_maps/{gt_path}")
            elif method == "ours_noloc":
                rgb = cv2.imread(f"data/logs/nolocfast/{scene}/rgb_maps/{os.path.splitext(gt_path)[0]}.png")
                if rgb is None:
                    rgb = cv2.imread(f"data/logs/nolocfast/{scene}/rgb_maps/{gt_path}")


            if rgb is None:
                print(f"{method}: {scene}, {gt_path} not found")
            else:
                rgb = rgb.astype(np.float32)[..., ::-1] / 255
                if rgb.shape != gt_rgb.shape:
                    rgb = cv2.resize(rgb, gt_rgb.shape[1::-1])

                mse = ((rgb - gt_rgb)**2).mean()
                psnr = 10 * np.log10(1 / mse)
                ssim = structural_similarity(gt_rgb, rgb, multichannel=True, data_range=1)
                lpips = rgb_lpips(gt_rgb, rgb, "alex", "cuda:0")

                framewise_mses.append(mse)
                framewise_psnrs.append(psnr)
                framewise_ssim.append(ssim)
                framewise_lpips.append(lpips)

        framewise_mses = np.stack(framewise_mses)
        framewise_psnrs = np.stack(framewise_psnrs)
        framewise_ssim = np.stack(framewise_ssim)
        framewise_lpips = np.stack(framewise_lpips)

        scenewise_psnr = (10 * np.log10(1 / framewise_mses.mean()))
        scenewise_ssim = (1 - np.sqrt(1 - framewise_ssim).mean() ** 2)
        scenewise_lpips = (framewise_lpips.mean())

        results_dict = {
            "n_frames": len(all_gt_paths),
            "scenewise_psnr": scenewise_psnr,
            "scenewise_ssim": scenewise_ssim,
            "scenewise_lpips": scenewise_lpips,
            "all_framewise_psnrs": list(framewise_psnrs),
            "all_framewise_ssim": list(framewise_ssim.astype(np.float64)),
            "all_framewise_lpips": list(framewise_lpips),
        }


        os.makedirs(f"results/{method}/{scene}", exist_ok=True)
        with open(f"results/{method}/{scene}/metrics.json", "w") as f:
            json.dump(results_dict, f, indent=2)

        # all_framewise_mses += framewise_mses
        # all_framewise_psnrs += framewise_psnrs
        # all_framewise_ssim += framewise_ssim
        # all_framewise_lpips += framewise_lpips


        # all_framewise_psnrs_dict[scene] = framewise_psnrs
        # all_framewise_ssim_dict[scene] = framewise_ssim
        # all_framewise_lpips_dict[scene] = framewise_lpips


    # all_framewise_mses = np.stack(all_framewise_mses)
    # all_framewise_psnrs = np.stack(all_framewise_psnrs)
    # all_framewise_ssim = np.stack(all_framewise_ssim)
    # all_framewise_lpips = np.stack(all_framewise_lpips)

    # avg_psnr = 10 * np.log10(1 / all_framewise_mses.mean())
    # avg_ssim = 1 - np.sqrt(1 - all_framewise_ssim).mean() ** 2
    # avg_lpips = all_framewise_lpips.mean()

    # print(f"{method} & {avg_psnr:.2f} & {avg_ssim:.3f} & {avg_lpips:.3f} \\")

    # results_dict = {
    #     "avg_psnr": avg_psnr,
    #     "avg_ssim": avg_ssim,
    #     "avg_lpips": avg_lpips,
    #     "scenewise_psnrs": scenewise_psnrs,
    #     "scenewise_ssim": scenewise_ssim,
    #     "scenewise_lpips": scenewise_lpips,
    #     "all_framewise_psnrs": all_framewise_psnrs,
    #     "all_framewise_ssim": all_framewise_ssim,
    #     "all_framewise_lpips": all_framewise_lpips,
    # }

    # with open(f"results/{method}.json", "w") as f:
    #     json.dump(results_dict, f, indent=2)