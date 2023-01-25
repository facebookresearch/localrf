import os
from itertools import product
from joblib import delayed, Parallel

website_path = "data/logs_supp"
ffmpeg = "ffmpeg"
ffmpeg = "~/ffmpeg/ffmpeg"
expnames = [
    "colmapfast", "self2", "colmap3",
    # "fhd_focalf90", "fhd_smooth", 
    # "fhd_focalf90", "fhd_offsets", "fhd_offsets_dist", "fhd_offsets_dist_hlr", "fhd_offsets_dist_uhlr"
    # "fhd_focalf", "fhd_nofocalf", "fhd_focalf80", "fhd_nofocalffov", "fhd_focalf90", "fhd_nofocalf90"
    # "fhd_coarsefine", "fhd_nodepth", "fhd_nodepth_g07d", "fhd_newopt", "fhd_focal", "fhd_focal2", "fhd_focal1", "fhd_focalbeta99", "fhd_focallowintrinlr"
    # "hd_default", "coarsefine", "coarsefinegamma08", "coarsefinegamma07", "fhd_coarsefine", "fhd_coarsefinegamma08"
    # "flow_depth_l1", "delaygamma1", "delaygamma07"
    # "delay_dist", "gamma05", "gamma07", "gamma07_double", "delay_l05", "delay_l05_adaptflow", "patch_tv_disp_e-2", "delay_l05_patchtv"
    # "test_opasity_gamma1", "test_opasity_gamma07-double", "test_opasity_gamma05",

    # "test_default", "test_pose_sche",

    # "test_default", "test_noup", "test_delay", "test_nonear", 
    # "test_adaptflow", "test_adaptflow_h", "test_lap", "test_lap_sig",
    # "test_l05", "test_lap_l05", "test_l05_e-3", "test_proper_delay",
    # "flow_depth_l1", "delay_l05_patchtv", "delay_l05", "delay_l05_adaptflow",

    # "default",
    # "flow",
    # "flow_depth",
    # "flow_depth_l1",
    # "flow_depth_l1_dist",
    # "init",
    # "init_flow",
    # "init_flow_depth",
    # "init_flow_depth_l1",
    # "init_flow_depth_l1_hlr",
    # "init_flow_depth_l1_hlr_04",
    # "init_flow_depth_l1_dist",

    # "flow_depth_l1_dist",
    # "depth_f_0.01",
    # "depth_rf_0.01",
    # "depth_rf_adapt_0.01",
    # "depth_rf2_0.01",
    # "hd512",
    # "hd3",
    # "hd_near0.1",
    # "fhd1024",
    # "fhd512",
    # "relu",
    # "relu_l1",
    # "softplus",
    # "softplus_l1",
    # "1.1",
    # "1",
    # "patch_tv_disp_e-2",
    # "softl1_1e-5",
    # "softl1_5e-4",
    # "add_logs",
    # "near0.1",
    # "l05",
    # "test",
    # "hd",
    # "softplus_l1",
    # "soft20",
    # "soft20_no_l1",
    # "relu_t05",
    # "relu_l1_1e-4_t05",
    # "relu_l1_5e-4_t05",
    # "softplus5_t05",
    # "softplus5_l1_5e-4_t05",
    # "softplus10_l1_1e-4_t05",
    # "softplus5_l1_1e-4_t05",
    # "softplus5_l1_1e-4_t05_pe2",
    # "softplus5_l1_1e-4_t05_pe1",
    # "softplus5_l1_1e-4_t05_pe0",
    # "softplus5_l1_1e-4_t05_pe3_no_view",
    # "softplus5_l1_1e-4_t05_pe0_no_view",
    # "softplus5_l1_1e-4_t05_no_init",
    # # "softplus5_tv_05_01_all_laploss_l11e-4",
    # # # "softplus5_tv_05_01_all_laploss_l11e-4f",
    # # "softplus5_tv_05_01_all_laploss_l11e-4f_03t",
    # # # "softplus5_tv_05_01_all_laploss_l11e-5",
    # # # "softplus5_tv_05_01_all_laploss_l11e-5f_03t",
    # # # "softplus5_tv_05_01_all_laploss_l15e-5",
    # # "softplus5_tv_05_01_all_laploss",
    # # "softplus5_tv_05_01_all",
    # # "relu_all_laploss",
    # # "relu_all_laploss_l11e-5",
]
# scenes = ["parkour/skip_0", "bear/skip_0", "office/skip_0", "office_gopro/skip_2", "hike_07_08_gopro_2/skip_0", "hike_07_08_gopro_3/skip_3", "hike_07_08_gopro_4/skip_3"]
scenes = [
    "ours/uw1/skip_0",
    "ours/uw2/skip_0",
    "ours/pg/skip_0",
    "ours/hike_07_08_gopro_4/skip_2",
    "ours/hike_1008_2/skip_2", 
    "ours/hike_09_26_1/skip_0",
]
render_paths = ["imgs_train_all", "smooth_spline", "smooth_spline1"]
render_paths = ["imgs_train_all", "smooth_spline1"]
prefixes = ["", "depth", "pose"]
prefixes = [""]
os.makedirs(website_path, exist_ok=True)


## Download data
# # for exp, scene, render_path, isdepth in product(expnames, scenes, render_paths, aredepth):
# #     os.system(
# #         f"scp -P 2302 docker@localhost:/mnt/uberlapse/ameuleman/data/logs/{exp}/{scene}/{render_path}/{isdepth}video.mp4 {website_path}/{exp}_{scene}_{render_path}{isdepth}.mp4")
Parallel(n_jobs=-1, backend="loky")(
        delayed(os.system)(
            f"scp -P 2302 docker@localhost:/mnt/uberlapse/ameuleman/data/logs/{exp}/{scene}/{render_path}/{prefix}video.mp4 {website_path}/{exp}_{scene.replace('/', '_')}_{render_path}{prefix}.mp4")
    for exp, scene, render_path, prefix in product(expnames, scenes, render_paths, prefixes))

scenes = [scene.replace('/', '_') for scene in scenes]

## Merge videos horizontally
for exp, scene in product(expnames, scenes):
    # os.system(f"cp {website_path}/{exp}_{scene}_{render_paths[0]}.mp4")
    n_stack = 0
    to_stack = ""
    for render_path in render_paths:
        if os.path.isfile(f"{website_path}/{exp}_{scene}_{render_path}.mp4"):
            to_stack += f"-i {website_path}/{exp}_{scene}_{render_path}.mp4 "
            n_stack += 1
    render_path = render_paths[0]
    for prefix in prefixes[1:]:
        if os.path.isfile(f"{website_path}/{exp}_{scene}_{render_path}{prefix}.mp4"):
            to_stack += f"-i {website_path}/{exp}_{scene}_{render_path}{prefix}.mp4 "
            n_stack += 1
    if n_stack > 0:
        os.system(f"{ffmpeg} -r 30 -y {to_stack} -filter_complex hstack=inputs={n_stack} {website_path}/stacked_{exp}_{scene}.mp4")

## Create the website
with open("visualization/website_base.html", "r") as f:
    website_html = f.read()
scene = scenes[0]
website_html = website_html.replace("DEFAULT_SCENE", scene)
for exp in expnames:
    website_html += f"""
    <div class="row">
        <video id="{exp}" height="200" autoplay loop muted>
        <source src="stack_{exp}_{scene}.mp4" type="video/mp4">
        </video>
        <p>{exp}</p>
    </div>
    """

swapping_code = "\n"
for exp in expnames:
    swapping_code += f"""
    var video = document.getElementById("{exp}");
    video.src = 'stacked_{exp}_'+sq[i]+'.mp4';
    video.play();\n"""
with open("visualization/js_base.html", "r") as f:
    js_code = f.read()
js_code = js_code.replace("SWAPPING_CODE", swapping_code)
js_code = js_code.replace("SCENES", str(scenes))
website_html += js_code
with open(f"{website_path}/website.html", "w") as f:
    f.write(website_html)
