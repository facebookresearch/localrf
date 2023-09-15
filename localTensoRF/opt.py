# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2022 Anpei Chen

import configargparse

def config_parser(cmd=None):
    parser = configargparse.ArgumentParser()
    parser.add_argument("--config", is_config_file=True, help="config file path")
    parser.add_argument(
        "--logdir", type=str, default="./log", help="where to store ckpts and logs"
    )
    parser.add_argument(
        "--datadir", type=str, default="./data/llff/fern", help="input data directory"
    )
    parser.add_argument(
        "--progress_refresh_rate",
        type=int,
        default=200,
        help="how many iterations to show iters",
    )

    parser.add_argument(
        "--downsampling", 
        type=float, 
        default=-1, 
        help="Downsampling ratio for training and testing. Test views rendered throughout optimization will be downsampled further two times."
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="TensorVMSplit",
        choices=["TensorVMSplit", "TensorCP", "TensorVMVt", "TensorMMt"],
    )

    # loader options
    parser.add_argument("--batch_size", type=int, default=4096)

    ## training options
    # learning rate
    parser.add_argument("--lr_R_init", type=float, default=5e-3, help="Rotation learning rate")
    parser.add_argument("--lr_t_init", type=float, default=5e-4, help="Translation learning rate")
    parser.add_argument("--lr_i_init", type=float, default=0, help="Intrinsics learning rate")
    parser.add_argument("--lr_exposure_init", type=float, default=1e-3, help="Exposure compensation learning rate")
    parser.add_argument("--lr_init", type=float, default=0.02, help="learning rate")
    parser.add_argument("--lr_basis", type=float, default=1e-3, help="learning rate")
    parser.add_argument(
        "--lr_decay_target_ratio",
        type=float,
        default=0.1,
        help="the target decay ratio; after decay_iters initial lr decays to lr*ratio",
    )
    parser.add_argument(
        "--lr_upsample_reset",
        type=int,
        default=1,
        help="reset lr to initial after upsampling",
    )

    # Basic scheduling options
    parser.add_argument("--N_voxel_init", type=int, default=64**3)
    parser.add_argument("--N_voxel_final", type=int, default=640**3)
    parser.add_argument("--n_iters_per_frame", type=int, default=600)
    parser.add_argument("--n_iters_reg", type=int, default=100)
    parser.add_argument(
        "--upsamp_list", 
        type=int, 
        default=[100, 150, 200, 250, 300],
        nargs='+')
    parser.add_argument("--update_AlphaMask_list", type=int, default=[100, 200, 300], nargs='+')
    parser.add_argument("--refinement_speedup_factor", type=float, default=1.0, 
                        help="Divides number of iterations in scheduling. Does not apply to progressive optimization.")

    # Progressive optimization scheduling options
    parser.add_argument(
        "--n_init_frames",
        type=int,
        default=5,
        help="Number of initial frames for the first RF optimization",
    )
    parser.add_argument(
        "--max_drift",
        type=float,
        default=1,
        help="Create a new RF once the camera pose shifts this amount w.r.t the last created RF",
    )
    parser.add_argument(
        "--n_max_frames",
        type=int,
        default=100,
        help="Maximum number of frames added before optimizing a new RF",
    )
    parser.add_argument(
        "--add_frames_every",
        type=int,
        default=100,
        help="Number of iterations before adding another frame",
    )
    parser.add_argument(
        "--n_overlap",
        type=int,
        default=30,
        help="Number of frames supervising two neighbour RFs",
    )
    parser.add_argument("--prog_speedup_factor", type=float, default=1.0, 
                        help="Divides number of iterations in progressive optimization scheduling. Multiplies pose lr.")

    # losses
    parser.add_argument("--loss_depth_weight_inital", type=float, default=0.1)
    parser.add_argument("--loss_flow_weight_inital", type=float, default=1)
    parser.add_argument("--L1_weight", type=float, default=1e-2, help="loss weight")
    parser.add_argument("--TV_weight_density", type=float, default=0.0, help="loss weight")#0.1
    parser.add_argument("--TV_weight_app", type=float, default=0.0, help="loss weight")#0.01

    # model
    # volume options
    parser.add_argument("--n_lamb_sigma", type=int, default=[8, 8, 8], action="append")
    parser.add_argument("--n_lamb_sh", type=int, default=[24, 24, 24], action="append")
    parser.add_argument("--data_dim_color", type=int, default=27)

    parser.add_argument(
        "--rm_weight_mask_thre",
        type=float,
        default=0.001,
        help="mask points in ray marching",
    )
    parser.add_argument(
        "--alpha_mask_thre",
        type=float,
        default=0.0001,
        help="threshold for creating alpha mask volume",
    )
    parser.add_argument(
        "--distance_scale",
        type=float,
        default=25,
        help="scaling sampling distance for computation",
    )
    parser.add_argument(
        "--density_shift",
        type=float,
        default=-5,
        help="shift density in softplus; making density = 0  when feature == 0",
    )

    # network decoder
    parser.add_argument(
        "--shadingMode", type=str, default="MLP_Fea_late_view", help="which shading mode to use"
    )
    parser.add_argument("--pos_pe", type=int, default=0, help="number of pe for pos")
    parser.add_argument("--view_pe", type=int, default=0, help="number of pe for view")
    parser.add_argument(
        "--fea_pe", type=int, default=0, help="number of pe for features"
    )
    parser.add_argument(
        "--featureC", type=int, default=128, help="hidden feature channel in MLP"
    )

    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="specific weights npy file to reload for coarse network",
    )
    parser.add_argument("--render_only", type=int, default=0)
    parser.add_argument("--render_test", type=int, default=1)
    parser.add_argument("--render_path", type=int, default=1)
    parser.add_argument("--render_from_file", type=str, default="", help="to load camera poses and render from them: https://github.com/facebookresearch/localrf/issues/20")

    ## ------ For saving RAM ------ ##
    # Set these flags to save your RAM. The final rendered images are still generated and saved.
    parser.add_argument("--skip_saving_video", action='store_true', help="If set, will not generate rendered video") # default False if not set, will be True if set.
    parser.add_argument("--skip_TB_images", action='store_true', help="If set, TensorBoard will not show the rendered images.") # default False if not set, will be True if set.


    # rendering options
    parser.add_argument("--fea2denseAct", type=str, default="softplus")
    parser.add_argument(
        "--nSamples",
        type=int,
        default=1e6,
        help="sample point each ray, pass 1e6 if automatic adjust",
    )
    parser.add_argument("--step_ratio", type=float, default=0.5)

    # Camera model options
    parser.add_argument("--fov", type=float, default=85.6, help="horizontal FoV in degree")
    parser.add_argument("--with_preprocessed_poses", type=int, default=0)

    parser.add_argument("--subsequence", default=[0, -1], type=int, nargs=2)
    parser.add_argument('--frame_step', type=int, default=1, help="Step between retained frames")
    parser.add_argument("--test_frame_every", default=10, type=int, help="Every test_frame_every-th frame is a test frame.")
    # logging/saving options
    parser.add_argument(
        "--vis_every", type=int, default=10000, help="Frequency of visualize the test images."
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    if cmd is not None:
        return parser.parse_args(cmd)
    else:
        return parser.parse_args()
