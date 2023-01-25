import configargparse


def config_parser(cmd=None):
    parser = configargparse.ArgumentParser()
    parser.add_argument("--config", is_config_file=True, help="config file path")
    parser.add_argument("--expname", type=str, help="experiment name")
    parser.add_argument(
        "--basedir", type=str, default="./log", help="where to store ckpts and logs"
    )
    parser.add_argument(
        "--datadir", type=str, default="./data/llff/fern", help="input data directory"
    )
    parser.add_argument(
        "--progress_refresh_rate",
        type=int,
        default=10,
        help="how many iterations to show psnrs or iters",
    )

    parser.add_argument("--with_depth", action="store_true")
    parser.add_argument("--resize_h", type=int, default=-1)

    parser.add_argument(
        "--model_name",
        type=str,
        default="TensorVMSplit",
        choices=["TensorVMSplit", "TensorCP", "TensorVMVt", "TensorMMt"],
    )

    # loader options
    parser.add_argument("--batch_size", type=int, default=8192)
    parser.add_argument("--n_iters", type=int, default=30000)

    parser.add_argument(
        "--dataset_name",
        type=str,
        default="blender",
        choices=[
            "nvidia_pose",
        ],
    )

    # training options
    # learning rate
    parser.add_argument("--lr_poses_init", type=float, default=1e-4, help="for poses")
    parser.add_argument("--lr_init", type=float, default=0.02, help="learning rate")
    parser.add_argument("--lr_basis", type=float, default=1e-3, help="learning rate")
    parser.add_argument(
        "--lr_decay_iters",
        type=int,
        default=-1,
        help="number of iterations the lr will decay to the target ratio; -1 will set it to n_iters",
    )
    parser.add_argument(
        "--lr_decay_target_ratio",
        type=float,
        default=0.1,
        help="the target decay ratio; after decay_iters inital lr decays to lr*ratio",
    )
    parser.add_argument(
        "--lr_upsample_reset",
        type=int,
        default=1,
        help="reset lr to inital after upsampling",
    )

    # loss
    parser.add_argument("--loss_sampleflow_weight_inital", type=float, default=0.0)
    parser.add_argument("--loss_depth_weight_inital", type=float, default=0.0)
    parser.add_argument("--loss_flow_weight_inital", type=float, default=0.0)
    parser.add_argument("--patch_size", type=int, default=0)
    parser.add_argument("--patch_batch_size", type=int, default=0)
    parser.add_argument("--patch_d_tv_weight", type=float, default=0.0)
    parser.add_argument("--L1_weight_inital", type=float, default=0.0, help="loss weight")
    parser.add_argument("--L1_weight_rest", type=float, default=0, help="loss weight")
    parser.add_argument("--Ortho_weight", type=float, default=0.0, help="loss weight")
    parser.add_argument("--TV_weight_density", type=float, default=0.0, help="loss weight")
    parser.add_argument("--TV_weight_app", type=float, default=0.0, help="loss weight")

    parser.add_argument("--pose_consistancy_loss", type=int, default=0)
    parser.add_argument("--distortion_loss_weight", type=float, default=0.0)

    # model
    # volume options
    parser.add_argument("--n_lamb_sigma", type=int, action="append")
    parser.add_argument("--n_lamb_sh", type=int, action="append")
    parser.add_argument("--data_dim_color", type=int, default=27)

    parser.add_argument(
        "--rm_weight_mask_thre",
        type=float,
        default=0.0001,
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
        default=-10,
        help="shift density in softplus; making density = 0  when feature == 0",
    )

    # network decoder
    parser.add_argument(
        "--shadingMode", type=str, default="MLP_PE", help="which shading mode to use"
    )
    parser.add_argument("--pos_pe", type=int, default=6, help="number of pe for pos")
    parser.add_argument("--view_pe", type=int, default=6, help="number of pe for view")
    parser.add_argument(
        "--fea_pe", type=int, default=6, help="number of pe for features"
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
    parser.add_argument("--render_test", type=int, default=0)
    parser.add_argument("--render_train", type=int, default=0)
    parser.add_argument("--render_path", type=int, default=0)
    parser.add_argument("--export_mesh", type=int, default=0)

    # rendering options
    parser.add_argument(
        "--lindisp",
        default=False,
        action="store_true",
        help="use disparity depth sampling",
    )
    parser.add_argument(
        "--perturb",
        type=float,
        default=1.0,
        help="set to 0. for no jitter, 1. for jitter",
    )
    parser.add_argument("--accumulate_decay", type=float, default=0.998)
    parser.add_argument("--fea2denseAct", type=str, default="softplus")
    parser.add_argument(
        "--ray_type",
        type=str,
        default="ndc",
        choices=["ndc", "contract"],
    )
    parser.add_argument(
        "--nSamples",
        type=int,
        default=1e6,
        help="sample point each ray, pass 1e6 if automatic adjust",
    )
    parser.add_argument("--step_ratio", type=float, default=0.5)

    # Rendering multiple RF options
    parser.add_argument("--expname_list", type=str, action="append")
    parser.add_argument("--render_multiple_rf", type=int, default=0)
    parser.add_argument(
        "--blending_mode", 
        type=str,
        default="image", 
        choices=["image", "samples_alphas", "samples_sigmas"]
    )

    # Progressive optimization
    parser.add_argument(
        "--n_init_frames",
        type=int,
        default=41,
        help="Number of initial frames for the first RF optimization",
    )
    parser.add_argument(
        "--max_drift",
        type=float,
        default=0.5,
        help="Create a new RF once the camera pose shifts this ammount w.r.t the last created RF",
    )
    parser.add_argument(
        "--n_max_frames",
        type=int,
        default=10,
        help="Maximum number of frames aded before optimizing a new RF",
    )
    parser.add_argument(
        "--add_frames_every",
        type=int,
        default=50,
        help="Number of iterations before adding another frame",
    )
    parser.add_argument(
        "--n_overlap",
        type=int,
        default=5,
        help="Number of frames supervising two neighbour RFs",
    )
    parser.add_argument(
        "--new_rf_init",
        type=int,
        default=0,
        help="Initialize newly created RF using the previous one",
    )
    parser.add_argument(
        "--add_frame_reset_rf_opt",
        type=int,
        default=0,
        help="Reset radiance field optimizers when adding a new frame",
    )
    parser.add_argument(
        "--n_max_training_rfs",
        type=int,
        default=2,
        help="Maximum number of radiance fields being optmized simultaneously",
    )
    parser.add_argument(
        "--n_it_rf_init",
        type=int,
        default=3000,
        help="",
    )

    # Camera model options
    parser.add_argument("--optimize_poses", type=int, default=0)
    parser.add_argument(
        "--pose_representation", type=str, default="6D", choices=["6D", "se3"]
    )
    parser.add_argument("--optimize_focal", type=int, default=0)
    parser.add_argument("--fov", type=float, default=85.6, help="horizontal FoV in degree")
    parser.add_argument("--use_foreground_mask", type=int, default=0)
    parser.add_argument("--with_GT_poses", type=int, default=0)


    parser.add_argument("--opasity_gamma", type=float, default=1.0)

    ## blender flags
    parser.add_argument(
        "--white_bkgd",
        action="store_true",
        help="set to render synthetic data on a white bkgd (always use for dvoxels)",
    )

    parser.add_argument("--N_voxel_init", type=int, default=100**3)
    parser.add_argument("--N_voxel_final", type=int, default=300**3)
    parser.add_argument("--upsamp_list", type=int, nargs='+')
    parser.add_argument("--update_AlphaMask_list", type=int, action="append")

    parser.add_argument("--idx_view", type=int, default=0)

    parser.add_argument("--subsequence", default=[0, -1], type=int, nargs=2)
    # logging/saving options
    parser.add_argument("--N_vis", type=int, default=5, help="N images to vis")
    parser.add_argument(
        "--vis_every", type=int, default=10000, help="frequency of visualize the image"
    )
    parser.add_argument(
        "--vis_train_every",
        type=int,
        default=2000,
        help="frequency of visualize the training on tensorboard",
    )
    parser.add_argument("--multiGPU", type=int, default=[0], nargs='+')
    if cmd is not None:
        return parser.parse_args(cmd)
    else:
        return parser.parse_args()
