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
        default=1000,
        help="how many iterations to show psnrs or iters",
    )

    parser.add_argument("--downsampling", type=int, default=-1)

    parser.add_argument(
        "--model_name",
        type=str,
        default="TensorVMSplit",
        choices=["TensorVMSplit", "TensorCP", "TensorVMVt", "TensorMMt"],
    )

    # loader options
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--n_iters_per_frame", type=int, default=600)
    parser.add_argument("--n_iters_reg", type=int, default=50)

    # training options
    # learning rate
    parser.add_argument("--lr_R_init", type=float, default=5e-3, help="Rotation learning rate")
    parser.add_argument("--lr_t_init", type=float, default=5e-4, help="Translation learning rate")
    parser.add_argument("--lr_i_init", type=float, default=1e-3, help="Intrinsics learning rate")
    parser.add_argument("--lr_exposure_init", type=float, default=5e-3, help="Per exposure compensation learning rate")
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
    parser.add_argument("--loss_depth_weight_inital", type=float, default=1e-1)
    parser.add_argument("--loss_flow_weight_inital", type=float, default=2)
    parser.add_argument("--L1_weight_inital", type=float, default=1e-3, help="loss weight")
    parser.add_argument("--L1_weight_rest", type=float, default=0, help="loss weight")
    parser.add_argument("--TV_weight_density", type=float, default=0.0, help="loss weight")#0.1
    parser.add_argument("--TV_weight_app", type=float, default=0.0, help="loss weight")#0.01

    # model
    # volume options
    parser.add_argument("--n_lamb_sigma", type=int, default=[8, 8, 8], action="append")
    parser.add_argument("--n_lamb_sh", type=int, default=[24, 24, 24], action="append")
    parser.add_argument("--data_dim_color", type=int, default=27)

    # parser.add_argument(
    #     "--rm_weight_mask_thre",
    #     type=float,
    #     default=0.0001,
    #     help="mask points in ray marching",
    # )
    parser.add_argument(
        "--alpha_mask_thre",
        type=float,
        default=0.001,
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
    parser.add_argument("--export_mesh", type=int, default=0)

    # rendering options
    parser.add_argument("--fea2denseAct", type=str, default="softplus")
    parser.add_argument(
        "--nSamples",
        type=int,
        default=1e6,
        help="sample point each ray, pass 1e6 if automatic adjust",
    )
    parser.add_argument("--step_ratio", type=float, default=0.5)

    # Progressive optimization
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
        help="Create a new RF once the camera pose shifts this ammount w.r.t the last created RF",
    )
    parser.add_argument(
        "--n_max_frames",
        type=int,
        default=200,
        help="Maximum number of frames aded before optimizing a new RF",
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

    # Camera model options
    parser.add_argument("--fov", type=float, default=85.6, help="horizontal FoV in degree")
    parser.add_argument("--with_GT_poses", type=int, default=0)

    parser.add_argument("--N_voxel_init", type=int, default=64**3)
    parser.add_argument("--N_voxel_final", type=int, default=640**3)
    parser.add_argument(
        "--upsamp_list", 
        type=int, 
        default=[100, 150, 200, 250, 300],
        nargs='+')
    parser.add_argument("--update_AlphaMask_list", type=int, default=[100, 200, 300], nargs='+')

    parser.add_argument("--subsequence", default=[0, -1], type=int, nargs=2)
    # logging/saving options
    parser.add_argument("--N_vis", type=int, default=5, help="N images to vis")
    parser.add_argument(
        "--vis_every", type=int, default=6000, help="frequency of visualize the image"
    )
    parser.add_argument("--multiGPU", type=int, default=[0], nargs='+')
    if cmd is not None:
        return parser.parse_args(cmd)
    else:
        return parser.parse_args()
