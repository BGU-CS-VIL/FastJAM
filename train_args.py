import argparse

def get_argparser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--run_name', type=str, default=None, help='Run name')
    parser.add_argument('--data_folder', type=str, required=True,
                        help='Data folder path. In this directory should be images/ dir (and pck/ for evaluation)')
    parser.add_argument('--seed', type=int,
                        default=-1, help='Randomize Seed, if -1 means random seed')
    # ---------------------------------- Matcher settings ----------------------------------
    parser.add_argument('--roma_coarse_res', nargs=2, type=int,
                        default=(560, 560), help='coarse resolution for RoMa model')
    parser.add_argument('--roma_upsample_res', nargs=2, type=int,
                        default=(560, 560), help='upsample resolution for RoMa model')
    parser.add_argument('--max_keypoints', type=int,
                        default=10, help='Maximal number of keypoints to extract')
    parser.add_argument('--nms_radius_prec', type=float,
                        default=0.054, help='Non Maximal Suppresion Radius in percentage')
    parser.add_argument('--roma_batch_size', type=int,
                        default=25, help='batch size of the number of pairs RoMa would process at once')
    # ---------------------------------- Matcher settings ----------------------------------
    # ---------------------------------- PCK settings ----------------------------------
    parser.add_argument('--alpha_list', type=list,
                        default=[0.1], help='alpha list for each alpha will be calculated, alpha@PCK')
    # ---------------------------------- PCK settings ----------------------------------
    # ---------------------------------- train settings ----------------------------------
    parser.add_argument('--start_with_id_matrix', action=argparse.BooleanOptionalAction,
                        default=True, help='Flag that if is True, sets the last layer of the model to create id transform matrix')
    parser.add_argument('--matrix_exp', action=argparse.BooleanOptionalAction,
                        default=True, help='Flag that if is True, activates the lie algebra logic in the code')
    parser.add_argument('--hidden_channels', type=int,
                        default=128, help='dim size of the feature vector in the hidden layers')
    parser.add_argument('--num_layers', type=int,
                        default=5, help='number of hiddne layers')
    parser.add_argument('--stn_n', type=int,
                        default=1, help='Number of times the IC-STN would be applied')
    parser.add_argument('--reflection_epoch_check', type=int,
                        default=100, help='frequency of epohcs to check for flips / reflections')
    parser.add_argument('--num_epochs', type=int,
                        default=600, help='Number of epochs to train the model')
    parser.add_argument('--lr', type=float,
                        default=5e-3, help='Learning Rate of train model')
    parser.add_argument('--weight_decay', type=float,
                        default=0, help='Weight Decay for the train model')
    parser.add_argument('--sigma', type=float,
                        default=0.25, help='sigma param for the Geman-McClure loss')
    parser.add_argument('--patience_sched', type=int,
                        default=200, help='number of epochs for ReduceLROnPlateau')
    parser.add_argument('--factor_sched', type=float,
                        default=0.5, help='factor to multiply with lr to reduce')
    # ---------------------------------- train settings ----------------------------------
    # ---------------------------------- Visualize Results settings ----------------------------------
    parser.add_argument('--visualize_results', action=argparse.BooleanOptionalAction,
                        default=True, help='Flag to visualize or not results')
    parser.add_argument('--ref', type=int,
                        default=0, help='defines which image are all images warped to for visualization only')
    # ---------------------------------- Visualize Results settings ----------------------------------
    # ---------------------------------- Graph Build settings ----------------------------------
    parser.add_argument('--k_clusters', type=int,
                        default=15, help='Number of clusters for k-means clustering (only used when clustering_method=kmeans)')
    parser.add_argument('--node_fuse_threshold', type=float,
                        default=0.02, help='Threshold to merge nodes / keypoints in the same image if their distance is smaller then node_fuse_threshold')
    parser.add_argument('--node_dist_outlier_threshold', type=float,
                        default=None, help='Threshold to consider nodes / keypoints outliers if their distance is larger then node_dist_outlier_threshold, defualt is None, if None outliers are if distance is larger than mean_dist + 1.5 * std_dist, if pass -1 is equal to None')
    parser.add_argument('--dpmeans_delta', type=float,
                        default=0.1, help='Delta parameter for DP-Means clustering (replaces lambda from the paper)')
    parser.add_argument('--dpmeans_n_init', type=int,
                        default=3, help='Number of initializations for DP-Means clustering')
    # ---------------------------------- Graph Build settings ----------------------------------

    return parser

if __name__ == '__main__':
    parser = get_argparser()
    args = parser.parse_args()
    if args.node_dist_outlier_threshold < 0:
        args.node_dist_outlier_threshold = None
    for arg in vars(args):
        print(arg, getattr(args, arg))