import argparse
from models.utils import str2bool


def parse_semantickitti_args():
    parser = argparse.ArgumentParser(description='Model Arguments')

    # dataset name
    parser.add_argument('--dataset', default='semantickitti', type=str, help='shapenet or semantickitti')
    # optimizer
    parser.add_argument('--lr', default=1e-3, type=float, help='initial learning rate for backbone')
    parser.add_argument('--aux_lr', default=1e-3, type=float, help='learning rate for entropy model')
    parser.add_argument('--weight_decay', default=0, type=float, help='weight decay for adam optimizer')
    parser.add_argument('--betas', default=(0.9, 0.999), type=float, help='betas for adam optimizer')
    # lr scheduler
    parser.add_argument('--lr_decay_step', default=15, type=int, help='learning rate decay step size')
    parser.add_argument('--gamma', default=0.5, type=float, help='gamma for scheduler_steplr')
    # dataset
    parser.add_argument('--train_data_path', default='./data/semantickitti/semantickitti_train_cube_size_12.pkl', type=str, help='path to train dataset')
    parser.add_argument('--train_cube_size', default=12, type=int, help='cube size of train dataset')
    parser.add_argument('--val_data_path', default='./data/semantickitti/semantickitti_val_cube_size_12.pkl', type=str, help='path to val dataset')
    parser.add_argument('--val_cube_size', default=12, type=int, help='cube size of val dataset')
    parser.add_argument('--test_data_path', default='./data/semantickitti/semantickitti_test_cube_size_12.pkl', type=str, help='path to test dataset')
    parser.add_argument('--test_cube_size', default=12, type=int, help='cube size of test dataset')
    parser.add_argument('--peak', default=None, type=float, help='peak value for PSNR calculation')
    # train
    parser.add_argument('--epochs', default=50, type=int, help='training epochs')
    parser.add_argument('--batch_size', default=1, type=int, help='batch_size')
    parser.add_argument('--print_freq', default=1000, type=int, help='loss print frequency')
    parser.add_argument('--save_freq', default=10, type=int, help='save frequency')
    parser.add_argument('--output_path', default='./output', type=str, help='output path')
    # attribute compression
    parser.add_argument('--compress_normal', default=False, type=str2bool, help='whether compress normals')
    parser.add_argument('--in_fdim', default=3, type=int, help='input dimension, may contain attributes')
    # model
    parser.add_argument('--k', default=16, type=int, help='knearest neighbor')
    parser.add_argument('--downsample_rate', default=[1/3, 1/3, 1/3], nargs='+', type=float, help='downsample rate')
    parser.add_argument('--max_upsample_num', default=[8, 8, 8], nargs='+', type=int, help='max upsmaple number, reversely symmetric with downsample_rate')
    parser.add_argument('--layer_num', default=3, type=int, help='downsample/upsmaple stage')
    parser.add_argument('--dim', default=8, type=int, help='feature dimension')
    parser.add_argument('--hidden_dim', default=64, type=int, help='hiddem dimension')
    parser.add_argument('--ngroups', default=1, type=int, help='groups for groupnorm')
    parser.add_argument('--quantize_latent_xyzs', default=True, type=str2bool, help='whether compress latent xyzs')
    parser.add_argument('--latent_xyzs_conv_mode', default='mlp', type=str, help='latent xyzs conv mode, mlp or edge_conv')
    parser.add_argument('--sub_point_conv_mode', default='mlp', type=str, help='sub-point conv mode, mlp or edge_conv')
    # loss
    parser.add_argument('--chamfer_coe', default=0.1, type=float, help='chamfer loss coefficient for intermediate point clouds')
    parser.add_argument('--pts_num_coe', default=5e-7, type=float, help='pts num loss coefficient')
    parser.add_argument('--normal_coe', default=1e-2, type=float, help='normal loss coefficient')
    parser.add_argument('--bpp_lambda', default=5e-4, type=float, help='bpp loss coefficient')
    parser.add_argument('--mean_distance_coe', default=5e1, type=float, help='mean distance loss coefficient')
    parser.add_argument('--density_coe', default=1e-4, type=float, help='density loss coefficient')
    parser.add_argument('--latent_xyzs_coe', default=1e-2, type=float, help='latent xyzs loss coefficient')
    # test
    parser.add_argument('--model_path', default='path to ckpt', type=str, help='path to ckpt')
    parser.add_argument('--density_radius', default=0.15, type=float, help='radius of query ball for density metric')
    parser.add_argument('--dist_coe', default=1e-5, type=float, help='distance coefficient for density metric')
    parser.add_argument('--omega_xyzs', default=0.5, type=float, help='xyzs threshold used for f1 score calculation')
    parser.add_argument('--omega_normals', default=0.5, type=float, help='normals threshold used for f1 score calculation')

    args = parser.parse_args()

    return args