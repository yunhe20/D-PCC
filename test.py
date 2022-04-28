import torch
import os
import sys
sys.path.append(os.getcwd())
import numpy as np
from models.autoencoder import AutoEncoder
import time
import argparse
from args.shapenet_args import parse_shapenet_args
from args.semantickitti_args import parse_semantickitti_args
from models.utils import save_pcd, AverageMeter, str2bool
from dataset.dataset import CompressDataset
from metrics.PSNR import get_psnr
from metrics.density import get_density_metric
from metrics.F1Score import get_f1_score
from models.Chamfer3D.dist_chamfer_3D import chamfer_3DDist
chamfer_dist = chamfer_3DDist()




def make_dirs(save_dir):
    gt_patch_dir = os.path.join(save_dir, 'patch/gt')
    if not os.path.exists(gt_patch_dir):
        os.makedirs(gt_patch_dir)
    pred_patch_dir = os.path.join(save_dir, 'patch/pred')
    if not os.path.exists(pred_patch_dir):
        os.makedirs(pred_patch_dir)
    gt_merge_dir = os.path.join(save_dir, 'merge/gt')
    if not os.path.exists(gt_merge_dir):
        os.makedirs(gt_merge_dir)
    pred_merge_dir = os.path.join(save_dir, 'merge/pred')
    if not os.path.exists(pred_merge_dir):
        os.makedirs(pred_merge_dir)

    return gt_patch_dir, pred_patch_dir, gt_merge_dir, pred_merge_dir




def load_model(args, model_path):
    # load model
    model = AutoEncoder(args).cuda()
    model.load_state_dict(torch.load(model_path))
    # update entropy bottleneck
    model.feats_eblock.update(force=True)
    if args.quantize_latent_xyzs == True:
        model.xyzs_eblock.update(force=True)
    model.eval()

    return model




def compress(args, model, xyzs, feats):
    # input: (b, c, n)

    encode_start = time.time()
    # raise dimension
    feats = model.pre_conv(feats)

    # encoder forward
    gt_xyzs, gt_dnums, gt_mdis, latent_xyzs, latent_feats = model.encoder(xyzs, feats)
    # decompress size
    feats_size = latent_feats.size()[2:]

    # compress latent feats
    latent_feats_str = model.feats_eblock.compress(latent_feats)

    # compress latent xyzs
    if args.quantize_latent_xyzs == True:
        analyzed_latent_xyzs = model.latent_xyzs_analysis(latent_xyzs)
        # decompress size
        xyzs_size = analyzed_latent_xyzs.size()[2:]
        latent_xyzs_str = model.xyzs_eblock.compress(analyzed_latent_xyzs)
    else:
        # half float representation
        latent_xyzs_str = latent_xyzs.half()
        xyzs_size = None

    encode_time = time.time() - encode_start

    # bpp calculation
    points_num = xyzs.shape[0] * xyzs.shape[2]
    feats_bpp = (sum(len(s) for s in latent_feats_str) * 8.0) / points_num
    if args.quantize_latent_xyzs == True:
        xyzs_bpp = (sum(len(s) for s in latent_xyzs_str) * 8.0) / points_num
    else:
        xyzs_bpp = (latent_xyzs.shape[0] * latent_xyzs.shape[2] * 16 * 3) / points_num
    actual_bpp = feats_bpp + xyzs_bpp


    return latent_xyzs_str, xyzs_size, latent_feats_str, feats_size, encode_time, actual_bpp




def decompress(args, model, latent_xyzs_str, xyzs_size, latent_feats_str, feats_size):
    decode_start = time.time()
    # decompress latent xyzs
    if args.quantize_latent_xyzs == True:
        analyzed_latent_xyzs_hat = model.xyzs_eblock.decompress(latent_xyzs_str, xyzs_size)
        latent_xyzs_hat = model.latent_xyzs_synthesis(analyzed_latent_xyzs_hat)
    else:
        latent_xyzs_hat = latent_xyzs_str

    # decompress latent feats
    latent_feats_hat = model.feats_eblock.decompress(latent_feats_str, feats_size)

    # decoder forward
    pred_xyzs, pred_unums, pred_mdis, upsampled_feats = model.decoder(latent_xyzs_hat, latent_feats_hat)

    decode_time = time.time() - decode_start

    return pred_xyzs[-1], upsampled_feats, decode_time




def test_xyzs(args):
    # load data
    test_dataset = CompressDataset(data_path=args.test_data_path, cube_size=args.test_cube_size)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.batch_size)
    # indicate the last patch number of each full point cloud
    pcd_last_patch_num = test_dataset.pcd_last_patch_num

    # set up folders for saving point clouds
    model_path = args.model_path
    experiment_id = model_path.split('/')[-3]
    save_dir = os.path.join(args.output_path, experiment_id, 'pcd')
    gt_patch_dir, pred_patch_dir, gt_merge_dir, pred_merge_dir = make_dirs(save_dir)

    # load model
    model = load_model(args, model_path)

    # metrics
    patch_bpp = AverageMeter()
    patch_chamfer_loss = AverageMeter()
    patch_psnr = AverageMeter()
    patch_density_metric = AverageMeter()
    patch_encode_time = AverageMeter()
    patch_decode_time = AverageMeter()
    pcd_num = 0
    pcd_bpp = AverageMeter()
    pcd_chamfer_loss = AverageMeter()
    pcd_psnr = AverageMeter()
    pcd_density_metric = AverageMeter()

    # merge xyzs
    pcd_gt_patches = []
    pcd_pred_patches = []

    # test
    with torch.no_grad():
        for i, input_dict in enumerate(test_loader):
            # input: (b, n, c)
            input = input_dict['xyzs'].cuda()
            # normals : (b, n, c)
            gt_normals = input_dict['normals'].cuda()
            # (b, c, n)
            input = input.permute(0, 2, 1).contiguous()
            xyzs = input[:, :3, :].contiguous()
            gt_patches = xyzs
            feats = input

            # compress
            latent_xyzs_str, xyzs_size, latent_feats_str, feats_size, encode_time, \
            actual_bpp = compress(args, model, xyzs, feats)

            # update metrics
            patch_encode_time.update(encode_time)
            patch_bpp.update(actual_bpp)
            pcd_bpp.update(actual_bpp)

            # decompress
            pred_patches, upsampled_feats, decode_time \
                = decompress(args, model, latent_xyzs_str, xyzs_size, latent_feats_str, feats_size)
            patch_decode_time.update(decode_time)

            # calculate metrics
            # (b, 3, n) -> (b, n, 3)
            gt_patches = gt_patches.permute(0, 2, 1).contiguous()
            pred_patches = pred_patches.permute(0, 2, 1).contiguous()
            # chamfer distance
            gt2pred_loss, pred2gt_loss, _, _ = chamfer_dist(gt_patches, pred_patches)
            chamfer_loss = gt2pred_loss.mean() + pred2gt_loss.mean()
            patch_chamfer_loss.update(chamfer_loss.item())
            pcd_chamfer_loss.update(chamfer_loss.item())
            # psnr
            psnr = get_psnr(gt_patches, gt_normals, pred_patches, test_loader, args)
            # the psnr may be inf when the normals are not accurate
            if not torch.isinf(psnr):
                patch_psnr.update(psnr.item())
                pcd_psnr.update(psnr.item())
            # density metric
            density_metric = get_density_metric(gt_patches, pred_patches, args)
            patch_density_metric.update(density_metric.item())
            pcd_density_metric.update(density_metric.item())

            # scale patches to original size: (n, 3)
            original_gt_patches = test_dataset.scale_to_origin(gt_patches.detach().cpu(), i).squeeze(0).numpy()
            original_pred_patches = test_dataset.scale_to_origin(pred_patches.detach().cpu(), i).squeeze(0).numpy()
            # save patches
            save_pcd(gt_patch_dir, str(i) + '.ply', original_gt_patches)
            save_pcd(pred_patch_dir, str(i) + '.ply', original_pred_patches)

            # merge patches
            pcd_gt_patches.append(original_gt_patches)
            pcd_pred_patches.append(original_pred_patches)
            # generate the full point cloud
            if i == pcd_last_patch_num[pcd_num] - 1:
                gt_pcd = np.concatenate((pcd_gt_patches), axis=0)
                pred_pcd = np.concatenate((pcd_pred_patches), axis=0)

                # averaged metrics of each full point cloud
                print("pcd:", pcd_num, "pcd bpp:", pcd_bpp.get_avg(), "pcd chamfer loss:", pcd_chamfer_loss.get_avg(),
                      "pcd psnr:", pcd_psnr.get_avg(), 'pcd density metric:', pcd_density_metric.get_avg())

                # save the full point cloud
                save_pcd(gt_merge_dir, str(pcd_num) + '.ply', gt_pcd)
                save_pcd(pred_merge_dir, str(pcd_num) + '.ply', pred_pcd)

                # reset
                pcd_num += 1
                pcd_gt_patches.clear()
                pcd_pred_patches.clear()
                pcd_bpp.reset()
                pcd_chamfer_loss.reset()
                pcd_psnr.reset()
                pcd_density_metric.reset()

            # current patch
            print("patch:", i, "patch bpp:", actual_bpp, "chamfer loss:", chamfer_loss.item(),
                  "psnr:", psnr.item(), "density metric:", density_metric.item(),
                  "encode time:", encode_time, "decode time:", decode_time)

    # averaged metrics of the whole dataset
    print("avg patch bpp:", patch_bpp.get_avg())
    print("avg chamfer loss:", patch_chamfer_loss.get_avg())
    print("avg psnr:", patch_psnr.get_avg())
    print("avg density metric:", patch_density_metric.get_avg())
    print("avg encode time:", patch_encode_time.get_avg())
    print("avg decode time:", patch_decode_time.get_avg())




def test_normals(args):
    # load data
    test_dataset = CompressDataset(data_path=args.test_data_path, cube_size=args.test_cube_size)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.batch_size)
    # indicate the last patch number of each full point cloud
    pcd_last_patch_num = test_dataset.pcd_last_patch_num

    # set up folders for saving point clouds
    model_path = args.model_path
    experiment_id = model_path.split('/')[-3]
    save_dir = os.path.join(args.output_path, experiment_id, 'pcd')
    gt_patch_dir, pred_patch_dir, gt_merge_dir, pred_merge_dir = make_dirs(save_dir)

    # load model
    args.in_fdim = 6
    model = load_model(args, model_path)

    # metrics
    patch_bpp = AverageMeter()
    patch_f1_score = AverageMeter()
    patch_encode_time = AverageMeter()
    patch_decode_time = AverageMeter()
    pcd_num = 0
    pcd_bpp = AverageMeter()
    pcd_f1_score = AverageMeter()

    # merge xyzs and normals
    pcd_gt_patches = []
    pcd_pred_patches = []
    pcd_gt_normals = []
    pcd_pred_normals = []

    # test
    with torch.no_grad():
        for i, input_dict in enumerate(test_loader):
            # input: (b, n, c)
            input = input_dict['xyzs'].cuda()
            # normals : (b, n, c)
            gt_normals = input_dict['normals'].cuda()
            # (b, c, n)
            input = input.permute(0, 2, 1).contiguous()
            # concat normals
            input = torch.cat((input, gt_normals.permute(0, 2, 1).contiguous()), dim=1)
            xyzs = input[:, :3, :].contiguous()
            gt_patches = xyzs
            feats = input

            # compress
            latent_xyzs_str, xyzs_size, latent_feats_str, feats_size, encode_time, \
            actual_bpp = compress(args, model, xyzs, feats)

            # update metrics
            patch_encode_time.update(encode_time)
            patch_bpp.update(actual_bpp)
            pcd_bpp.update(actual_bpp)

            # decompress
            pred_patches, upsampled_feats, decode_time \
                = decompress(args, model, latent_xyzs_str, xyzs_size, latent_feats_str, feats_size)
            pred_normals = torch.tanh(upsampled_feats).permute(0, 2, 1).contiguous()
            patch_decode_time.update(decode_time)

            # calculate metrics
            # (b, 3, n) -> (b, n, 3)
            gt_patches = gt_patches.permute(0, 2, 1).contiguous()
            pred_patches = pred_patches.permute(0, 2, 1).contiguous()
            # f1 score
            f1_score = get_f1_score(gt_patches, gt_normals, pred_patches, pred_normals, args)
            patch_f1_score.update(f1_score)
            pcd_f1_score.update(f1_score)

            # scale patches to original size: (n, 3)
            original_gt_patches = test_dataset.scale_to_origin(gt_patches.detach().cpu(), i).squeeze(0).numpy()
            original_pred_patches = test_dataset.scale_to_origin(pred_patches.detach().cpu(), i).squeeze(0).numpy()
            # tensor -> numpy
            gt_normals = gt_normals.squeeze(0).detach().cpu().numpy()
            pred_normals = pred_normals.squeeze(0).detach().cpu().numpy()
            # save xyzs and normals
            save_pcd(gt_patch_dir, str(i) + '.ply', original_gt_patches, gt_normals)
            save_pcd(pred_patch_dir, str(i) + '.ply', original_pred_patches, pred_normals)

            # merge patches
            pcd_gt_patches.append(original_gt_patches)
            pcd_pred_patches.append(original_pred_patches)
            pcd_gt_normals.append(gt_normals)
            pcd_pred_normals.append(pred_normals)
            # generate the full point cloud
            if i == pcd_last_patch_num[pcd_num] - 1:
                gt_pcd = np.concatenate((pcd_gt_patches), axis=0)
                pred_pcd = np.concatenate((pcd_pred_patches), axis=0)

                # averaged metrics of each full point cloud
                print("pcd:", pcd_num, "pcd bpp:", pcd_bpp.get_avg(), "pcd f1 score:", pcd_f1_score.get_avg())

                # save the full point cloud
                save_pcd(gt_merge_dir, str(pcd_num) + '.ply', gt_pcd, np.concatenate((pcd_gt_normals), axis=0))
                save_pcd(pred_merge_dir, str(pcd_num) + '.ply', pred_pcd, np.concatenate((pcd_pred_normals), axis=0))

                # reset
                pcd_num += 1
                pcd_gt_patches.clear()
                pcd_pred_patches.clear()
                pcd_gt_normals.clear()
                pcd_pred_normals.clear()
                pcd_bpp.reset()
                pcd_f1_score.reset()

            # current patch
            print("patch:", i, "patch bpp:", actual_bpp, "f1 score loss:", f1_score,
                  "encode time:", encode_time, "decode time:", decode_time)

    # averaged metrics of the whole dataset
    print("avg patch bpp:", patch_bpp.get_avg())
    print("avg f1 score:", patch_f1_score.get_avg())
    print("avg encode time:", patch_encode_time.get_avg())
    print("avg decode time:", patch_decode_time.get_avg())




def reset_model_args(test_args, model_args):
    for arg in vars(test_args):
        setattr(model_args, arg, getattr(test_args, arg))




def parse_test_args():
    parser = argparse.ArgumentParser(description='Test Arguments')

    # dataset
    parser.add_argument('--dataset', default='shapenet', type=str, help='shapenet or semantickitti')
    parser.add_argument('--model_path', default='path to ckpt', type=str, help='path to ckpt')
    parser.add_argument('--batch_size', default=1, type=int, help='the test batch_size must be 1')
    parser.add_argument('--downsample_rate', default=[1/3, 1/3, 1/3], nargs='+', type=float, help='downsample rate')
    parser.add_argument('--max_upsample_num', default=[8, 8, 8], nargs='+', type=int, help='max upsmaple number, reversely symmetric with downsample_rate')
    parser.add_argument('--bpp_lambda', default=1e-3, type=float, help='bpp loss coefficient')
    # normal compression
    parser.add_argument('--compress_normal', default=False, type=str2bool, help='whether compress normals')
    # compress latent xyzs
    parser.add_argument('--quantize_latent_xyzs', default=True, type=str2bool, help='whether compress latent xyzs')
    parser.add_argument('--latent_xyzs_conv_mode', default='mlp', type=str, help='latent xyzs conv mode, mlp or edge_conv')
    # sub_point_conv mode
    parser.add_argument('--sub_point_conv_mode', default='mlp', type=str, help='sub-point conv mode, mlp or edge_conv')

    args = parser.parse_args()
    return args




if __name__ == "__main__":
    test_args = parse_test_args()
    assert test_args.dataset in ['shapenet', 'semantickitti']
    # the test batch_size must be 1
    assert test_args.batch_size == 1

    if test_args.dataset == 'shapenet':
        model_args = parse_shapenet_args()
    else:
        model_args = parse_semantickitti_args()

    reset_model_args(test_args, model_args)

    if model_args.compress_normal == False:
        test_xyzs(model_args)
    else:
        test_normals(model_args)
