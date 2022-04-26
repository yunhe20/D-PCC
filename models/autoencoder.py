import torch
import torch.nn as nn
from models.encoder import  Encoder
from models.decoder import Decoder
from compressai.entropy_models import EntropyBottleneck
import math
from models.layer import EdgeConv
from models.loss import get_chamfer_loss, get_density_loss, get_pts_num_loss, get_normal_loss, get_latent_xyzs_loss




# compression model
class AutoEncoder(nn.Module):
    def __init__(self, args):
        super(AutoEncoder, self).__init__()

        self.args = args

        self.pre_conv = nn.Sequential(
            nn.Conv1d(args.in_fdim, args.hidden_dim, 1),
            nn.GroupNorm(args.ngroups, args.hidden_dim),
            nn.ReLU(),
            nn.Conv1d(args.hidden_dim, args.dim, 1)
        )
        self.encoder = Encoder(args)
        self.feats_eblock = EntropyBottleneck(args.dim)
        self.decoder = Decoder(args)

        if args.quantize_latent_xyzs == True:
            assert args.latent_xyzs_conv_mode in ['edge_conv', 'mlp']
            if args.latent_xyzs_conv_mode == 'edge_conv':
                self.latent_xyzs_analysis = EdgeConv(args, 3, args.dim)
            else:
                self.latent_xyzs_analysis = nn.Sequential(
                    nn.Conv1d(3, args.hidden_dim, 1),
                    nn.GroupNorm(args.ngroups, args.hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Conv1d(args.hidden_dim, args.dim, 1)
                )
            self.xyzs_eblock = EntropyBottleneck(args.dim)
            if args.latent_xyzs_conv_mode == 'edge_conv':
                self.latent_xyzs_synthesis = EdgeConv(args, args.dim, 3)
            else:
                self.latent_xyzs_synthesis = nn.Sequential(
                    nn.Conv1d(args.dim, args.hidden_dim, 1),
                    nn.GroupNorm(args.ngroups, args.hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Conv1d(args.hidden_dim, 3, 1)
                )


    def get_loss(self, gt_xyzs, gt_dnums, gt_mdis, pred_xyzs, pred_unums, pred_mdis):
        chamfer_loss, all_pred2gt_idx = get_chamfer_loss(gt_xyzs, pred_xyzs, self.args)
        density_loss = get_density_loss(gt_dnums, gt_mdis, pred_unums, pred_mdis, all_pred2gt_idx, self.args)
        pts_num_loss = get_pts_num_loss(gt_xyzs, pred_unums, self.args)

        loss = chamfer_loss + density_loss + pts_num_loss

        loss_items = {
            'chamfer_loss': chamfer_loss.item(),
            'density_loss': density_loss.item(),
            'pts_num_loss': pts_num_loss.item()
        }

        return loss, loss_items, all_pred2gt_idx


    def forward(self, input):
        # input: (b, c, n)
        points_num = input.shape[0] * input.shape[2]
        xyzs = input[:, :3, :].contiguous()
        # compress normal
        if self.args.compress_normal == True:
            gt_normals = input[:, 3:, :].contiguous()
        else:
            gt_normals = None
        feats = input

        # raise dimension
        feats = self.pre_conv(feats)

        # downsample
        gt_xyzs, gt_dnums, gt_mdis, latent_xyzs, latent_feats = self.encoder(xyzs, feats)

        # entropy bottleneck: compress latent feats
        latent_feats_hat, latent_feats_likelihoods = self.feats_eblock(latent_feats)
        # feats bpp calculation
        feats_size = (torch.log(latent_feats_likelihoods).sum()) / (-math.log(2))
        feats_bpp = feats_size / points_num

        if self.args.quantize_latent_xyzs == True:
            # compress latent xyzs, it is essentially a quantization
            gt_latent_xyzs = latent_xyzs
            analyzed_latent_xyzs = self.latent_xyzs_analysis(latent_xyzs)
            analyzed_latent_xyzs_hat, analyzed_latent_xyzs_likelihoods = self.xyzs_eblock(analyzed_latent_xyzs)
            pred_latent_xyzs = self.latent_xyzs_synthesis(analyzed_latent_xyzs_hat)
            # xyzs bpp calculation
            xyzs_size = (torch.log(analyzed_latent_xyzs_likelihoods).sum()) / (-math.log(2))
            xyzs_bpp = xyzs_size / points_num
        else:
            # half float representation
            gt_latent_xyzs = latent_xyzs
            pred_latent_xyzs = latent_xyzs.half()
            xyzs_size = pred_latent_xyzs.shape[0] * pred_latent_xyzs.shape[2] * 16 * 3
            xyzs_bpp = xyzs_size / points_num

        # upsample
        pred_xyzs, pred_unums, pred_mdis, upsampled_feats = self.decoder(pred_latent_xyzs, latent_feats_hat)

        # get loss
        loss, loss_items, all_pred2gt_idx = self.get_loss(gt_xyzs, gt_dnums, gt_mdis, pred_xyzs, pred_unums, pred_mdis)

        # latent_xyzs_loss
        if self.args.quantize_latent_xyzs == True:
            latent_xyzs_loss = get_latent_xyzs_loss(gt_latent_xyzs, pred_latent_xyzs, self.args)
            loss = loss + latent_xyzs_loss
            loss_items['latent_xyzs_loss'] = latent_xyzs_loss.item()
        else:
            loss_items['latent_xyzs_loss'] = 0.0

        # normal_loss
        if self.args.compress_normal == True:
            pred_normals = torch.tanh(upsampled_feats)
            normal_loss = get_normal_loss(gt_normals, pred_normals, all_pred2gt_idx[-1], self.args)
            loss = loss + normal_loss
            loss_items['normal_loss'] = normal_loss.item()
        else:
            loss_items['normal_loss'] = 0.0

        # bpp loss
        bpp = feats_bpp + xyzs_bpp
        bpp_loss = bpp * self.args.bpp_lambda
        loss = loss + bpp_loss
        loss_items['bpp_loss'] = bpp_loss.item()

        # decompressed xyzs
        decompressed_xyzs = pred_xyzs[-1]

        return decompressed_xyzs, loss, loss_items, bpp
