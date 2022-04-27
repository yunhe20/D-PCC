import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from models.pointops.functions import pointops
from models.Chamfer3D.dist_chamfer_3D import chamfer_3DDist
from einops import rearrange, repeat
from models.utils import index_points
import math
from models.icosahedron2sphere import icosahedron2sphere
chamfer_dist = chamfer_3DDist()




class PointTransformerLayer(nn.Module):
    def __init__(self, in_fdim, out_fdim, args):
        super(PointTransformerLayer, self).__init__()

        self.w_qs = nn.Conv1d(in_fdim, args.hidden_dim, 1)
        self.w_ks = nn.Conv1d(in_fdim, args.hidden_dim, 1)
        self.w_vs = nn.Conv1d(in_fdim, args.hidden_dim, 1)

        self.conv_delta = nn.Sequential(
            nn.Conv2d(3, args.hidden_dim, 1),
            nn.GroupNorm(args.ngroups, args.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(args.hidden_dim, args.hidden_dim, 1)
        )

        self.conv_gamma = nn.Sequential(
            nn.Conv2d(args.hidden_dim, args.hidden_dim, 1),
            nn.GroupNorm(args.ngroups, args.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(args.hidden_dim, args.hidden_dim, 1)
        )

        self.post_conv = nn.Conv1d(args.hidden_dim, out_fdim, 1)


    def forward(self, q_xyzs, k_xyzs, q_feats, k_feats, v_feats, knn_idx, mask):
        # q: (b, c, m), k: (b, c, n), knn_idx: (b, m, k), mask: (b, m, k)

        # (b, 3, m, k)
        knn_xyzs = index_points(k_xyzs, knn_idx)

        # note it's q_feats but not v_feats
        pre = q_feats
        # (b, c, m)
        query = self.w_qs(q_feats)
        # (b, c, m, k)
        key = index_points(self.w_ks(k_feats), knn_idx)
        value = index_points(self.w_vs(v_feats), knn_idx)

        # (b, c, m, k)
        pos_enc = self.conv_delta(q_xyzs.unsqueeze(-1) - knn_xyzs)
        # attention
        attn = self.conv_gamma(query.unsqueeze(-1) - key + pos_enc)
        # (b, c, m, k)
        attn = attn / math.sqrt(key.shape[1])

        # mask
        mask_value = -(torch.finfo(attn.dtype).max)
        attn.masked_fill_(~mask[:, None], mask_value)
        attn = F.softmax(attn, dim=-1)

        res = torch.einsum('bcmk, bcmk->bcm', attn, value + pos_enc)
        res = self.post_conv(res) + pre

        return res




class PositionEmbeddingLayer(nn.Module):
    def __init__(self, args):
        super(PositionEmbeddingLayer, self).__init__()

        self.pre_nn = nn.Sequential(
            nn.Conv2d(4, args.hidden_dim, 1),
            nn.GroupNorm(args.ngroups, args.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(args.hidden_dim, args.dim, 1)
        )

        self.attn_nn = nn.Sequential(
            nn.Conv2d(args.dim, args.hidden_dim, 1),
            nn.GroupNorm(args.ngroups, args.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(args.hidden_dim, args.dim, 1)
        )


    def forward(self, q_xyzs, k_xyzs, knn_idx, mask):
        # q_xyzs: (b, 3, m), k_xyzs: (b, 3, n), knn_idx and mask: (b, m, k)

        # (b, 3, m, k)
        knn_xyzs = index_points(k_xyzs, knn_idx)
        # (b, 3, m, k)
        k = knn_xyzs.shape[-1]
        repeated_xyzs = q_xyzs[..., None].repeat(1, 1, 1, k)

        # (b, 3, m, k)
        direction = F.normalize(knn_xyzs - repeated_xyzs, p=2, dim=1)
        # (b, 1, m, k)
        distance = torch.norm(knn_xyzs - repeated_xyzs, p=2, dim=1, keepdim=True)
        # (b, 4, m, k)
        local_pattern = torch.cat((direction, distance), dim=1)

        # (b, c, m, k)
        position_embedding = self.pre_nn(local_pattern)

        # (b, c, m, k)
        attn = self.attn_nn(position_embedding)
        # mask
        mask_value = -(torch.finfo(attn.dtype).max)
        attn.masked_fill_(~mask[:, None], mask_value)
        attn = F.softmax(attn, dim=-1)

        position_embedding = position_embedding * attn
        # (b, c, m)
        position_embedding = position_embedding.sum(dim=-1)

        return position_embedding




class DensityEmbeddingLayer(nn.Module):
    def __init__(self, args):
        super(DensityEmbeddingLayer, self).__init__()

        self.pre_nn = nn.Sequential(
            nn.Conv1d(1, args.hidden_dim, 1),
            nn.GroupNorm(args.ngroups, args.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(args.hidden_dim, args.dim, 1)
        )


    def forward(self, downsample_num):
        # input: (b, 1, n)

        density_embedding = self.pre_nn(downsample_num)

        return density_embedding




class DownsampleLayer(nn.Module):
    def __init__(self, args, layer_idx):
        super(DownsampleLayer, self).__init__()

        self.args = args
        self.k = args.k
        self.downsample_rate = args.downsample_rate[layer_idx]

        self.pre_conv = nn.Conv1d(args.dim, args.dim, 1)

        self.feats_agg_nn = PointTransformerLayer(args.dim, args.dim, args)
        self.position_embedding_nn = PositionEmbeddingLayer(args)
        self.density_embedding_nn = DensityEmbeddingLayer(args)

        self.post_conv = nn.Conv1d(args.dim * 3, args.dim, 1)


    def get_density(self, sampled_xyzs, xyzs):
        # input: (b, 3, m), (b, 3, n)

        batch_size = xyzs.shape[0]
        sample_num = sampled_xyzs.shape[2]
        # (b, n, 3)
        xyzs_trans = xyzs.permute(0, 2, 1).contiguous()
        # (b, sample_num, 3)
        sampled_xyzs_trans = sampled_xyzs.permute(0, 2, 1).contiguous()

        # find the nearest neighbor in sampled_xyzs_trans: (b, n, 1)
        ori2sample_idx = pointops.knnquery_heap(1, sampled_xyzs_trans, xyzs_trans)

        # (b, sample_num)
        downsample_num = torch.zeros((batch_size, sample_num)).cuda()
        for i in range(batch_size):
            uniques, counts = torch.unique(ori2sample_idx[i], return_counts=True)
            downsample_num[i][uniques.long()] = counts.float()

        # (b, m, k)
        knn_idx = pointops.knnquery_heap(self.k, xyzs_trans, sampled_xyzs_trans).long()
        torch.cuda.empty_cache()

        # mask: (m)
        expect_center = torch.arange(0, sample_num).cuda()
        # (b, m, k)
        expect_center = repeat(expect_center, 'm -> b m k', b=batch_size, k=self.k)
        # (b, 1, m, k)
        real_center = index_points(ori2sample_idx.permute(0, 2, 1).contiguous(), knn_idx)
        # (b, m, k)
        real_center = real_center.squeeze(1)
        # mask those points that not belong to collapsed points set
        mask = torch.eq(expect_center, real_center)

        # (b, 3, m, k)
        knn_xyzs = index_points(xyzs, knn_idx)
        # (b, 1, m, k)
        distance = torch.norm(knn_xyzs - sampled_xyzs[..., None], p=2, dim=1, keepdim=True)
        # mask
        mask_value = 0
        distance.masked_fill_(~mask[:, None], mask_value)
        # (b, m)
        distance = distance.sum(dim=-1).squeeze(1)
        # (b, m)
        mean_distance = distance / downsample_num

        return downsample_num, mean_distance, mask, knn_idx


    def forward(self, xyzs, feats):
        # xyzs: (b, 3, n), features: (b, cin, n)
        if self.k > xyzs.shape[2]:
            self.k = xyzs.shape[2]

        sample_num = round(xyzs.shape[2] * self.downsample_rate)
        # (b, n, 3)
        xyzs_trans = xyzs.permute(0, 2, 1).contiguous()

        # FPS, (b, sample_num)
        sample_idx = pointops.furthestsampling(xyzs_trans, sample_num).long()
        # (b, 3, sample_num)
        sampled_xyzs = index_points(xyzs, sample_idx)

        # get density
        downsample_num, mean_distance, mask, knn_idx = self.get_density(sampled_xyzs, xyzs)

        identity = feats
        feats = self.pre_conv(feats)
        # (b, c, sample_num)
        sampled_feats = index_points(feats, sample_idx)

        # embedding
        ancestor_embedding = self.feats_agg_nn(sampled_xyzs, xyzs, sampled_feats, feats, feats, knn_idx, mask)
        position_embedding = self.position_embedding_nn(sampled_xyzs, xyzs, knn_idx, mask)
        density_embedding = self.density_embedding_nn(downsample_num.unsqueeze(1))
        # (b, 3c, m)
        agg_embedding = torch.cat((ancestor_embedding, position_embedding, density_embedding), dim=1)
        # (b, c, m)
        agg_embedding = self.post_conv(agg_embedding)

        # residual connection: (b, c, m)
        sampled_feats = agg_embedding + index_points(identity, sample_idx)

        return sampled_xyzs, sampled_feats, downsample_num, mean_distance



class EdgeConv(nn.Module):
    def __init__(self, args, in_fdim, out_fdim):
        super(EdgeConv, self).__init__()

        self.k = args.k

        self.conv = nn.Sequential(
            nn.Conv2d(2*in_fdim, args.hidden_dim, 1),
            nn.ReLU(),
            nn.Conv2d(args.hidden_dim, out_fdim, 1)
        )


    def knn(self, feats):
        inner = -2 * torch.matmul(feats.transpose(2, 1), feats)
        xx = torch.sum(feats ** 2, dim=1, keepdim=True)
        pairwise_distance = -xx - inner - xx.transpose(2, 1)
        # (b, n, k)
        knn_idx = pairwise_distance.topk(k=self.k, dim=-1)[1]

        return knn_idx


    def get_graph_features(self, feats):
        dim = feats.shape[1]
        if dim == 3:
            # (b, n, 3)
            feats_trans = feats.permute(0, 2, 1).contiguous()
            # (b, n, k)
            knn_idx = pointops.knnquery_heap(self.k, feats_trans, feats_trans).long()
        else:
            # it needs a huge memory cost
            knn_idx = self.knn(feats)
        torch.cuda.empty_cache()
        # (b, c, n, k)
        knn_feats = index_points(feats, knn_idx)
        repeated_feats = repeat(feats, 'b c n -> b c n k', k=self.k)
        # (b, 2c, n, k)
        graph_feats = torch.cat((knn_feats-repeated_feats, repeated_feats), dim=1)

        return graph_feats


    def forward(self, feats):
        # input: (b, c, n)
        if feats.shape[2] < self.k:
            self.k = feats.shape[2]

        graph_feats = self.get_graph_features(feats)
        # (b, cout*g, n, k)
        expanded_feats = self.conv(graph_feats)
        # (b, cout*g, n)
        expanded_feats = torch.max(expanded_feats, dim=-1)[0]

        return expanded_feats




class SubPointConv(nn.Module):
    def __init__(self, args, in_fdim, out_fdim, group_num):
        super(SubPointConv, self).__init__()

        assert args.sub_point_conv_mode in ['mlp', 'edge_conv']
        self.mode = args.sub_point_conv_mode
        self.hidden_dim = args.hidden_dim
        self.group_num = group_num
        self.group_in_fdim = in_fdim // group_num
        self.group_out_fdim = out_fdim // group_num

        # mlp
        if self.mode == 'mlp':
            self.mlp = nn.Sequential(
                nn.Conv2d(self.group_in_fdim, self.hidden_dim, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.hidden_dim, self.group_out_fdim, 1)
            )
        else:
            # edge_conv
            self.edge_conv = EdgeConv(args, in_fdim, out_fdim)


    def forward(self, feats):
        if self.mode == 'mlp':
            # per-group conv: (b, cin, n, g)
            feats = rearrange(feats, 'b (c g) n -> b c n g', g=self.group_num).contiguous()
            # (b, cout, n, g)
            expanded_feats = self.mlp(feats)
        else:
            # (b, cout*g, n)
            expanded_feats = self.edge_conv(feats)
            # shuffle: (b, cout, n, g)
            expanded_feats = rearrange(expanded_feats, 'b (c g) n -> b c n g', g=self.group_num).contiguous()

        return expanded_feats




class XyzsUpsampleLayer(nn.Module):
    def __init__(self, args, layer_idx, upsample_rate=None):
        super(XyzsUpsampleLayer, self).__init__()

        if upsample_rate == None:
            self.upsample_rate = args.max_upsample_num[layer_idx]
        else:
            self.upsample_rate = upsample_rate

        # each point has fixed 43 candidate directions, size (43, 3)
        hypothesis, _ = icosahedron2sphere(1)
        hypothesis = np.append(np.zeros((1,3)), hypothesis, axis=0)
        self.hypothesis = torch.from_numpy(hypothesis).float().cuda()

        # weights
        self.weight_nn = SubPointConv(args, args.dim, 43*self.upsample_rate, self.upsample_rate)

        # scales
        self.scale_nn = SubPointConv(args, args.dim, 1*self.upsample_rate, self.upsample_rate)


    def forward(self, xyzs, feats):
        # xyzs: (b, 3, n)  feats (b, c, n)
        batch_size = xyzs.shape[0]
        points_num = xyzs.shape[2]

        # (b, 43, n, u)
        weights = self.weight_nn(feats)
        # (b, 43, 1, n, u)
        weights = weights.unsqueeze(2)
        weights = F.softmax(weights, dim=1)

        # (b, 43, 3, n, u)
        hypothesis = repeat(self.hypothesis, 'h c -> b h c n u', b=batch_size, n=points_num, u=self.upsample_rate)
        weighted_hypothesis = weights * hypothesis
        # (b, 3, n, u)
        directions = torch.sum(weighted_hypothesis, dim=1)
        # normalize
        directions = F.normalize(directions, p=2, dim=1)

        # (b, 1, n, u)
        scales = self.scale_nn(feats)

        # (b, 3, n, u)
        deltas = directions * scales

        # (b, 3, n, u)
        repeated_xyzs = repeat(xyzs, 'b c n -> b c n u', u=self.upsample_rate)
        upsampled_xyzs = repeated_xyzs + deltas

        return upsampled_xyzs




class FeatsUpsampleLayer(nn.Module):
    def __init__(self, args, layer_idx, upsample_rate=None, decompress_normal=False):
        super(FeatsUpsampleLayer, self).__init__()

        if upsample_rate == None:
            self.upsample_rate = args.max_upsample_num[layer_idx]
        else:
            self.upsample_rate = upsample_rate

        # weather decompress normal
        self.decompress_normal = decompress_normal
        if self.decompress_normal:
            self.out_fdim = 3
        else:
            self.out_fdim = args.dim

        self.feats_nn = SubPointConv(args, args.dim, self.out_fdim * self.upsample_rate, self.upsample_rate)


    def forward(self, feats):
        # (b, c, n, u)
        upsampled_feats = self.feats_nn(feats)

        if self.decompress_normal == False:
            # shortcut
            repeated_feats = repeat(feats, 'b c n -> b c n u', u=self.upsample_rate)
            # (b, c, n, u)
            upsampled_feats = upsampled_feats + repeated_feats

        return upsampled_feats




class UpsampleLayer(nn.Module):
    def __init__(self, args, layer_idx):
        super(UpsampleLayer, self).__init__()

        self.xyzs_upsample_nn = XyzsUpsampleLayer(args, layer_idx)
        self.feats_upsample_nn = FeatsUpsampleLayer(args, layer_idx)


    def forward(self, xyzs, feats):
        upsampled_xyzs = self.xyzs_upsample_nn(xyzs, feats)
        upsampled_feats = self.feats_upsample_nn(feats)

        return upsampled_xyzs, upsampled_feats
