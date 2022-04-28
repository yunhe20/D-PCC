import torch
import torch.nn as nn
from models.layer import UpsampleLayer, XyzsUpsampleLayer, FeatsUpsampleLayer
from models.utils import index_points
from einops import rearrange, repeat
from models.pointops.functions import pointops
import random




def select_xyzs_and_feats(candidate_xyzs, candidate_feats, upsample_num):
    '''
    note that the batch_size should be 1
    :param candidate_xyzs: (b, 3, n, max_upsample_num)
    :param candidate_feats: (b, c, n, max_upsample_num)
    :param upsample_num: (b, n)
    :return: (b, 3, m), (b, c, m)
    '''
    upsample_num = upsample_num.round().long()

    batch_size = candidate_xyzs.shape[0]
    assert batch_size == 1
    f_dim = candidate_feats.shape[1]
    points_num = candidate_xyzs.shape[2]
    max_upsample_num = candidate_xyzs.shape[3]

    # (b, c, n*max_upsample_num)
    candidate_xyzs_trans = candidate_xyzs.view(batch_size, 3, -1)
    candidate_feats_trans  = candidate_feats.view(batch_size, f_dim, -1)

    # (n, max_upsample_num)
    mask = torch.arange(0, max_upsample_num).cuda().view(1, -1).repeat(points_num, 1)
    cur_upsample_num = upsample_num[0].view(-1, 1)
    mask = torch.where(mask >= cur_upsample_num, 0, 1)
    # select the first upsample_num xyzs and feats: (m)
    mask = mask.view(-1)
    # (m)
    selected_idx = torch.where(mask == 1)[0]
    # (b, m)
    selected_idx = selected_idx.unsqueeze(0)
    # (b, c, m)
    selected_xyzs = index_points(candidate_xyzs_trans, selected_idx)
    selected_feats = index_points(candidate_feats_trans, selected_idx)

    return selected_xyzs, selected_feats




def multi_batch_select(candidate_xyzs, candidate_feats, upsample_num, cur_upsample_rate):
    '''
    support multi_batch, but it will degrade the performance!
    :param candidate_xyzs: (b, 3, n, max_upsample_num)
    :param candidate_feats: (b, c, n, max_upsample_num)
    :param upsample_num: (b, n)
    :param cur_upsample_rate
    :return: (b, 3, m), (b, c, m)
    '''
    batch_size = candidate_xyzs.shape[0]
    theoretical_points_num = int(candidate_xyzs.shape[2] * cur_upsample_rate)

    selected_xyzs_list = []
    selected_feats_list = []
    for i in range(batch_size):
        # batch_size is 1
        selected_xyzs, selected_feats = select_xyzs_and_feats(candidate_xyzs[[i]], candidate_feats[[i]], upsample_num[[i]])
        real_points_num = selected_xyzs.shape[2]
        if real_points_num > theoretical_points_num:
            selected_xyzs_trans = selected_xyzs.permute(0, 2, 1).contiguous()
            # FPS, (b, theoretical_points_num)
            sample_idx = pointops.furthestsampling(selected_xyzs_trans, theoretical_points_num).long()
            selected_xyzs = index_points(selected_xyzs, sample_idx)
            selected_feats = index_points(selected_feats, sample_idx)
        elif real_points_num < theoretical_points_num:
            # repeat
            repeat_idx = random.sample(range(selected_xyzs.shape[2]), theoretical_points_num-real_points_num)
            # (1, repeat_num)
            repeat_idx = torch.tensor(repeat_idx).cuda().long().view(1, -1)
            selected_xyzs = torch.cat((selected_xyzs, index_points(selected_xyzs, repeat_idx)), dim=2)
            selected_feats = torch.cat((selected_feats, index_points(selected_feats, repeat_idx)), dim=2)

        selected_xyzs_list.append(selected_xyzs)
        selected_feats_list.append(selected_feats)

    selected_xyzs = torch.cat(selected_xyzs_list, dim=0)
    selected_feats = torch.cat(selected_feats_list, dim=0)

    return selected_xyzs, selected_feats




class UpsampleNumLayer(nn.Module):
    def __init__(self, args, layer_idx):
        super(UpsampleNumLayer, self).__init__()

        self.max_upsample_num = args.max_upsample_num[layer_idx]

        self.upsample_num_nn = nn.Sequential(
            nn.Conv1d(args.dim, args.hidden_dim, 1),
            nn.ReLU(),
            nn.Conv1d(args.hidden_dim, 1, 1),
            nn.Sigmoid()
        )


    def forward(self, feats):
        # (b, 1, n)
        upsample_num = self.upsample_num_nn(feats)
        # (b, n)
        upsample_num = upsample_num.squeeze(1) * (self.max_upsample_num-1)
        # the upsample_num is at least 1
        upsample_num = upsample_num + 1

        return upsample_num




class RefineLayer(nn.Module):
    def __init__(self, args, layer_idx):
        super(RefineLayer, self).__init__()

        self.xyzs_refine_nn = XyzsUpsampleLayer(args, layer_idx, upsample_rate=1)

        # decompress normal
        if args.compress_normal == True and layer_idx == args.layer_num-1:
            self.feats_refine_nn = FeatsUpsampleLayer(args, layer_idx, upsample_rate=1, decompress_normal=True)
        else:
            self.feats_refine_nn = FeatsUpsampleLayer(args, layer_idx, upsample_rate=1, decompress_normal=False)


    def forward(self, xyzs, feats):
        # (b, 3, n, 1)
        refined_xyzs = self.xyzs_refine_nn(xyzs, feats)
        # (b, 3, n)
        refined_xyzs = rearrange(refined_xyzs, "b c n u -> b c (n u)")

        # (b, c, n, 1)
        refined_feats = self.feats_refine_nn(feats)
        # (b, c, n)
        refined_feats = rearrange(refined_feats, "b c n u -> b c (n u)")

        return refined_xyzs, refined_feats




class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()

        self.args = args

        self.decoder_layers = nn.ModuleList([])
        for i in range(args.layer_num):
            self.decoder_layers.append(nn.ModuleList([
                UpsampleLayer(args, i),
                UpsampleNumLayer(args, i),
                RefineLayer(args, i)
            ]))


    def get_mean_distance(self, pre_xyzs, cur_xyzs, upsample_num):
        # input: (b, 3, n), (b, 3, m), (b, n)
        batch_size = pre_xyzs.shape[0]
        pre_xyzs_trans = pre_xyzs.permute(0, 2, 1).contiguous()
        cur_xyzs_trans = cur_xyzs.permute(0, 2, 1).contiguous()
        # find the nearest neighbor in pre_xyzs_trans: (b, m, 1)
        cur2pre_idx = pointops.knnquery_heap(1, pre_xyzs_trans, cur_xyzs_trans)

        # (b, n, k)
        knn_idx = pointops.knnquery_heap(self.args.k, cur_xyzs_trans, pre_xyzs_trans).long()
        torch.cuda.empty_cache()

        # mask: (n)
        expect_center = torch.arange(0, pre_xyzs.shape[2]).cuda()
        # (b, n, k)
        expect_center = repeat(expect_center, 'n -> b n k', b=batch_size, k=self.args.k)
        # (b, 1, n, k)
        real_center = index_points(cur2pre_idx.permute(0, 2, 1).contiguous(), knn_idx)
        # (b, n, k)
        real_center = real_center.squeeze(1)
        # mask those points that not belong to upsampled points set: (b, n, k)
        mask = torch.eq(expect_center, real_center)

        # (b, 3, n, k)
        knn_xyzs = index_points(cur_xyzs, knn_idx)
        # (b, 1, n, k)
        distance = torch.norm(knn_xyzs - pre_xyzs[..., None], p=2, dim=1, keepdim=True)
        # mask_matrix: (b, 1, n, k)
        mask_matrix = mask.unsqueeze(1).float().cuda()
        distance = distance * mask_matrix
        # (b, n)
        distance = distance.sum(dim=-1).squeeze(1)
        # (b, n)
        mean_distance = distance / upsample_num

        return mean_distance


    def get_pred_mdis(self, latent_xyzs, pred_xyzs, pred_unums):
        pred_mdis = []
        pre_xyzs = latent_xyzs
        cur_xyzs = pred_xyzs[0]
        for i in range(self.args.layer_num):
            cur_mdis = self.get_mean_distance(pre_xyzs, cur_xyzs, pred_unums[i])
            pred_mdis.append(cur_mdis)
            pre_xyzs = cur_xyzs
            if i+1 < len(pred_xyzs):
                cur_xyzs = pred_xyzs[i+1]

        return pred_mdis


    def forward(self, xyzs, feats):
        # input: (b, c, n)
        batch_size = xyzs.shape[0]
        if self.args.quantize_latent_xyzs == False:
            xyzs = xyzs.float()
        latent_xyzs = xyzs.clone()

        pred_xyzs = []
        pred_unums = []
        for i, (upsample_nn, upsample_num_nn, refine_nn) in enumerate(self.decoder_layers):
            # upsample xyzs and feats: (b, c, n u)
            candidate_xyzs, candidate_feats = upsample_nn(xyzs, feats)

            # predict upsample_num: (b, n)
            upsample_num = upsample_num_nn(feats)
            pred_unums.append(upsample_num)

            # select the first upsample_num xyzs and feats: (b, c, m)
            if batch_size == 1:
                xyzs, feats = select_xyzs_and_feats(candidate_xyzs, candidate_feats, upsample_num)
            else:
                cur_upsample_rate = 1 / self.args.downsample_rate[self.args.layer_num-1-i]
                xyzs, feats = multi_batch_select(candidate_xyzs, candidate_feats, upsample_num, cur_upsample_rate)

            # refine xyzs and feats: (b, c, m)
            xyzs, feats = refine_nn(xyzs, feats)

            pred_xyzs.append(xyzs)

        # get mean distance in upsampled points set
        pred_mdis = self.get_pred_mdis(latent_xyzs, pred_xyzs, pred_unums)

        return pred_xyzs, pred_unums, pred_mdis, feats
