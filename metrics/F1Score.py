import torch
import numpy as np
from models.pointops.functions import pointops




def get_f1_score(gt_xyzs, gt_normals, pred_xyzs, pred_normals, args):
    # input: (1, n, 3)

    gt_num = gt_xyzs.shape[1]
    pred_num = pred_xyzs.shape[1]
    # (m)
    pred2gt_idx = pointops.knnquery_heap(1, gt_xyzs, pred_xyzs).view(-1).long()

    # (n, 3)
    gt_xyzs = gt_xyzs.squeeze(0)
    gt_normals = gt_normals.squeeze(0)
    # (m, 3)
    pred_xyzs = pred_xyzs.squeeze(0)
    pred_normals = pred_normals.squeeze(0)
    # (m, 3)
    nearest_xyzs = gt_xyzs[pred2gt_idx, :]
    nearest_normals = gt_normals[pred2gt_idx, :]

    # (m)
    xyzs_dist = torch.norm(nearest_xyzs - pred_xyzs, p=2, dim=-1)
    normals_dist = torch.norm(nearest_normals - pred_normals, p=2, dim=-1)

    tp_xyzs = torch.where(xyzs_dist <= args.omega_xyzs)[0].detach().cpu().numpy()
    tp_normals = torch.where(normals_dist <= args.omega_normals)[0].detach().cpu().numpy()

    # the intersection of tp_xyzs and tp_normals
    pred_tp_idx = np.intersect1d(tp_xyzs, tp_normals)
    pred_tp_idx = torch.from_numpy(pred_tp_idx).long().cuda()
    gt_tp_idx = pred2gt_idx[pred_tp_idx]
    gt_tp_idx = torch.unique(gt_tp_idx)

    tp = pred_tp_idx.shape[0]
    fp = pred_num - tp
    fn = gt_num - gt_tp_idx.shape[0]

    f1_score = (2*tp) / (2*tp+fp+fn)

    return f1_score
