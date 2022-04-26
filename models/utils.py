import numpy as np
import os
import torch
from einops import rearrange
import open3d as o3d
import random
import argparse




class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = 1.0 * self.sum / self.count

    def get_avg(self):
        return self.avg

    def get_count(self):
        return self.count




def index_points(xyzs, idx):
    """
    Input:
        xyzs: input points data, [B, C, N]
        idx: sample index data, [B, S, [K]]
    Return:
        new_points:, indexed points data, [B, C, S, [K]]
    """
    batch_size = idx.shape[0]
    sample_num = idx.shape[1]
    fdim = xyzs.shape[1]

    reshape = False
    if len(idx.shape) == 3:
        reshape = True
        idx = idx.reshape(batch_size, -1)

    # (b, c, (s k))
    res = torch.gather(xyzs, 2, idx[:, None].repeat(1, fdim, 1))

    if reshape:
        res = rearrange(res, 'b c (s k) -> b c s k', s=sample_num)

    return res




def save_pcd(dir, name, xyzs, normals=None):
    # input: (n, 3)
    path = os.path.join(dir, name)
    f = open(path, 'w')
    f.write("ply\n")
    f.write("format ascii 1.0\n")
    f.write("element vertex " + str(xyzs.shape[0]) + "\n")
    f.write("property float x\n")
    f.write("property float y\n")
    f.write("property float z\n")
    if isinstance(normals, np.ndarray):
        f.write("property float nx\n")
        f.write("property float ny\n")
        f.write("property float nz\n")
    f.write("element face 0\n")
    f.write("property list uchar int vertex_indices\n")
    f.write("end_header\n")
    f.close()

    with open(path, 'ab') as f:
        if isinstance(normals, np.ndarray):
            # (n, 6)
            xyzs_and_normals = np.concatenate((xyzs, normals), axis=1)
            np.savetxt(f, xyzs_and_normals, fmt='%s')
        else:
            np.savetxt(f, xyzs, fmt='%s')




def str2bool(val):
    if isinstance(val, bool):
        return val
    if val.lower() in ['yes', 'true', 't', 'y']:
        return True
    elif val.lower() in ['no', 'false', 'f', 'n']:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
