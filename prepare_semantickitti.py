import torch
import pickle as pkl
import math
import os
import random
import argparse
import numpy as np
import open3d as o3d
from collections import Counter
import os
from sklearn.neighbors import KDTree
from sklearn.model_selection import train_test_split




def generate_path_list(ply_path, output_path, model_name):
    data_path = os.listdir(ply_path)
    data_path = [os.path.join(ply_path, p+'\n') for p in data_path]
    train_path, test_path = train_test_split(data_path, test_size=0.4)
    test_path, val_path = train_test_split(test_path, test_size=0.3)
    modes = ['train', 'test', 'val']
    for mode in modes:
        with open(os.path.join(output_path, f'{model_name}_'+mode+'.txt'), 'w')as f:
            f.writelines(eval(mode+'_path'))


def load_pcd(path, dataset_name='semantickitti'):
    assert dataset_name == 'semantickitti'
    # xyz + intensity
    points = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
    return points


def search_path(data_root, seq):
    seq_dir = [os.path.join(data_root, s, 'velodyne') for s in seq if os.path.isdir(os.path.join(data_root, s))]
    pcd_path = []
    for dir in seq_dir:
        cur_pcd_path = [os.path.join(dir, p) for p in os.listdir(dir) if p.endswith('.bin')]
        pcd_path += cur_pcd_path
    return pcd_path


def normalize_pcd(xyzs):
    '''
     normalize xyzs to [0,1], keep ratio unchanged
    '''
    shift = np.mean(xyzs, axis=0)
    xyzs -= shift
    max_coord, min_coord = np.max(xyzs), np.min(xyzs)
    xyzs = xyzs - min_coord
    xyzs = xyzs / (max_coord - min_coord)
    meta_data = {}
    meta_data['shift'] = shift
    meta_data['max_coord'] = max_coord
    meta_data['min_coord'] = min_coord
    return xyzs, meta_data


def divide_cube(xyzs, attribute, map_size=100, cube_size=10, min_num=100, max_num=30000, sample_num=None):
    '''
    xyzs: N x 3
    resolution: 100 x 100 x 100
    cube_size: 10 x 10 x 10
    min_num and max_num points in each cube, if small than min_num or larger than max_num then discard it
    return label that indicates each points' cube_idx
    '''

    output_points = {}
    # points = np.dot(points, get_rotate_matrix())
    xyzs, meta_data = normalize_pcd(xyzs)

    map_xyzs = xyzs * (map_size)
    xyzs = np.floor(map_xyzs).astype('float32')
    label = np.zeros(xyzs.shape[0]).astype(int) - 1

    cubes = {}
    for idx, point_idx in enumerate(xyzs):
        # the cube_idx is a 3-dim tuple
        tuple_cube_idx = tuple((point_idx//cube_size).astype(int))
        if not tuple_cube_idx in cubes.keys():
            cubes[tuple_cube_idx] = []
        cubes[tuple_cube_idx].append(idx)

    # remove those cubes whose points_num is small than min_num
    del_k = -1
    k_del = []
    for k in cubes.keys():
        if len(cubes[k]) < min_num:
            label[cubes[k]] = del_k
            del_k -= 1
            k_del.append(k)
        if len(cubes[k]) >max_num:
            label[cubes[k]] = del_k
            del_k -= 1
            k_del.append(k)
    for k in k_del:
        del cubes[k]

    for tuple_cube_idx, point_idx in cubes.items():
        dim_cube_num = np.ceil(map_size/cube_size).astype(int)
        # indicate which cube a point belongs to
        cube_idx = tuple_cube_idx[0] * dim_cube_num * dim_cube_num + \
                  tuple_cube_idx [1] * dim_cube_num + tuple_cube_idx[2]
        label[point_idx] = cube_idx

        if sample_num is not None and type(sample_num)==int :
            # dim = new_points.shape[-1]
            tmp_points = np.concatenate([map_xyzs[point_idx, :], attribute[point_idx, :]], axis=-1)
            # print(tmp_points[0,:3].shape)
            kdt = KDTree(tmp_points[:,:3])

            reversed_idx = kdt.query(tmp_points[0,:3].reshape(1,-1),k=tmp_points.shape[0]//2, return_distance=False)[0]
            reversed_points = tmp_points[reversed_idx, :]

            need_sample_points = tmp_points[np.setdiff1d(np.arange(tmp_points.shape[0]), reversed_idx), :]
            sample_idx = np.random.choice(np.arange(need_sample_points.shape[0]), need_sample_points.shape[0] // 10)

            resample_points = need_sample_points[sample_idx,:]
            output_points[cube_idx] = np.concatenate([reversed_points, resample_points], axis=0)

            print('reversed points {}, resample points {}'.format(reversed_points.shape[0], resample_points.shape[0]))

        else:
            output_points[cube_idx] = np.concatenate([map_xyzs[point_idx, :], attribute[point_idx, :]], axis=-1)

    # label indicates each points' cube index
    # label may have some value less than 0, which should be ignored
    return label, output_points, meta_data


def generate_dataset(path_list, dataset_name='semantickitti', mode='train', cube_size=20, sample_num=None, min_num=15000, max_num=100000, save_path='./data/semantickitti'):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    data = {}
    idx = 0
    patch_num = 0
    print(len(path_list))
    path_list = [p.strip('\n') for p in path_list]
    for path in path_list:
            if path.endswith('.ply') or path.endswith('.xyz') or path.endswith('.bin'):
                points = load_pcd(path, dataset_name)
                xyzs = points[:,:3]
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(xyzs)
                # estimate the normal
                pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=12))
                pcd.normalize_normals()
                normals = np.array(pcd.normals)

                mask, points, meta_data= divide_cube(xyzs, attribute=normals, cube_size=cube_size, min_num=min_num, max_num=max_num, sample_num=sample_num)
                key = Counter(mask)
                # print(len(np.unique(mask)))
                print('----------------')
                print('valid patch number:', len(points))
                k = np.array(list(key.values()))
                if len(points) == 0:
                    print(20*'***')
                    continue

                data[idx] = {}
                data[idx]['points'] = points
                data[idx]['meta_data'] = meta_data
                # data[index] = points
                idx += 1
                patch_num += len(points)

    with open(os.path.join(save_path, dataset_name + '_{}_cube_size_{}.pkl'.format(mode, cube_size)), 'wb')as f:
        pkl.dump(data, f, protocol=2)


def parse_dataset_args():
    parser = argparse.ArgumentParser(description='SemanticKITTI Dataset Arguments')

    # data root
    parser.add_argument('--data_root', default=None, type=str, help='dir of semantickitti dataset')
    # cube size
    parser.add_argument('--cube_size', default=12, type=int, help='cube size')
    # minimum points number in each cube when training
    parser.add_argument('--train_min_num', default=1024, type=int, help='minimum points number in each cube when training')
    # minimum points number in each cube when testing
    parser.add_argument('--test_min_num', default=100, type=int, help='minimum points number in each cube when testing')
    # maximum points number in each cube
    parser.add_argument('--max_num', default=500000, type=int, help='maximum points number in each cube')

    args = parser.parse_args()
    return args




if __name__ == '__main__':
    dataset_args = parse_dataset_args()
    dataset_args.data_root = os.path.join(dataset_args.data_root, 'dataset/sequences')
    assert  dataset_args.data_root != None and os.path.exists(dataset_args.data_root)

    # 1. get pcd path
    train_seq = ['00', '01', '02', '03', '04', '05', '06', '07', '09', '10']
    test_seq = ['08']
    train_path = search_path(dataset_args.data_root, train_seq)
    train_path, val_path = train_test_split(train_path, test_size=0.045)
    test_path = search_path(dataset_args.data_root, test_seq)

    # 2. generate dataset
    generate_dataset(train_path, 'semantickitti', 'train', cube_size=dataset_args.cube_size, min_num=dataset_args.train_min_num,
                     max_num=dataset_args.max_num, save_path='./data/semantickitti')
    generate_dataset(val_path, 'semantickitti', 'val', cube_size=dataset_args.cube_size, min_num=dataset_args.train_min_num,
                     max_num=dataset_args.max_num, save_path='./data/semantickitti')
    generate_dataset(test_path, 'semantickitti', 'test', cube_size=dataset_args.cube_size, min_num=dataset_args.test_min_num,
                     max_num=dataset_args.max_num, save_path='./data/semantickitti')
