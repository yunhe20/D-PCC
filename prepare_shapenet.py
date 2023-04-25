import os
import sys
sys.path.append(os.getcwd())
import random
from sklearn.model_selection import train_test_split
import numpy as np
import pickle as pkl
import open3d as o3d
from dataset.sample_points import SampleMethod
from models.utils import save_pcd
from collections import Counter
import argparse




def convert_to_mesh(instance_dir, sample_strategy, simplify=False, output_dir='./data/shapenet/mesh'):
    os.makedirs(output_dir, exist_ok=True)
    manifold_script_path = os.path.join(MANIFOLD_DIR, 'manifold')
    simplify_sciprt_path = os.path.join(MANIFOLD_DIR, 'simplify')
    for instance in instance_dir:
        prefix_name = instance.split('/')[-1]
        instance_path = os.path.join(instance, 'model.obj')
        output_path = os.path.join(output_dir, prefix_name + '.obj')
        if sample_strategy[prefix_name] == 'montecarlo_sample':
            if not simplify:
                # generate watertight mesh
                cmd = "{} {} {} {}".format(manifold_script_path, instance_path, output_path, 20000)
                print(cmd)
                os.system(cmd)
            else:
                # the input mesh should be watertight
                mesh = o3d.io.read_triangle_mesh(instance_path)
                assert mesh.is_edge_manifold() == True

                cmd = "{} -i {} -o {} -f {}".format(simplify_sciprt_path, instance_path, output_path, 40000)
                print(cmd)
                os.system(cmd)
        else:
            # directly use the cad model
            cad_model = o3d.io.read_triangle_mesh(instance_path)
            o3d.io.write_triangle_mesh(output_path, cad_model, write_vertex_colors=False, write_triangle_uvs=False)


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


def get_instance_dir(data_root, instance_num):
    data_dir = [os.path.join(data_root, dir) for dir in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, dir))]
    instance_dir = []
    for dir in data_dir:
        full_instance_name = os.listdir(dir)
        if len(full_instance_name) > instance_num:
            instance_name = random.sample(full_instance_name, instance_num)
        else:
            instance_name = full_instance_name
        instance_dir += [os.path.join(dir, name) for name in instance_name]
    return instance_dir


def get_sample_strategy(instance_dir, split_rate):
    sample_strategy = {}
    montecarlo_sample_instance = set(random.sample(instance_dir, int(len(instance_dir)*split_rate)))
    for instance_path in instance_dir:
        instance_name = instance_path.split('/')[-1]
        if instance_path in montecarlo_sample_instance:
            cur_sample_strategy = 'montecarlo_sample'
        else:
            cur_sample_strategy = 'subdivide'
        sample_strategy[instance_name] = cur_sample_strategy
    return sample_strategy


def sample_pcd_from_mesh(mesh_dir, pcd_dir, sample_strategy):
    mesh_name = os.listdir(mesh_dir)
    os.makedirs(pcd_dir, exist_ok=True)
    for name in mesh_name:
        if not name.endswith('.obj'):
            continue
        prefix_name = os.path.splitext(name)[0]
        mesh_path = os.path.join(mesh_dir, name)
        mesh = o3d.io.read_triangle_mesh(mesh_path)
        # current sample strategy
        cur_sample_strategy = sample_strategy[prefix_name]
        if cur_sample_strategy == 'montecarlo_sample':
            mesh.compute_vertex_normals(normalized=True)
            pcd = mesh.sample_points_uniformly(200000)
        else:
            subdivide_mesh = mesh.subdivide_midpoint(3)
            # the normals may be [0, 0, 0]
            subdivide_mesh.compute_vertex_normals(normalized=True)
            pcd = o3d.geometry.PointCloud()
            pcd.points = subdivide_mesh.vertices
            pcd.normals = subdivide_mesh.vertex_normals
        save_pcd(pcd_dir, prefix_name + '.ply', np.asarray(pcd.points), np.asarray(pcd.normals))
        # o3d.io.write_point_cloud(os.path.join(pcd_dir, prefix_name + '.ply'), pcd)


def generate_path_txt(pcd_dir, output_dir, sample_strategy):
    pcd_name = os.listdir(pcd_dir)
    montecarlo_sample_path = [os.path.join(pcd_dir, p)+' montecarlo_sample\n' for p in pcd_name if sample_strategy[os.path.splitext(p)[0]]=='montecarlo_sample']
    subdivide_path = [os.path.join(pcd_dir, p)+' subdivide\n' for p in pcd_name if sample_strategy[os.path.splitext(p)[0]]=='subdivide']
    # train: 7 test: 2 val: 1
    montecarlo_sample_train_path, montecarlo_sample_test_path = train_test_split(montecarlo_sample_path, test_size=0.3)
    montecarlo_sample_test_path, montecarlo_sample_val_path = train_test_split(montecarlo_sample_test_path, test_size=1/3)
    subdivide_train_path, subdivide_test_path = train_test_split(subdivide_path, test_size=0.3)
    subdivide_test_path, subdivide_val_path = train_test_split(subdivide_test_path, test_size=1/3)
    train_path = montecarlo_sample_train_path + subdivide_train_path
    test_path = montecarlo_sample_test_path + subdivide_test_path
    val_path = montecarlo_sample_val_path + subdivide_val_path
    # generate txt file
    modes = ['train', 'test', 'val']
    for mode in modes:
        with open(os.path.join(output_dir, mode + '.txt'), 'w')as f:
            f.writelines(eval(mode+'_path'))


def divide_cube(xyzs, attribute, map_size=100, cube_size=10, min_num=100, max_num=30000, cur_sample_strategy=None):
    '''
    points : N x 3
    resolution: 100 x 100 x 100
    cube_size: 10 x 10 x 10
    min_num and max_num points in each cube, if small than min_num or larger than max_num then discard it
    cur_sample_strategy: montecarlo_sample or subdivide
    return label that indicates each points' cube_idx
    '''
    assert cur_sample_strategy in ['montecarlo_sample', 'subdivide']
    Sample = SampleMethod(rate=0.7)

    output_points = {}
    xyzs , meta_data= normalize_pcd(xyzs)

    map_xyzs = xyzs * (map_size)
    xyzs = np.floor(map_xyzs).astype('float32')
    label = np.zeros(xyzs.shape[0]).astype(int) - 1

    cubes = {}
    for idx, p in enumerate(xyzs):
        tuple_cube_idx = tuple((p//cube_size).astype(int))
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
        if len(cubes[k]) > max_num:
            label[cubes[k]] = del_k
            del_k -= 1
            k_del.append(k)
    for k in k_del:
        del cubes[k]

    double_check = []
    for idx, p in cubes.items():
        if len(p) < min_num:
            # print('del ', len(p))
            double_check.append(idx)

    for i in double_check:
        del cubes[i]

    for tuple_cube_idx, point_idx in cubes.items():
        dim_cube_num = np.ceil(map_size/cube_size).astype(int)
        # indicate which cube a point belongs to
        cube_idx = tuple_cube_idx[0] * dim_cube_num * dim_cube_num + \
                  tuple_cube_idx [1] * dim_cube_num + tuple_cube_idx[2]
        label[point_idx] = cube_idx

        if cur_sample_strategy == 'montecarlo_sample':
            tmp_p, tmp_normal = map_xyzs[point_idx, :], attribute[point_idx, :]
            if tmp_p.shape[0] < min_num:
                print(tmp_p.shape)
                raise ValueError
            # dim = new_points.shape[-1]
            final_points = Sample.random_sample(tmp_p, tmp_normal, min_point=min_num)
            output_points[cube_idx] = final_points
            if final_points.shape[0] < min_num:
                print(final_points.shape)
                raise ValueError
        else:
            output_points[cube_idx] = np.concatenate([map_xyzs[point_idx, :], attribute[point_idx, :]], -1)

    # label indicates each points' cube index
    # label may have some values less than 0, which should be ignored
    num = [len(k)<min_num for k in output_points.values()]
    if sum(num) > 0:
        print(sorted([len(k) for k in output_points.values()]))
        raise ValueError

    return label, output_points, meta_data


def generate_dataset(mode='train', dataset_name='shapenet', cube_size=20, min_num=100, max_num=100000, save_path='./data/shapenet'):
    path_txt = os.path.join(save_path, f'{mode}.txt')
    with open(path_txt, 'r')as f:
        pcd_item = f.readlines()

    data = {}
    invalid_path =[]
    idx = 0
    patch_num = 0
    for item in pcd_item:
        path, cur_sample_strategy = item.strip('\n').split(' ')

        pcd = o3d.io.read_point_cloud(path)
        xyzs = np.array(pcd.points)
        normals = np.array(pcd.normals)

        mask, points, meta_data = divide_cube(xyzs, attribute=normals, cube_size=cube_size, min_num=min_num,
                                              max_num=max_num, cur_sample_strategy=cur_sample_strategy)
        key = Counter(mask)
        # print(len(np.unique(mask)))
        print('----------------')
        print('valid patch number:', len(points))
        k = np.array(list(key.values()))
        if len(points) == 0:
            print(20 * '***')
            invalid_path.append(path)
            continue

        data[idx] = {}
        data[idx]['points'] = points
        data[idx]['meta_data'] = meta_data
        idx += 1
        patch_num += len(points)

    with open(os.path.join(save_path, f'{dataset_name}_{mode}_cube_size_{cube_size}.pkl'), 'wb')as f:
        pkl.dump(data, f, protocol=2)


def parse_dataset_args():
    parser = argparse.ArgumentParser(description='Shapenet Dataset Arguments')

    # data root
    parser.add_argument('--data_root', default='path to ShapeNet core dataset', type=str, help='dir of shapenet core dataset')
    # instance number of each class
    parser.add_argument('--instance_num', default=50, type=int, help='instance number of each class')
    # split rate for montecarlo sampling and subdivision
    parser.add_argument('--split_rate', default=0.8, type=float, help='split rate for montecarlo sampling and subdivision')
    # save dir for watertight mesh
    parser.add_argument('--output_mesh_dir', default='./data/shapenet/mesh', type=str, help='save dir for watertight mesh')
    # save dir for sampled point cloud
    parser.add_argument('--output_pcd_dir', default='./data/shapenet/pcd', type=str, help='save dir for sampled point cloud')
    # cube size
    parser.add_argument('--cube_size', default=22, type=int, help='cube size')
    # minimum points number in each cube when training
    parser.add_argument('--train_min_num', default=1024, type=int, help='minimum points number in each cube when training')
    # minimum points number in each cube when testing
    parser.add_argument('--test_min_num', default=100, type=int, help='minimum points number in each cube when testing')
    # maximum points number in each cube
    parser.add_argument('--max_num', default=500000, type=int, help='maximum points number in each cube')

    args = parser.parse_args()
    return args




if __name__ == '__main__':
    MANIFOLD_DIR = './Manifold/build/'
    dataset_args = parse_dataset_args()
    dataset_args.data_root = os.path.join(dataset_args.data_root, 'ShapeNetCore.v1')
    assert  dataset_args.data_root != None and os.path.exists(dataset_args.data_root)

    # 1. first generate mesh
    instance_dir = get_instance_dir(dataset_args.data_root, instance_num=dataset_args.instance_num)
    sample_strategy = get_sample_strategy(instance_dir, dataset_args.split_rate)
    convert_to_mesh(instance_dir, sample_strategy, simplify=False, output_dir=dataset_args.output_mesh_dir)

    # 2. sample pcd from mesh, montecarlo sampling or subdivision
    sample_pcd_from_mesh(dataset_args.output_mesh_dir, dataset_args.output_pcd_dir, sample_strategy)

    # 3. generate data path list
    generate_path_txt(dataset_args.output_pcd_dir, './data/shapenet', sample_strategy)

    # 4. generate dataset
    generate_dataset('train', 'shapenet', cube_size=dataset_args.cube_size, min_num=dataset_args.train_min_num,
                     max_num=dataset_args.max_num, save_path='./data/shapenet')
    generate_dataset('val', 'shapenet', cube_size=dataset_args.cube_size, min_num=dataset_args.train_min_num,
                     max_num=dataset_args.max_num, save_path='./data/shapenet')
    generate_dataset('test', 'shapenet', cube_size=dataset_args.cube_size, min_num=dataset_args.test_min_num,
                     max_num=dataset_args.max_num, save_path='./data/shapenet')
