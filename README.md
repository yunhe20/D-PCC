# DPCC

 [arXiv](https://arxiv.org/abs/1912.03264.pdf) | [Project Page](https://yunhe20.github.io/DPCC) | [Code](https://github.com/yunhe20/DPCC) 

This is the PyTorch implementation of "Density-preserving Deep Point Cloud Compression" (CVPR 2022).



## Installation
* Install the following packages
```
python==3.7.12
torch==1.7.1
CUDA==11.0
numpy==1.20.3
open3d==0.9.0.0
einops==0.3.2
scikit-learn==1.0.1
pickle
argparse
```
* Install the built-in libraries
```
cd models/Chamfer3D
python setup.py
cd ../pointops
python setup.py
```    
These commands are tested on an ubuntu 16.04 system.

## Data Preparation 
First download the [ShapeNetCore](https://shapenet.org/download/shapenetcore) v1 and [SemanticKITTI](http://semantic-kitti.org/dataset.html#download) datasets, and then divide them into non-overlapping blocks.

* ShapeNet
```
# install the `Manifold' program
cd ./dataset
git clone https://github.com/hjwdzh/Manifold
cd Manifold && mkdir build
cd build 
cmake .. -DCMAKE_BUILD_TYPE=Release
make 
cd ..

# divide into blocks
python prepare_shapenet.py --date_root path/to/shapenet
```

* SemanticKITTI
```
python prepare_semantickitti.py --data_root path/to/semantickitti
```

Please refer to the associated code files for the detailed usages and meanings of other arguments (e.g. `cube_size`, `max_num`, etc), and you can adjust them by yourself.

The final file structure is shown as follow:
```
data  
└───semantickitti
│   │   semantickitti_test_xxx.pkl 
│   │   semantickitti_train_xxx.pkl
│   │   semantickitti_val_xxx.pkl
└───shapenet
│   │   mesh # watertight meshes or cad models
│   │   pcd # sampled point clouds
│   │   shapenet_test_xxx.pkl
│   │   shapenet_train_xxx.pkl
│   │   shapenet_val_xxx.pkl
│   │   test.txt
│   │   train.txt
│   │   val.txt
```
    
## Train
* Position Compression
```
# shapenet
python train.py --dataset shapenet
# semantickitti
python train.py --dataset semantickitti
```

* Normal Compression
```
# shapenet
python train.py --dataset shapenet --compress_normal True
# semantickitti
python train.py --dataset semantickitti --compress_normal True
```
You can manually adjust the `downsample_rate`, `bpp_lambda` and `quantize_latent_xyzs` arguments to achieve different compression ratios, please refer to the `./args/semantickitti_args.py` and `./args/shapenet_args.py` files for the details.
The output files will be saved at `./output/experiment_id/ckpt` by default.

## Test
* Position Compression
```
# shapenet
python test.py --dataset shapenet --model_path path/to/checkpoint
# semantickitti
python test.py --dataset semantickitti --model_path path/to/checkpoint
```

* Normal Compression
```
# shapenet
python test.py --dataset shapenet --compress_normal True --model_path path/to/checkpoint
# semantickitti
python test.py --dataset semantickitti --compress_normal True --model_path path/to/checkpoint
```

The decompressed patches and full point clouds will also be saved at `./output/experiment_id/pcd` by default.

## Acknowledgments

Our code is built upon the following repositories: [DEPOCO](https://github.com/PRBonn/deep-point-map-compression), [PAConv](https://github.com/CVMI-Lab/PAConv), [Point Transformer](https://github.com/qq456cvb/Point-Transformers) and [MCCNN](https://github.com/viscom-ulm/MCCNN), thanks for their great work.


## Citation

If you find our project is useful, please consider citing:

<!-- ```
@inProceedings{wei2020deepsfm,
  title={DeepSFM: Structure From Motion Via Deep Bundle Adjustment},
  author={Xingkui Wei and Yinda Zhang and Zhuwen Li and Yanwei Fu and Xiangyang Xue},
  booktitle={ECCV},
  year={2020}
}
``` -->

