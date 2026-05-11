# Campus-NAT
PointNAT implementation for IIIT-B Campus Point Cloud benchmarking dataset

## Setup
- Install packages with a setup file
```
source install-fixed.sh
```

Or better
```
conda env create -f environment.yml -n openpoints
```

## Train
```
CUDA_VISIBLE_DEVICES=0 python examples/segmentation/main.py --cfg cfgs/parislille3d/pointnat.yaml wandb.use_wandb=True batch_size=2
```
