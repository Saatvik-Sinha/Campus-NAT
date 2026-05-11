"""
Paris-Lille-3D Dataset
.npy files have columns: [x, y, z, class]
Train: Lille1_1, Lille1_2, Lille2
Test:  Paris
"""
import os
import numpy as np
import pickle
import logging
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from ..build import DATASETS
from ..data_util import crop_pc, voxelize

logger = logging.getLogger(__name__)

# 9 semantic classes
CLASSES = [
    'ground', 'building', 'pole', 'bollard', 'trash_can',
    'barrier', 'pedestrian', 'car', 'natural'
]

TRAIN_FILES = ['Lille1_1', 'Lille1_2', 'Lille2']
TEST_FILES  = ['Paris']


@DATASETS.register_module()
class ParisLille3D(Dataset):
    num_classes = 9
    classes = CLASSES
    gravity_dim = 2  # z is up

    def __init__(self,
                 data_root,
                 split='train',
                 voxel_size=0.04,
                 voxel_max=None,
                 transform=None,
                 loop=1,
                 **kwargs):
        super().__init__()
        self.data_root = data_root
        self.split = split
        self.voxel_size = voxel_size
        self.voxel_max = voxel_max
        self.transform = transform
        self.loop = loop

        raw_root = os.path.join(data_root, 'raw')
        processed_root = os.path.join(data_root, 'processed')
        os.makedirs(processed_root, exist_ok=True)

        # Decide which files belong to this split
        if split == 'train':
            file_stems = TRAIN_FILES
        else:
            file_stems = TEST_FILES

        # Load or preprocess each file
        processed_path = os.path.join(
            processed_root,
            f'parislille3d_{split}_{voxel_size}.pkl'
        )

        if os.path.exists(processed_path):
            with open(processed_path, 'rb') as f:
                self.data = pickle.load(f)
            logger.info(f'{processed_path} loaded successfully')
        else:
            self.data = []
            for stem in tqdm(file_stems,
                             desc=f'Loading ParisLille3D {split} split'):
                npy_path = os.path.join(raw_root, f'{stem}.npy')
                raw = np.load(npy_path).astype(np.float32)  # (N, 4)
                coord = raw[:, :3]
                label = raw[:, 3].astype(np.int64)

                # Voxel downsample the full cloud
                if voxel_size is not None:
                    idx = voxelize(coord, voxel_size)
                    coord = coord[idx]
                    label = label[idx]

                self.data.append((coord, label))

            with open(processed_path, 'wb') as f:
                pickle.dump(self.data, f)
            logger.info(f'{processed_path} saved successfully')

        logger.info(
            f'\nTotally {len(self.data)} samples in {split} set'
        )
        logger.info(f'length of {split} dataset: {len(self.data) * loop}')
        logger.info(f'number of classes of the dataset: {self.num_classes}')

    def __len__(self):
        return len(self.data) * self.loop

    def __getitem__(self, idx):
        idx = idx % len(self.data)
        coord, label = self.data[idx]

        # Random crop to voxel_max points
        coord, feat, label = crop_pc(
            coord, None, label,
            self.split, self.voxel_size,
            self.voxel_max,
            variable=True
        )

        # Use xyz as features (no color available)
        feat = coord.copy()

        if self.transform is not None:
            coord, feat, label = self.transform(coord, feat, label)

        coord = torch.from_numpy(coord.astype(np.float32))
        feat  = torch.from_numpy(feat.astype(np.float32))
        label = torch.from_numpy(label.astype(np.int64))

        return 'parislille3d', 'dataset', dict(
            pos=coord, x=feat, y=label
        )
