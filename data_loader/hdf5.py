import torch
from torch.utils.data import Dataset
import h5py
import logging
from time import time

logger = logging.getLogger(__name__)


class NonInteractiveDataset(Dataset):
    def __init__(self, dataset_path, keypoint_dim, is_testing, use_mask, is_visualizing, use_quaternion):
        self.data = h5py.File(dataset_path, 'r')
        self.keypoint_dim = keypoint_dim
        self.is_testing = is_testing
        self.use_mask = use_mask
        self.is_visualizing = is_visualizing
        self.use_quaternion = use_quaternion

        assert 'observed_pose' in self.data.keys(), 'dataset must include observed_pose'
        self.keypoints_num = int(self.data['observed_pose']['0'].shape[-1] / self.keypoint_dim)
        self.obs_frames_num = self.data['observed_pose']['0'].shape[-2]
        if not self.is_testing:
            assert 'future_pose' in self.data.keys(), 'dataset must include future_pose'
            self.future_frames_num = self.data['future_pose']['0'].shape[-2]

    def __len__(self):
        return len(self.data['observed_pose'])

    def __getitem__(self, index):
        seq = str(index)

        outputs_keys = ['observed_pose']
        if self.use_mask:
            outputs_keys.append('observed_mask')
        if self.use_quaternion:
            outputs_keys.append('observed_quaternion_pose')
        if not self.is_testing:
            outputs_keys.append('future_pose')
            if self.use_mask:
                outputs_keys.append('future_mask')
            if self.use_quaternion:
                outputs_keys.append('future_quaternion_pose')

        outputs = dict()
        for key in outputs_keys:
            if key in self.data.keys():
                outputs[key] = torch.tensor(self.data[key][seq], dtype=torch.float)

        if self.is_visualizing:
            if 'observed_image_path' in self.data.keys():
                outputs['observed_image'] = self.data['observed_image_path'][seq]
            if 'future_image_path' in self.data.keys():
                outputs['future_image'] = self.data['future_image_path'][seq]
            if 'observed_cam_extrinsic' in self.data.keys():
                outputs['observed_cam_ex'] = torch.tensor(self.data['observed_cam_extrinsic'][seq], dtype=torch.float)
            if 'future_cam_extrinsic' in self.data.keys():
                outputs['future_cam_ex'] = torch.tensor(self.data['future_cam_extrinsic'][seq], dtype=torch.float)
            if 'cam_intrinsic' in self.data.keys():
                outputs['cam_in'] = torch.tensor(self.data['cam_intrinsic'][seq], dtype=torch.float)

        return outputs


# a = torch.tensor(f['observed_pose']['1'])
start = time()

a = NonInteractiveDataset('../preprocessed_data/train_16_14_1_JTA.h5', 3, False, False, False, False)

print('time:', time() - start)
