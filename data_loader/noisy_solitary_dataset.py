import os
import logging

import json
import jsonlines
import torch
import numpy as np
from torch.utils.data import Dataset

from path_definition import PREPROCESSED_DATA_DIR

logger = logging.getLogger(__name__)


dim_used = np.array([6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 51, 52, 53, 54, 55, 56, 57, 58, 59, 63, 64, 65, 66, 67, 68, 75, 76, 77, 78, 79, 80, 81, 82, 83, 87, 88, 89, 90, 91, 92])

class NoisySolitaryDataset(Dataset):
    def __init__(self, dataset_path, keypoint_dim, is_testing, use_mask, is_visualizing, use_quaternion, normalize,
                 metadata_path, noise_rate, noise_keypoint=None, overfit=None):

        self.normalize = normalize
        if normalize:
            assert metadata_path, "Specify metadata_path when normalize is true."
            with open(os.path.join(PREPROCESSED_DATA_DIR, metadata_path)) as meta_file:
                meta_data = json.load(meta_file)
                self.mean_pose = list(meta_data['avg_pose'])
                self.std_pose = list(meta_data['std_pose'])
        else:
            self.mean_pose = None
            self.std_pose = None

        data = list()
        tensor_keys = ['observed_pose', 'future_pose', 'observed_mask', 'future_mask']
        with jsonlines.open(dataset_path) as reader:
            for seq in reader:
                seq_tensor = {}
                for k, v in seq.items():
                    if k in tensor_keys:
                        seq_tensor[k] = torch.tensor(v, dtype=torch.float32)
                    else:
                        seq_tensor[k] = v
                data.append(seq_tensor)

        if overfit is not None:
            data = [data[overfit]]

        self.data = data
        self.keypoint_dim = keypoint_dim
        self.is_testing = is_testing
        self.use_mask = use_mask
        self.is_visualizing = is_visualizing
        self.use_quaternion = use_quaternion

        seq = self.data[0]
        assert 'observed_pose' in seq.keys(), 'dataset must include observed_pose'
        self.keypoints_num = int(seq['observed_pose'].shape[-1] / self.keypoint_dim)
        self.obs_frames_num = seq['observed_pose'].shape[-2]
        if not self.is_testing:
            assert 'future_pose' in seq.keys(), 'dataset must include future_pose'
            self.future_frames_num = seq['future_pose'].shape[-2]

        self.noise = None
        if isinstance(noise_rate, float) or isinstance(noise_rate, int):
            self.noise = torch.FloatTensor(len(data), self.obs_frames_num, self.keypoints_num).uniform_() < noise_rate
        elif noise_keypoint is not None:
            self.noise = torch.zeros((len(data), self.obs_frames_num, self.keypoints_num))
            self.noise[:, :, noise_keypoint] = True
        elif noise_rate != 'mask':
            raise Exception('''noise_rate must be either a float number or the term 'mask' ''')

        self.gz_o = 0
        self.lz_o = 0
        self.gz_f = 0
        self.lz_f = 0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        seq = self.data[index]
        seq = seq[:, :, dim_used]
        outputs_keys = ['observed_pose']
        if self.use_mask:
            outputs_keys.append('observed_mask')
        if not self.is_testing:
            outputs_keys.append('future_pose')
            if self.use_mask:
                outputs_keys.append('future_mask')

        outputs = dict()
        for key in outputs_keys:
            if key in seq.keys():
                outputs[key] = seq[key]
            else:
                raise Exception('dataset must include ' + key)

        if self.noise is not None:
            outputs['observed_noise'] = self.noise[index]
        else:
            if 'observed_mask' not in seq.keys():
                raise Exception('data-mask is not available. assign a value to noise_rate to get a uniform noise.')
            outputs['observed_noise'] = seq['observed_mask']

        if self.use_quaternion:
            outputs['observed_quaternion_pose'] = torch.tensor(seq['observed_quaternion_pose'])
            outputs['future_quaternion_pose'] = torch.tensor(seq['future_quaternion_pose'])

        if self.is_visualizing:
            if 'video_section' in seq.keys():
                outputs['video_section'] = seq['video_section']
            if 'observed_image_path' in seq.keys():
                outputs['observed_image'] = seq['observed_image_path']
            if 'future_image_path' in seq.keys():
                outputs['future_image'] = seq['future_image_path']
            if 'observed_cam_extrinsic' in seq.keys():
                outputs['observed_cam_ex'] = torch.tensor(seq['observed_cam_extrinsic'])
            if 'future_cam_extrinsic' in seq.keys():
                outputs['future_cam_ex'] = torch.tensor(seq['future_cam_extrinsic'])
            if 'cam_intrinsic' in seq.keys():
                outputs['cam_in'] = torch.tensor(seq['cam_intrinsic'])

        return outputs
