import torch
import json
import pathlib
import os
from torch.utils.data import Dataset
import jsonlines
import logging

logger = logging.getLogger(__name__)


class SolitaryDataset(Dataset):
    def __init__(self, dataset_path, keypoint_dim, is_testing, use_mask, is_visualizing, use_quaternion, normalize):
        self.normalize = normalize
        if self.normalize:
            with open(os.path.join(pathlib.Path(dataset_path).parent.resolve(),
                                   f'{keypoint_dim}D_meta.json')) as meta_file:
                self.meta_data = json.load(meta_file)

        tensor_keys = ['observed_pose', 'future_pose', 'observed_mask', 'future_mask']
        data = list()
        with jsonlines.open(dataset_path) as reader:
            for seq in reader:
                seq_tensor = {}
                for k, v in seq.items():
                    if k in tensor_keys:
                        seq_tensor[k] = torch.tensor(v, dtype=torch.float32)
                    else:
                        seq_tensor[k] = v
                data.append(seq_tensor)

        self.data = data
        self.keypoint_dim = keypoint_dim
        self.is_testing = is_testing
        self.use_mask = use_mask
        self.is_visualizing = is_visualizing
        self.use_quaternion = use_quaternion
        self.normalized_indices = []

        seq = self.data[0]
        assert 'observed_pose' in seq.keys(), 'dataset must include observed_pose'
        self.keypoints_num = int(seq['observed_pose'].shape[-1] / self.keypoint_dim)
        self.obs_frames_num = seq['observed_pose'].shape[-2]
        if not self.is_testing:
            assert 'future_pose' in seq.keys(), 'dataset must include future_pose'
            self.future_frames_num = seq['future_pose'].shape[-2]

    def normalize_data(self, output, index):
        if index in self.normalized_indices:
            return output
        obs = output['observed_pose'].view(*output['observed_pose'].shape[:-1], -1, self.keypoint_dim)
        future = output['future_pose'].view(*output['future_pose'].shape[:-1], -1, self.keypoint_dim)
        for i in range(self.keypoint_dim):
            obs[:, :, i] = (obs[:, :, i] - self.meta_data['avg_pose'][i]) / self.meta_data['std_pose'][i]
            future[:, :, i] = (future[:, :, i] - self.meta_data['avg_pose'][i]) / self.meta_data['std_pose'][i]
        self.normalized_indices.append(index)
        return output

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        seq = self.data[index]

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

        if self.use_quaternion:
            outputs['observed_quaternion_pose'] = torch.tensor(seq['observed_quaternion_pose'])
            outputs['future_quaternion_pose'] = torch.tensor(seq['future_quaternion_pose'])

        if self.normalize:
            outputs = self.normalize_data(outputs, index)

        if self.is_visualizing:
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
