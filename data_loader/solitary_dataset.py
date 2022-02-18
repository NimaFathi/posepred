import os
import logging

import json
import jsonlines
import torch
from torch.utils.data import Dataset

from path_definition import PREPROCESSED_DATA_DIR

logger = logging.getLogger(__name__)


class SolitaryDataset(Dataset):
    def __init__(self, 
            dataset_path, 
            keypoint_dim, 
            is_testing, 
            use_mask, 
            is_visualizing, 
            use_expmap,
            use_quaternion, 
            normalize,
            metadata_path):

        self.normalize = normalize
        self.use_expmap = use_expmap
        self.use_quaternion = use_quaternion

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
        tensor_keys = [
                'observed_pose', 
                'future_pose', 
                'observed_mask', 
                'future_mask'
                ]
        if self.use_expmap:
            tensor_keys.append('observed_expmap_pose')
            tensor_keys.append('future_expmap_pose')

        if self.use_quaternion:
            tensor_keys.append('observed_quaternion_pose')
            tensor_keys.append('future_quaternion_pose')

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
        self.use_expmap = use_expmap
        self.use_quaternion = use_quaternion

        seq = self.data[0]
        assert 'observed_pose' in seq.keys(), 'dataset must include observed_pose'
        self.keypoints_num = int(seq['observed_pose'].shape[-1] / self.keypoint_dim)
        self.obs_frames_num = seq['observed_pose'].shape[-2]
        if not self.is_testing:
            assert 'future_pose' in seq.keys(), 'dataset must include future_pose'
            self.future_frames_num = seq['future_pose'].shape[-2]

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
        
        if self.use_expmap:
            outputs['observed_expmap_pose'] = \
                    torch.tensor(seq['observed_expmap_pose'])
            outputs['future_expmap_pose'] = \
                    torch.tensor(seq['future_expmap_pose'])

        if self.use_quaternion:
            outputs['observed_quaternion_pose'] = \
                    torch.tensor(seq['observed_quaternion_pose'])
            outputs['future_quaternion_pose'] = \
                    torch.tensor(seq['future_quaternion_pose'])

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
