import os
import logging

import json
import jsonlines
import torch
from torch.utils.data import Dataset

from path_definition import PREPROCESSED_DATA_DIR

logger = logging.getLogger(__name__)


class OurDataset(Dataset):
    def __init__(self,
                 dataset_path,
                 keypoint_dim,
                 is_testing,
                 use_mask,
                 is_visualizing,
                 use_expmap,
                 use_rotmat,
                 use_euler,
                 use_quaternion,
                 use_xyz,
                 normalize,
                 metadata_path,
                 seq_rate,
                 frame_rate,
                 len_observed,
                 len_future):

        self.normalize = normalize
        self.use_expmap = use_expmap
        self.use_rotmat = use_rotmat
        self.use_euler = use_euler
        self.use_quaternion = use_quaternion
        self.use_xyz = use_xyz
        total_len = (len_observed+len_future)*frame_rate
        self.frame_rate = frame_rate
        self.total_len = total_len
        self.len_observed = len_observed
        self.len_future = len_future

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
        self.tensor_keys = [
        ]

        if self.use_expmap:
            self.tensor_keys.append('expmap_pose')

        if self.use_rotmat:
            self.tensor_keys.append('rotmat_pose')

        if self.use_euler:
            self.tensor_keys.append('euler_pose')

        if self.use_quaternion:
            self.tensor_keys.append('quaternion_pose')

        if self.use_xyz:
            self.tensor_keys.append('xyz_pose')

        assert len(
            self.tensor_keys) > 0, "please determine the kind(s) of pose(es) you want to use in config file"
        if 'xyz_pose' in self.tensor_keys:
            assert len(self.tensor_keys)==1, "you can't use xyz with other poses"

        indexes = []
        with jsonlines.open(dataset_path) as reader:
            for seq in reader:
                seq_tensor = {}
                for k, v in seq.items():
                    if k in self.tensor_keys:
                        seq_tensor[k] = torch.tensor(v, dtype=torch.float32)

                    else:
                        seq_tensor[k] = v
                data.append(seq_tensor)
                len_seq = seq_tensor[self.tensor_keys[0]].shape[0]
                print("the sequence length is: ", len_seq)
                indexes = indexes + [(len(data)-1, i)
                                     for i in range(0, len_seq-total_len+1, seq_rate)]

        self.data = data
        self.indexes = indexes
        self.keypoint_dim = keypoint_dim
        self.is_testing = is_testing
        self.use_mask = use_mask
        self.is_visualizing = is_visualizing
        self.use_expmap = use_expmap
        self.use_quaternion = use_quaternion

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, index):
        data_index, seq_index = self.data[index]
        seq = self.data[data_index][seq_index:seq_index+self.total_len]
        outputs = {"observed_pose": {}, "future_pose": {}}
        
        for k in self.tensor_keys:
            temp_seq = seq[k]
            s = temp_seq.shape
            temp_seq = temp_seq.view(-1,self.frame_rate, s[2], s[3])[:, 0, :, :]
            outputs['observed_pose'][k] = temp_seq[:self.len_observed]
            outputs['future_pose'][k] = temp_seq[self.len_observed:]

        return outputs
