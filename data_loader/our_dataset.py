import json
import logging
import os

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

        print("Initialzing Dataset:")

        self.normalize = normalize
        self.use_expmap = use_expmap
        self.use_rotmat = use_rotmat
        self.use_euler = use_euler
        self.use_quaternion = use_quaternion
        self.use_xyz = use_xyz
        total_len = (len_observed + len_future) * frame_rate
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
        self.tensor_keys_to_keep = []
        self.tensor_keys_to_ignore = []

        for key, value in zip(
                ['expmap_pose', 'rotmat_pose', 'euler_pose', 'quaternion_pose', 'xyz_pose'],
                [self.use_expmap, self.use_rotmat, self.use_euler, self.use_quaternion, self.use_xyz]
        ):
            if value:
                self.tensor_keys_to_keep.append(key)
            else:
                self.tensor_keys_to_ignore.append(key)

        assert len(
            self.tensor_keys_to_keep) > 0, "please determine the kind(s) of pose(es) you want to use in config file"

        indexes = []
        with jsonlines.open(dataset_path) as reader:
            for seq in reader:

                seq_tensor = {}
                for k, v in seq.items():
                    if k in self.tensor_keys_to_keep:
                        seq_tensor[k] = torch.tensor(v, dtype=torch.float32)
                    elif k not in self.tensor_keys_to_ignore:
                        seq_tensor[k] = v

                data.append(seq_tensor)
                len_seq = seq_tensor[self.tensor_keys_to_keep[0]].shape[0]
                indexes = indexes + [(len(data) - 1, i)
                                     for i in range(0, len_seq - total_len + 1, seq_rate)]

        self.keypoints_num = 3
        self.obs_frames_num = self.len_observed
        self.future_frames_num = self.len_future

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
        data_index, seq_index = self.indexes[index]
        seq = self.data[data_index]
        outputs = {}

        for k in self.tensor_keys_to_keep:
            temp_seq = seq[k][seq_index:seq_index + self.total_len]
            s = temp_seq.shape
            temp_seq = temp_seq.view(-1, self.frame_rate, s[1], s[2])[:, 0, :, :]
            outputs["observed_" + k] = temp_seq[:self.len_observed]
            outputs["future_" + k] = temp_seq[self.len_observed:]

        outputs["action"] = seq["action"]
        # todo: save other keys as well, not just the actions

        return outputs
