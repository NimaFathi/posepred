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
                 model_pose_format,
                 metric_pose_format,
                 normalize,
                 metadata_path,
                 seq_rate,
                 frame_rate,
                 len_observed,
                 len_future):

        print("Initialzing Our Dataset:")

        self.normalize = normalize
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

        # TODO
        if not metric_pose_format:
            metric_pose_format = model_pose_format

        indexes = []
        self.extra_keys_to_keep = ['video_section', 'action', 'cam_intrinsic']

        with jsonlines.open(dataset_path) as reader:
            for seq in reader:

                seq_tensor = {}
                for k, v in seq.items():
                    if k == "{}_pose".format(model_pose_format):
                        seq_tensor["pose"] = torch.tensor(v, dtype=torch.float32)
                    if k == "{}_pose".format(metric_pose_format):
                        seq_tensor["metric_pose"] = torch.tensor(v, dtype=torch.float32)
                    if k in 'total_mask':
                        seq_tensor['mask'] = torch.tensor(v, dtype=torch.float32)
                    if k in ['total_image_path', 'total_cam_extrinsic']:
                        seq_tensor[k[6:]] = torch.tensor(v)
                    if k in self.extra_keys_to_keep:
                        seq_tensor[k] = v

                # print(seq.keys())
                assert "pose" in seq_tensor, "model pose format not found in the sequence"
                assert "metric_pose" in seq_tensor, "metric pose format not found in the sequence"

                data.append(seq_tensor)
                len_seq = seq_tensor['pose'].shape[0]
                indexes = indexes + [(len(data) - 1, i)
                                     for i in range(0, len_seq - total_len + 1, seq_rate)]

        self.obs_frames_num = self.len_observed
        self.future_frames_num = self.len_future

        self.keypoints_num = int(data[0]['pose'].shape[-1] // keypoint_dim)

        self.data = data
        self.indexes = indexes
        self.keypoint_dim = keypoint_dim
        self.is_testing = is_testing
        self.use_mask = use_mask
        self.is_visualizing = is_visualizing

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, index):
        data_index, seq_index = self.indexes[index]
        seq = self.data[data_index]
        outputs = {}

        output_keys = ['metric_pose', 'pose']
        if self.use_mask:
            output_keys.append('mask')
        if self.is_visualizing:
            if 'image_path' in seq.keys():
                output_keys.append('image_path')
            if 'cam_extrinsic' in seq.keys():
                output_keys.append('cam_extrinsic')

        for k in output_keys:
            temp_seq = seq[k][seq_index:seq_index + self.total_len]
            s = temp_seq.shape  # T , JD
            temp_seq = temp_seq.view(-1, self.frame_rate, s[1])[:, 0, :]
            outputs["observed_" + k] = temp_seq[:self.len_observed]
            outputs["future_" + k] = temp_seq[self.len_observed:]

        for k in self.extra_keys_to_keep:
            if k in seq:
                outputs[k] = seq[k]

        return outputs
