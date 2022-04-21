import json
import logging
import os

import jsonlines
import numpy as np
import torch
from torch.utils.data import Dataset

from path_definition import PREPROCESSED_DATA_DIR

logger = logging.getLogger(__name__)


def find_indices_256(frame_num1, frame_num2, seq_len, input_n=10):
    """
    Adapted from https://github.com/una-dinosauria/human-motion-prediction/blob/master/src/seq2seq_model.py#L478
    which originaly from
    In order to find the same action indices as in SRNN.
    https://github.com/asheshjain399/RNNexp/blob/master/structural_rnn/CRFProblems/H3.6m/processdata.py#L325
    """

    # Used a fixed dummy seed, following
    # https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/forecastTrajectories.py#L29
    SEED = 1234567890
    rng = np.random.RandomState(SEED)

    T1 = frame_num1 - 150
    T2 = frame_num2 - 150  # seq_len
    idxo1 = None
    idxo2 = None
    for _ in np.arange(0, 128):
        idx_ran1 = rng.randint(16, T1)
        idx_ran2 = rng.randint(16, T2)
        idxs1 = np.arange(idx_ran1 + 50 - input_n, idx_ran1 + 50 - input_n + seq_len)
        idxs2 = np.arange(idx_ran2 + 50 - input_n, idx_ran2 + 50 - input_n + seq_len)
        if idxo1 is None:
            idxo1 = idxs1
            idxo2 = idxs2
        else:
            idxo1 = np.vstack((idxo1, idxs1))
            idxo2 = np.vstack((idxo2, idxs2))
    return idxo1, idxo2


class RandomCropDataset(Dataset):
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
                 len_future,
                 is_h36_testing):

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
                        seq_tensor[k[6:]] = v
                    if k in self.extra_keys_to_keep:
                        seq_tensor[k] = v

                assert "pose" in seq_tensor, "model pose format not found in the sequence"
                assert "metric_pose" in seq_tensor, "metric pose format not found in the sequence"

                data.append(seq_tensor)
                len_seq = seq_tensor['pose'].shape[0]
                indexes = indexes + [(len(data) - 1, i)
                                     for i in range(0, len_seq - total_len + 1, seq_rate)]

        if is_h36_testing:
            indexes = []
            for i in range(0, len(data), 2):

                len1 = (data[i]['pose'].shape[0] + frame_rate - 1) // frame_rate
                len2 = (data[i + 1]['pose'].shape[0] + frame_rate - 1) // frame_rate

                idxo1, idxo2 = find_indices_256(len1, len2,
                                                len_observed + len_future, len_observed)
                indexes = indexes + [(i, j * frame_rate) for j in idxo1[:, 0]]
                indexes = indexes + [(i + 1, j * frame_rate) for j in idxo2[:, 0]]

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
            temp_seq = temp_seq[::self.frame_rate]
            outputs["observed_" + k] = temp_seq[:self.len_observed]
            outputs["future_" + k] = temp_seq[self.len_observed:]

        for k in self.extra_keys_to_keep:
            if k in seq:
                outputs[k] = seq[k]

        return outputs
