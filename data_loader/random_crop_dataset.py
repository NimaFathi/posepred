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
        self.is_h36_testing = is_h36_testing
        print(dataset_path, is_testing, is_h36_testing)

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, index):

        return self.get_reconstruction_item(index)

        random_reverse = np.random.choice([False, True])
        if self.is_testing or self.is_h36_testing:
            random_reverse = False

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
            if random_reverse:
                temp_seq = torch.flip(temp_seq, [0])
            temp_seq = temp_seq[::self.frame_rate]
            outputs["observed_" + k] = temp_seq[:self.len_observed]
            outputs["future_" + k] = temp_seq[self.len_observed:]

        for k in self.extra_keys_to_keep:
            if k in seq:
                outputs[k] = seq[k]

        return outputs

    def get_reconstruction_item(self, index):
        random_reverse = np.random.choice([False, True])
        mask = self.get_mask()

        if self.is_testing or self.is_h36_testing:
            random_reverse = False
            mask[:, :] = 0
            mask[:self.len_observed, :] = 1

        data_index, seq_index = self.indexes[index]
        seq = self.data[data_index]
        outputs = {'reconstruction_mask': mask.float()}

        output_keys = ['metric_pose', 'pose']

        for k in output_keys:
            temp_seq = seq[k][seq_index:seq_index + self.total_len]
            if random_reverse:
                temp_seq = torch.flip(temp_seq, [0])
            temp_seq = temp_seq[::self.frame_rate]
            outputs["future_" + k] = temp_seq
            outputs["observed_" + k] = temp_seq

        for k in self.extra_keys_to_keep:
            if k in seq:
                outputs[k] = seq[k]

        return outputs

    def get_mask(self):

        ratio = 100 // 40
        T = self.total_len // self.frame_rate
        J = self.keypoints_num
        C = self.keypoint_dim
        JC = J * C

        mask = torch.zeros(T, JC)

        mode = np.random.choice(range(6))

        if mode == 0:  # random structural : leg, hand, ...
            part = []
            part.append(np.array([2, 3, 4, 5]))
            part.append(np.array([7, 8, 9, 10]))
            part.append(np.array([12, 13, 14, 15]))
            part.append(np.array([17, 18, 19, 21, 22]))
            part.append(np.array([25, 26, 27, 29, 30]))
            part = part[np.random.choice(range(5))]
            part = np.concatenate((part * 3, part * 3 + 1, part * 3 + 2))
            mask[:, part] = 1
        if mode == 1:  # random structural : joints
            part = np.arange(J)
            np.random.shuffle(part)
            part = part[:part.shape[0] // ratio]
            part = np.concatenate((part * 3, part * 3 + 1, part * 3 + 2))
            mask[:, part] = 1
        if mode == 2:  # random frames
            part = np.arange(T)
            np.random.shuffle(part)
            part = part[:part.shape[0] // ratio]
            mask[part, :] = 1
        if mode == 3:  # random segment
            max_len = 3 * T // 4
            l = np.random.choice(range(1, max_len + 1))
            start = np.random.choice(range(T - max_len + 1))
            mask[start:start + l, :] = 1
        if mode == 4:  # random numbers
            mask = torch.tensor(np.random.choice([0.0, 1.0], p=((ratio - 1) / ratio, 1 / ratio), size=mask.shape))
        if mode == 5:  # random X, Y, Z
            part = np.random.choice(range(3))
            mask[:, 3 * np.arange(J) + part] = 1

        return 1 - mask
