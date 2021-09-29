import pandas as pd
import torch
from numpy import random
from torch.utils.data import Dataset
from ast import literal_eval

import logging
from logging import config
from path_definition import LOGGER_CONF

config.fileConfig(LOGGER_CONF)
logger = logging.getLogger('consoleLogger')


class InteractiveDataset(Dataset):
    def __init__(self, dataset_path, keypoint_dim, persons_num, is_testing, use_mask, skip_frame, is_visualizing):
        data = pd.read_csv(dataset_path)

        for col in list(data.columns[1:].values):
            try:
                data.loc[:, col] = data.loc[:, col].apply(lambda x: literal_eval(x))
            except Exception:
                msg = "Each row must be convertible to python list"
                logger.exception(msg=msg)
                raise Exception(msg)

        self.data = data.copy().reset_index(drop=True)
        self.persons_num = persons_num
        self.is_testing = is_testing
        self.use_mask = use_mask
        self.skip_frame = skip_frame
        self.is_visualizing = is_visualizing

        seq = self.data.iloc[0]  # [1:].apply(lambda x: literal_eval(x))
        self.keypoint_dim = keypoint_dim
        self.keypoints_num = len(seq.observed_pose[0][0]) / self.keypoint_dim
        self.obs_frames_num = len(seq.observed_pose[0])
        if not self.is_testing:
            self.future_frames_num = len(seq.future_pose[0])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        seq = self.data.iloc[index]  # [1:].apply(lambda x: literal_eval(x))
        persons_in_seq = self.select_persons(seq)

        try:
            observed_pose = self.get_tensor(seq, 'observed_pose', persons_in_seq, self.obs_frames_num)
            outputs = {'observed_pose': observed_pose}
        except:
            logger.warning('faulty row skipped.')
            return self.__getitem__((index + 1) % self.__len__())

        if self.use_mask:
            observed_mask = self.get_tensor(seq, 'observed_mask', persons_in_seq, self.obs_frames_num)
            outputs['observed_mask'] = observed_mask

        if not self.is_testing:
            assert len(seq.observed_pose) == len(seq.future_pose), "unequal persons in observed and future frames."
            future_pose = self.get_tensor(seq, 'future_pose', persons_in_seq, self.future_frames_num)
            outputs['future_pose'] = future_pose

            if self.use_mask:
                future_mask = self.get_tensor(seq, 'future_mask', persons_in_seq, self.future_frames_num)
                outputs['future_mask'] = future_mask

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

    def select_persons(self, seq):
        persons_in_seq = list(range(len(seq['observed_pose'])))
        if self.persons_num < len(persons_in_seq):
            return random.choice(persons_in_seq, self.persons_num, replace=False)
        else:
            return persons_in_seq

    def get_tensor(self, seq, segment, persons_in_seq, frames_num):
        assert segment in seq, 'No segment named: ' + segment
        result = torch.tensor(
            [[seq[segment][person_idx][frame_idx]
              for frame_idx in range(0, frames_num, self.skip_frame + 1)]
             for person_idx in persons_in_seq], dtype=torch.float32)

        if len(persons_in_seq) < self.persons_num:
            padding = torch.zeros(self.persons_num - len(persons_in_seq), result.shape[1], result.shape[2])
            result = torch.cat((result, padding), dim=0)

        return result
