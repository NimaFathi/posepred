import torch
from torch.utils.data import Dataset
import pandas as pd
from ast import literal_eval

import logging
from logging import config
from path_definition import LOGGER_CONF

config.fileConfig(LOGGER_CONF)
logger = logging.getLogger('consoleLogger')


class NonInteractiveDataset(Dataset):
    def __init__(self, dataset_path, keypoint_dim, is_testing, use_mask, skip_frame, is_visualizing):
        data = pd.read_csv(dataset_path)

        for col in list(data.columns[1:].values):
            try:
                data.loc[:, col] = data.loc[:, col].apply(lambda x: literal_eval(x))
            except Exception:
                msg = "Each row must be convertible to python list"
                logger.exception(msg=msg)
                raise Exception(msg)

        self.data = data.copy().reset_index(drop=True)
        self.is_testing = is_testing
        self.use_mask = use_mask
        self.skip_frame = skip_frame
        self.is_visualizing = is_visualizing

        seq = self.data.iloc[0]  # [1:].apply(lambda x: literal_eval(x))
        self.keypoint_dim = keypoint_dim
        self.keypoints_num = int(len(seq.observed_pose[0]) / self.keypoint_dim)
        self.obs_frames_num = len(seq.observed_pose)
        if not self.is_testing:
            self.future_frames_num = len(seq.future_pose)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        try:
            seq = self.data.iloc[index]  # [1:].apply(lambda x: literal_eval(x))
        except Exception:
            msg = "data must be convertible to valid data-structures"
            logger.exception(msg=msg)
            raise Exception(msg)

        try:
            observed_pose = self.get_tensor(seq, 'observed_pose')
            outputs = {'observed_pose': observed_pose}
        except:
            logger.warning('faulty row skipped.')
            return self.__getitem__((index + 1) % self.__len__())

        if self.use_mask:
            observed_mask = self.get_tensor(seq, 'observed_mask')
            outputs['observed_mask'] = observed_mask

        if not self.is_testing:
            future_pose = self.get_tensor(seq, 'future_pose')
            outputs['future_pose'] = future_pose

            if self.use_mask:
                future_mask = self.get_tensor(seq, 'future_mask')
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

    def get_tensor(self, seq, segment):
        assert segment in seq, 'No segment named: ' + segment
        frames_num = len(seq[segment])
        return torch.tensor([seq[segment][frame_idx] for frame_idx in range(0, frames_num, self.skip_frame + 1)],
                            dtype=torch.float32)
