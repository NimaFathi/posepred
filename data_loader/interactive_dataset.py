import torch
from torch.utils.data import Dataset
from numpy import random
import jsonlines
import logging

logger = logging.getLogger(__name__)


class InteractiveDataset(Dataset):
    def __init__(self, dataset_path, keypoint_dim, persons_num, is_testing, use_mask, skip_frame, is_visualizing):

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
        self.persons_num = persons_num
        self.is_testing = is_testing
        self.use_mask = use_mask
        self.skip_frame = skip_frame
        self.is_visualizing = is_visualizing

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
        persons_in_seq = self.select_persons(seq)

        observed_pose = self.fix_persons(seq, 'observed_pose', persons_in_seq)
        outputs = {'observed_pose': observed_pose}

        if self.use_mask:
            observed_mask = self.fix_persons(seq, 'observed_mask', persons_in_seq)
            outputs['observed_mask'] = observed_mask

        if not self.is_testing:
            assert seq['observed_pose'].shape[0] == seq['future_pose'].shape[
                0], "unequal number of persons in observed and future frames."
            future_pose = self.fix_persons(seq, 'future_pose', persons_in_seq)
            outputs['future_pose'] = future_pose

            if self.use_mask:
                future_mask = self.fix_persons(seq, 'future_mask', persons_in_seq)
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
        persons_in_seq = list(range(seq['observed_pose'].shape[0]))
        if self.persons_num < len(persons_in_seq):
            return random.choice(persons_in_seq, self.persons_num, replace=False)
        else:
            return persons_in_seq

    def fix_persons(self, seq, segment, persons_in_seq):
        assert segment in seq, 'No segment named: ' + segment
        result = seq[segment][persons_in_seq]
        if len(persons_in_seq) < self.persons_num:
            padding = torch.zeros(self.persons_num - len(persons_in_seq), result.shape[1], result.shape[2])
            result = torch.cat((result, padding), dim=0)

        return result
