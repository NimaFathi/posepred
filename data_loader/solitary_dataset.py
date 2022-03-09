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
            dataset_pathes, 
            keypoint_dim, 
            is_testing, 
            use_mask, 
            is_visualizing, 
            use_action,
            normalize,
            metadata_path):

        self.pose_formats = list(dataset_pathes.keys())

        self.normalize = normalize
        self.use_action = use_action

        self.actions_dict = {
            	"walking": 0,
                "eating": 1,
                "smoking": 2,
                "discussion": 3,
                "directions": 4,
                "greeting": 5,
                "phoning": 6,
                "posing": 7,
                "purchases": 8,
                "sitting": 9,
                "sittingdown": 10,
                "takingphoto": 11,
                "photo": 11,
                "takephoto": 11,
                "waiting": 12,
                "walkingdog": 13,
                "walkdog": 13,
                "walkingtogether": 14,
                "walktogether": 14
        }

        if normalize:
            assert metadata_path, "Specify metadata_path when normalize is true."
            with open(os.path.join(PREPROCESSED_DATA_DIR, metadata_path)) as meta_file:
                meta_data = json.load(meta_file)
                self.mean_pose = {pose_format: list(meta_data[pose_format]['avg_pose']) for pose_format in self.pose_formats}
                self.std_pose = {pose_format: list(meta_data[pose_format]['std_pose']) for pose_format in self.pose_formats}
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
        
        if self.use_action:
            tensor_keys.append('use_action')

        data = {pose_format: [] for pose_format in self.pose_formats}
        for pose_format, dataset_path in dataset_pathes.items():
            with jsonlines.open(dataset_path) as reader:
                for seq in reader:
                    seq_tensor = {}
                    for k, v in seq.items():
                        if k in tensor_keys:
                            seq_tensor[k] = torch.tensor(v, dtype=torch.float32)
                        else:
                            seq_tensor[k] = v
                    data[pose_format].append(seq_tensor)

        self.data = data
        self.keypoint_dim = keypoint_dim
        self.is_testing = is_testing
        self.use_mask = use_mask
        self.is_visualizing = is_visualizing
        
        for pose_format in self.pose_formats:
            seq = self.data[pose_format][0]
            assert 'observed_pose' in seq.keys(), 'dataset must include observed_pose'
            self.keypoints_num = int(seq['observed_pose'].shape[-1] / self.keypoint_dim)
            self.obs_frames_num = seq['observed_pose'].shape[-2]
            if not self.is_testing:
                assert 'future_pose' in seq.keys(), 'dataset must include future_pose'
                self.future_frames_num = seq['future_pose'].shape[-2]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sequences = {pose_format: self.data[pose_format][index] for pose_format in self.pose_formats}

        outputs_keys = ['observed_pose']
        if self.use_mask:
            outputs_keys.append('observed_mask')
        if not self.is_testing:
            outputs_keys.append('future_pose')
            if self.use_mask:
                outputs_keys.append('future_mask')

        outputs = dict()
        for pose_format in self.pose_formats:
            seq = sequences[pose_format]
            for key in outputs_keys:
                if key in seq.keys():
                    outputs[pose_format + '_' + key] = seq[key]
                else:
                    raise Exception('dataset must include ' + key)
        
        if self.use_action:
            outputs['action_ids'] = torch.tensor(self.actions_dict[seq['action'].lower()])

        #print(outputs['action_ids'])
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
