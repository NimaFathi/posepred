import logging
import os
from glob import glob

import cdflib
import jsonlines
import numpy as np

from path_definition import PREPROCESSED_DATA_DIR
from preprocessor.preprocessor import Processor

logger = logging.getLogger(__name__)

SPLIT = {
    'train': ['S1', 'S6', 'S7', 'S8', 'S9', 'S11'],
    'validation': ['S1', 'S6', 'S7', 'S8', 'S9', 'S11'],
    'test': ['S5']
}


class PreprocessorHuman36mCategorical(Processor):
    def __init__(self, dataset_path, is_interactive, obs_frame_num, pred_frame_num, skip_frame_num,
                 use_video_once, custom_name):
        super(PreprocessorHuman36mCategorical, self).__init__(
            dataset_path, is_interactive, obs_frame_num,
            pred_frame_num, skip_frame_num, use_video_once, custom_name
        )
        assert self.is_interactive is False, 'human3.6m is not interactive'
        self.output_dir = os.path.join(
            PREPROCESSED_DATA_DIR, 'human36m_categorical_interactive') if self.is_interactive else os.path.join(
            PREPROCESSED_DATA_DIR, 'human36m_categorical'
        )
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.meta_data = {
            'avg_person': [],
            'max_pose': np.zeros(3),
            'min_pose': np.array([1000.0, 1000.0, 1000.0]),
            'count': 0,
            'sum2_pose': np.zeros(3),
            'sum_pose': np.zeros(3)
        }

    def create_test(self, num_samples, category):
        samples = []
        subject_pose_path = os.path.join(self.dataset_path, self.subjects[0], 'MyPoseFeatures/D3_Positions/*.cdf')
        file_list_pose = glob(subject_pose_path)
        for f in file_list_pose:
            num_create_samples = 0
            action = os.path.splitext(os.path.basename(f))[0]
            if not action.lower().split(" ")[0] == category.lower():
                continue
            hf = cdflib.CDF(f)
            positions = hf['Pose'].reshape(-1, 96)
            positions /= 1000
            for i in range(0,
                           positions.shape[0] - (self.skip_frame_num + 1) * (self.pred_frame_num + self.obs_frame_num),
                           self.obs_frame_num):
                sample = positions[i: i + (self.skip_frame_num + 1) * (self.obs_frame_num + self.pred_frame_num):(
                        self.skip_frame_num + 1)]
                num_create_samples += 1
                if num_create_samples > num_samples / 2:
                    break
                samples.append(sample)
        return samples

    def sample(self, category):
        subject = np.random.choice(self.subjects)
        subject_pose_paths = []
        categories = [category]
        if category == 'Photo':
            categories.append('TakingPhoto')
        elif category == 'WalkDog':
            categories.append('WalkingDog')
        for cat in categories:
            for file_no in range(0, 5):
                possible_path = os.path.join(self.dataset_path, subject,
                                             f'MyPoseFeatures/D3_Positions/{cat} {file_no}.cdf')
                if os.path.exists(possible_path):
                    subject_pose_paths.append(possible_path)
            if os.path.exists(
                    os.path.join(self.dataset_path, subject, f'MyPoseFeatures/D3_Positions/{cat}.cdf')):
                subject_pose_paths.append(
                    os.path.join(self.dataset_path, subject, f'MyPoseFeatures/D3_Positions/{cat}.cdf'))
        file = np.random.choice(subject_pose_paths)
        hf = cdflib.CDF(file)
        positions = hf['Pose'].reshape(-1, 96)
        positions /= 1000
        fr_start = np.random.randint(
            positions.shape[0] - (self.skip_frame_num + 1) * (self.pred_frame_num + self.obs_frame_num))
        fr_end = fr_start + (self.skip_frame_num + 1) * (self.pred_frame_num + self.obs_frame_num)
        traj = positions[fr_start: fr_end: (self.skip_frame_num + 1)]
        return traj

    def normal(self, data_type='train'):
        category = 'WalkTogether'
        self.subjects = SPLIT[data_type]
        logger.info('start creating Human3.6m normal static data from original Human3.6m dataset (CDF files) ... ')
        if self.custom_name:
            output_file_name = f'{data_type}_{self.obs_frame_num}_{self.pred_frame_num}_{self.skip_frame_num}_{self.custom_name}_{category}.jsonl'
        else:
            output_file_name = f'{data_type}_{self.obs_frame_num}_{self.pred_frame_num}_{self.skip_frame_num}_human3.6m_categorical_{category}.jsonl'
        assert os.path.exists(os.path.join(
            self.output_dir,
            output_file_name
        )) is False, f"preprocessed file exists at {os.path.join(self.output_dir, output_file_name)}"
        if data_type == 'train' or data_type == 'validation':
            if data_type == 'train':
                samples_num = 2000
            else:
                samples_num = 256
            for i in range(samples_num):
                pose = self.sample(category)
                with jsonlines.open(os.path.join(self.output_dir, output_file_name), mode='a') as writer:
                    writer.write({
                        'observed_pose': pose[:self.obs_frame_num, :].tolist(),
                        'future_pose': pose[self.obs_frame_num:, :].tolist(),
                    })
        else:
            samples = self.create_test(256, category)
            with jsonlines.open(os.path.join(self.output_dir, output_file_name), mode='a') as writer:
                for sample in samples:
                    writer.write({
                        'observed_pose': sample[:self.obs_frame_num, :].tolist(),
                        'future_pose': sample[self.obs_frame_num:, :].tolist(),
                    })
