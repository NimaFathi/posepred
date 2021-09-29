import os
from glob import glob

import cdflib
import jsonlines
import numpy as np

from path_definition import PREPROCESSED_DATA_DIR
from preprocessor.preprocessor import Processor


class PreprocessorHuman36m(Processor):
    def __init__(self, dataset_path, is_interactive, obs_frame_num, pred_frame_num, skip_frame_num,
                 use_video_once, custom_name):
        super(PreprocessorHuman36m, self).__init__(dataset_path, is_interactive, obs_frame_num,
                                                   pred_frame_num, skip_frame_num, use_video_once, custom_name)

        self.output_dir = os.path.join(
            PREPROCESSED_DATA_DIR, 'human36m_interactive') if self.is_interactive else os.path.join(
            PREPROCESSED_DATA_DIR, 'human36m'
        )
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.meta_data = {
            'avg_person': [],
            'count': 0,
            'sum2_pose': np.zeros(3),
            'sum_pose': np.zeros(3)
        }
        self.subjects = ['S1', 'S5', 'S6', 'S7', 'S8', 'S9', 'S11']

    def normal(self, data_type='train'):
        print('start creating Human3.6m normal static data from origian Human3.6m dataset (CDF files) ... ')
        if self.custom_name:
            output_file_name = f'{data_type}_{self.obs_frame_num}_{self.pred_frame_num}_{self.skip_frame_num}_{self.custom_name}.jsonl'
        else:
            output_file_name = f'{data_type}_{self.obs_frame_num}_{self.pred_frame_num}_{self.skip_frame_num}_human3.6m.jsonl'
        assert os.path.exists(os.path.join(self.output_dir, output_file_name)) is False, "preprocessed file exists"
        for subject in self.subjects:
            file_list = glob(self.dataset_path + '/' + subject + '/MyPoseFeatures/D3_Positions/*.cdf')
            assert len(file_list) == 30, "Expected 30 files for subject " + subject + ", got " + str(len(file_list))
            for f in file_list:
                action = os.path.splitext(os.path.basename(f))[0]

                if subject == 'S11' and action == 'Directions':
                    continue  # Discard corrupted video

                canonical_name = action.replace('TakingPhoto', 'Photo') \
                    .replace('WalkingDog', 'WalkDog')
                hf = cdflib.CDF(f)
                positions = hf['Pose'].reshape(-1, 96)
                positions /= 1000
                total_frame_num = self.obs_frame_num + self.pred_frame_num
                section_range = positions.shape[0] // (
                            total_frame_num * (self.skip_frame_num + 1)) if self.use_video_once is False else 1
                for i in range(section_range):
                    video_data = {
                        'obs_pose': list(),
                        'future_pose': list(),
                    }
                    for j in range(1, total_frame_num * (self.skip_frame_num + 1) + 1, self.skip_frame_num + 1):
                        if j <= (self.skip_frame_num + 1) * self.obs_frame_num:
                            video_data['obs_pose'].append(
                                positions[i * total_frame_num * (self.skip_frame_num + 1) + j].tolist())
                        else:
                            video_data['future_pose'].append(
                                positions[i * total_frame_num * (self.skip_frame_num + 1) + j].tolist())
                    self.update_meta_data(self.meta_data, video_data['obs_pose'], 3)
                    with jsonlines.open(os.path.join(self.output_dir, output_file_name), mode='a') as writer:
                        writer.write({
                            'video_section': f'{subject}-{canonical_name}-{i}',
                            'observed_pose': video_data['obs_pose'],
                            'future_pose': video_data['future_pose']
                        })
        self.save_meta_data(self.meta_data, self.output_dir, True, data_type)
