import logging
import os
import re
from collections import defaultdict

import jsonlines
import numpy as np
import pandas as pd

from path_definition import PREPROCESSED_DATA_DIR
from preprocessor.preprocessor import Processor

logger = logging.getLogger(__name__)


class Preprocessor3DPW(Processor):
    def __init__(self, dataset_path, is_interactive, obs_frame_num, pred_frame_num, skip_frame_num,
                 use_video_once, custom_name):
        super(Preprocessor3DPW, self).__init__(dataset_path, is_interactive, obs_frame_num,
                                               pred_frame_num, skip_frame_num, use_video_once, custom_name)

        self.output_dir = os.path.join(
            PREPROCESSED_DATA_DIR, '3DPW_interactive') if self.is_interactive else os.path.join(
            PREPROCESSED_DATA_DIR, '3DPW'
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

    def normal(self, data_type='train'):
        logger.info('start creating 3DPW normal static data ... ')
        total_frame_num = self.obs_frame_num + self.pred_frame_num

        if self.custom_name:
            output_file_name = f'{data_type}_{self.obs_frame_num}_{self.pred_frame_num}_{self.skip_frame_num}_{self.custom_name}.jsonl'
        else:
            output_file_name = f'{data_type}_{self.obs_frame_num}_{self.pred_frame_num}_{self.skip_frame_num}_3dpw.jsonl'
        assert os.path.exists(os.path.join(
            self.output_dir,
            output_file_name
        )) is False, f"preprocessed file exists at {os.path.join(self.output_dir, output_file_name)}"
        if data_type == 'test':
            print('FALSE')
            org_obs_frame_num = self.obs_frame_num
            self.obs_frame_num = 50
            for entry in os.scandir(self.dataset_path):
                if not entry.name.endswith('.pkl'):
                    continue
                logger.info(f'file name: {entry.name}')
                pickle_obj = pd.read_pickle(entry.path)
                video_name = re.search('(\w+).pkl', entry.name).group(1)
                pose_data = np.array(pickle_obj['jointPositions'])
                data = []
                data_range = [np.arange(i, i + total_frame_num) for i in range(pose_data.shape[1] - total_frame_num + 1)]
                for i, frame_list in enumerate(data_range):
                    video_data = {
                        'obs_pose': defaultdict(list),
                        'future_pose': defaultdict(list),

                    }
                    for frame in frame_list:
                        for p_id in range(pose_data.shape[0]):
                            if frame - min(frame_list) < org_obs_frame_num:
                                if data_type == 'test' and frame - min(frame_list) >= self.obs_frame_num - org_obs_frame_num:
                                    # print(frame, self.obs_frame_num, org_obs_frame_num)
                                    video_data['obs_pose'][p_id].append(pose_data[p_id, frame, :].tolist())
                                else:
                                    video_data['obs_pose'][p_id].append(pose_data[p_id, frame, :].tolist())
                            else:
                                video_data['future_pose'][p_id].append(pose_data[p_id, frame, :].tolist())
                    if len(list(video_data['obs_pose'].values())) > 0:
                        if data_type == 'train':
                            self.update_meta_data(self.meta_data, list(video_data['obs_pose'].values()), 3)
                        if not self.is_interactive:
                            for p_id in range(len(pose_data)):
                                data.append([
                                    '%s-%d' % (video_name, i),
                                    list(video_data['obs_pose'][p_id]), list(video_data['future_pose'][p_id]),
                                ])
                        else:
                            data.append([
                                '%s-%d' % (video_name, i),
                                video_data['obs_pose'].values(), video_data['future_pose'].values(),
                            ])
                with jsonlines.open(os.path.join(self.output_dir, output_file_name), 'a') as writer:
                    for data_row in data:
                        writer.write({
                            'video_section': data_row[0],
                            'observed_pose': data_row[1],
                            'future_pose': data_row[2],
                        })
        else:
            print('TRUEE')
            dataset_paths = ['/work/vita/JTA_dataset/Original_JTA_dataset/annotations/train',
                             '/work/vita/JTA_dataset/Original_JTA_dataset/annotations/val'
                             ]
            for data_path in dataset_paths:
                for entry in os.scandir(data_path):
                    if not entry.name.endswith('.pkl'):
                        continue
                    print('entry')
                    logger.info(f'file name: {entry.name}')
                    pickle_obj = pd.read_pickle(entry.path)
                    video_name = re.search('(\w+).pkl', entry.name).group(1)
                    pose_data = np.array(pickle_obj['jointPositions'])
                    data = []
                    data_range = [np.arange(i, i + total_frame_num) for i in
                                  range(pose_data.shape[1] - total_frame_num + 1)]
                    for i, frame_list in enumerate(data_range):
                        video_data = {
                            'obs_pose': defaultdict(list),
                            'future_pose': defaultdict(list),

                        }
                        for frame in frame_list:
                            for p_id in range(pose_data.shape[0]):
                                if frame - min(frame_list) < self.obs_frame_num:
                                    video_data['obs_pose'][p_id].append(pose_data[p_id, frame, :].tolist())
                                else:
                                    video_data['future_pose'][p_id].append(pose_data[p_id, frame, :].tolist())
                        if len(list(video_data['obs_pose'].values())) > 0:
                            if not self.is_interactive:
                                for p_id in range(len(pose_data)):
                                    data.append([
                                        '%s-%d' % (video_name, i),
                                        list(video_data['obs_pose'][p_id]), list(video_data['future_pose'][p_id]),
                                    ])
                            else:
                                data.append([
                                    '%s-%d' % (video_name, i),
                                    video_data['obs_pose'].values(), video_data['future_pose'].values(),
                                ])
                    with jsonlines.open(os.path.join(self.output_dir, output_file_name), 'a') as writer:
                        for data_row in data:
                            writer.write({
                                'video_section': data_row[0],
                                'observed_pose': data_row[1],
                                'future_pose': data_row[2],
                            })
        # self.save_meta_data(self.meta_data, self.output_dir, True, data_type)
