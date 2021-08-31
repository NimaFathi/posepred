import csv
import os
import re
from collections import defaultdict

import numpy as np
import pandas as pd

from path_definition import PREPROCESSED_DATA_DIR
from preprocessor.preprocessor import Processor


class Preprocessor3DPW(Processor):
    def __init__(self, dataset_path, obs_frame_num, pred_frame_num, skip_frame_num,
                 use_video_once, custom_name, is_interactive):
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
            'count': 0,
            'sum2_pose': np.zeros(3),
            'sum_pose': np.zeros(3)
        }

    def normal(self, data_type='train'):
        print('start creating 3DPW normal static data ... ')
        header = [
            'video_section', 'observed_pose', 'future_pose',
            'observed_frames_related_path', 'future_frames_related_path',
            'obs_cam_extrinsic', 'future_camera_extrinsic'
        ]
        total_frame_num = self.obs_frame_num + self.pred_frame_num

        if self.custom_name:
            output_file_name = f'{data_type}_{self.obs_frame_num}_{self.pred_frame_num}_{self.skip_frame_num}_{self.custom_name}.csv'
        else:
            output_file_name = f'{data_type}_{self.obs_frame_num}_{self.pred_frame_num}_{self.skip_frame_num}_3dpw.csv'

        with open(os.path.join(self.output_dir, output_file_name), 'w') as f_object:
            writer = csv.writer(f_object)
            writer.writerow(header)
        for entry in os.scandir(self.dataset_path):
            if not entry.name.endswith('.pkl'):
                continue
            print(f'file name: {entry.name}')
            pickle_obj = pd.read_pickle(entry.path)
            video_name = re.search('(\w+).pkl', entry.name).group(1)
            pose_data = np.array(pickle_obj['jointPositions'])
            frame_ids_data = pickle_obj['img_frame_ids']
            cam_extrinsic = pickle_obj['cam_poses']
            cam_intrinsic = pickle_obj['cam_intrinsics'].tolist()
            section_range = pose_data.shape[1] // (total_frame_num * 2) if self.use_video_once is False else 1
            data = []
            for i in range(section_range):
                video_data = {
                    'obs_pose': defaultdict(list),
                    'future_pose': defaultdict(list),
                    'obs_frames': defaultdict(list),
                    'future_frames': defaultdict(list),
                    'obs_cam_ext': list(),
                    'future_cam_ext': list()
                }
                for j in range(1, total_frame_num * self.skip_frame_num + 1, self.skip_frame_num):
                    for p_id in range(pose_data.shape[0]):
                        if j <= self.skip_frame_num * self.obs_frame_num:
                            video_data['obs_pose'][p_id].append(
                                pose_data[p_id, i * total_frame_num * self.skip_frame_num + j - 1, :].tolist()
                            )
                            video_data['obs_frames'][p_id].append(
                                f'{video_name}/image_{frame_ids_data[i * total_frame_num * self.skip_frame_num + j - 1]:05}.jpg'
                            )
                            if p_id == 0:
                                video_data['obs_cam_ext'].append(
                                    cam_extrinsic[i * total_frame_num * self.skip_frame_num + j - 1].tolist()
                                )
                        else:
                            video_data['future_pose'][p_id].append(
                                pose_data[p_id, i * total_frame_num * 2 + j - 1, :].tolist()
                            )
                            video_data['future_frames'][p_id].append(
                                f'{video_name}/image_{frame_ids_data[i * total_frame_num * self.skip_frame_num + j - 1]:05}.jpg'
                            )
                            if p_id == 0:
                                video_data['future_cam_ext'].append(
                                    cam_extrinsic[i * total_frame_num * self.skip_frame_num + j - 1].tolist()
                                )

                if len(list(video_data['obs_pose'].values())) > 0:
                    if data_type == 'train':
                        self.update_meta_data(self.meta_data, list(video_data['obs_pose'].values()), 3)
                    if not self.is_interactive:
                        for p_id in range(len(pose_data)):
                            data.append([
                                '%s-%d' % (video_name, i),
                                video_data['obs_pose'][p_id], video_data['future_pose'][p_id],
                                video_data['obs_frames'][p_id], video_data['future_frames'][p_id],
                                video_data['obs_cam_ext'], video_data['future_cam_ext'], cam_intrinsic
                            ])
                    else:
                        data.append([
                            '%s-%d' % (video_name, i),
                            list(video_data['obs_pose'].values()), list(video_data['future_pose'].values()),
                            video_data['obs_frames'][0], video_data['future_frames'][0],
                            video_data['obs_cam_ext'], video_data['future_cam_ext'], cam_intrinsic
                        ])
            with open(os.path.join(self.output_dir, output_file_name), 'a') as f_object:
                writer = csv.writer(f_object)
                writer.writerows(data)
        self.save_meta_data(self.meta_data, self.output_dir, True, data_type)
