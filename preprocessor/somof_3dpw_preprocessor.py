import csv
import json
import os
from collections import defaultdict

import numpy as np

from path_definition import PREPROCESSED_DATA_DIR
from preprocessor.preprocessor import Processor


class SoMoF3DPWPreprocessor(Processor):
    def __init__(self, dataset_path, is_interactive, obs_frame_num, pred_frame_num,
                 skip_frame_num, use_video_once, custom_name):
        super(SoMoF3DPWPreprocessor, self).__init__(dataset_path, is_interactive, obs_frame_num,
                                                    pred_frame_num, skip_frame_num, use_video_once, custom_name)
        self.output_dir = os.path.join(
            PREPROCESSED_DATA_DIR, 'SoMoF_3DPW_interactive') if self.is_interactive else os.path.join(
            PREPROCESSED_DATA_DIR, 'SoMoF_3DPW'
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
        print('start creating SoMoF-3DPW normal static data ... ')
        preprocessed_data = self.__clean_data(data_type)
        self.__save_csv(data_type, preprocessed_data)
        self.save_meta_data(self.meta_data, self.output_dir, True, data_type)

    def __save_csv(self, data_type, processed_data, file_type=None):
        if self.custom_name:
            if file_type is None:
                output_name = f'{data_type}_16_14_1_{self.custom_name}.csv'
            else:
                output_name = f'{file_type}_{data_type}_16_14_1_{self.custom_name}.csv'
        else:
            if file_type is None:
                output_name = f'{data_type}_16_14_1_SoMoF_3dpw.csv'
            else:
                output_name = f'{file_type}_{data_type}_16_14_1_SoMoF_3dpw.csv'
        data = []
        if data_type == 'test':
            header = ['video_section', 'observed_pose', 'observed_image_path']
        else:
            header = ['video_section', 'observed_pose', 'future_pose', 'observed_image_path']
        with open(os.path.join(self.output_dir, output_name), 'w') as f_object:
            writer = csv.writer(f_object)
            writer.writerow(header)
        if data_type == 'test':
            if self.is_interactive:
                for vid_id in range(len(processed_data['obs_pose'])):
                    data.append(['%d-%d' % (vid_id, 0),
                                 processed_data['obs_pose'][vid_id].tolist(),
                                 processed_data['obs_frames_path'][vid_id].tolist()
                                 ])
            else:
                for vid_id in range(len(processed_data['obs_pose'])):
                    for p_id in range(len(processed_data['future_pose'][vid_id])):
                        data.append([
                            '%d-%d' % (vid_id, 0),
                            processed_data['obs_pose'][vid_id][p_id].tolist(),
                            processed_data['obs_frames_path'][vid_id].tolist()
                        ])
        else:
            if self.is_interactive:
                for vid_id in range(processed_data['obs_pose'].__len__()):
                    self.update_meta_data(self.meta_data, processed_data['obs_pose'][vid_id], 3)
                    data.append([
                        '%d-%d' % (vid_id, 0), processed_data['obs_pose'][vid_id].tolist(),
                        processed_data['future_pose'][vid_id].tolist(),
                        processed_data['obs_frames_path'][vid_id].tolist()
                    ])
            else:
                for vid_id in range(len(processed_data['obs_pose'])):
                    for p_id in range(len(processed_data['future_pose'][vid_id])):
                        data.append([
                            '%d-%d' % (vid_id, 0),
                            processed_data['obs_pose'][vid_id][p_id].tolist(),
                            processed_data['future_pose'][vid_id][p_id].tolist(),
                            processed_data['obs_frames_path'][vid_id].tolist()
                        ])
        with open(os.path.join(self.is_interactive, output_name), 'a') as f_object:
            writer = csv.writer(f_object)
            writer.writerows(data)

    def __clean_data(self, data_type):
        if data_type == 'validation':
            data_type = 'valid'
        files_names = defaultdict()
        processed_data = defaultdict(np.array)
        files_names['obs_pose'] = f'3dpw_{data_type}_in.json'
        files_names['obs_frames_path'] = f'3dpw_{data_type}_frames_in.json'
        if data_type == 'train' or data_type == 'valid':
            files_names['future_pose'] = f'3dpw_{data_type}_out.json'
        for file_name_key, file_name in files_names.items():
            with open(os.path.join(self.dataset_path, file_name), 'r') as json_file:
                processed_data[file_name_key] = np.array(json.load(json_file), dtype=object)
        return processed_data
