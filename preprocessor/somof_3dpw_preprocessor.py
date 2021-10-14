import json
import logging
import os
from collections import defaultdict

import jsonlines
import numpy as np

from path_definition import PREPROCESSED_DATA_DIR
from preprocessor.preprocessor import Processor

logger = logging.getLogger(__name__)


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
        logger.info('start creating SoMoF-3DPW normal static data ... ')
        preprocessed_data = self.__clean_data(data_type)
        self.__save_csv(data_type, preprocessed_data)
        self.save_meta_data(self.meta_data, self.output_dir, True, data_type)

    def __save_csv(self, data_type, processed_data, file_type=None):
        if self.custom_name:
            if file_type is None:
                output_file_name = f'{data_type}_16_14_1_{self.custom_name}.csv'
            else:
                output_file_name = f'{file_type}_{data_type}_16_14_1_{self.custom_name}.csv'
        else:
            if file_type is None:
                output_file_name = f'{data_type}_16_14_1_SoMoF_3dpw.jsonl'
            else:
                output_file_name = f'{file_type}_{data_type}_16_14_1_SoMoF_3dpw.jsonl'
        data = []
        assert os.path.exists(os.path.join(
            self.output_dir,
            output_file_name
        )) is False, f"preprocessed file exists at {os.path.join(self.output_dir, output_file_name)}"
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
                    self.update_meta_data(self.meta_data, processed_data['obs_pose'][vid_id], 3)
                    for p_id in range(len(processed_data['future_pose'][vid_id])):
                        data.append([
                            '%d-%d' % (vid_id, 0),
                            processed_data['obs_pose'][vid_id][p_id].tolist(),
                            processed_data['future_pose'][vid_id][p_id].tolist(),
                            processed_data['obs_frames_path'][vid_id].tolist()
                        ])
        with jsonlines.open(os.path.join(self.output_dir, output_file_name), 'a') as writer:
            if data_type == 'test':
                for data_row in data:
                    writer.write({
                        'video_section': data_row[0],
                        'observed_pose': data_row[1],
                        'observed_image_path': data_row[2]
                    })
            else:
                for data_row in data:
                    writer.write({
                        'video_section': data_row[0],
                        'observed_pose': data_row[1],
                        'future_pose': data_row[2],
                        'observed_image_path': data_row[3]
                    })

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
