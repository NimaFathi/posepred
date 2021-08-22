import csv
import json
import os
import numpy as np
from collections import defaultdict

from preprocessor.preprocessor import Processor, OUTPUT_DIR


class SoMoFPoseTrackPreprocessor(Processor):
    def __init__(self, mask, dataset_path, is_disentangle, is_interactive, obs_frame_num, pred_frame_num,
                 skip_frame_num,
                 use_video_once, custom_name):
        super(SoMoFPoseTrackPreprocessor, self).__init__(dataset_path, is_disentangle, is_interactive, obs_frame_num,
                                                    pred_frame_num, skip_frame_num, use_video_once, custom_name)
        self.mask = mask
        self.output_dir = os.path.join(OUTPUT_DIR, 'PoseTrack_interactive') if self.is_interactive else os.path.join(
            OUTPUT_DIR, 'PoseTrack')
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def normal(self, data_type='train'):
        files_names = defaultdict()
        processed_data = defaultdict(np.array)
        files_names['obs_pose'] = f'posetrack_{data_type}_in.json'
        files_names['obs_mask'] = f'posetrack_{data_type}_masks_in.json'
        files_names['obs_frames_path'] = f'posetrack_{data_type}_frames_in.json'
        if data_type == 'train' or data_type == 'valid':
            files_names['future_pose'] = f'posetrack_{data_type}_out.json'
            files_names['future_mask'] = f'posetrack_{data_type}_masks_out.json'
        for file_name_key, file_name in files_names.items():
            with open(os.path.join(self.dataset_path, file_name), 'r') as json_file:
                processed_data[file_name_key] = np.array(json.load(json_file), dtype=object)
        self.__save_csv(data_type, processed_data)

    def disentangle(self, data_type='train'):
        pass

    def __save_csv(self, data_type, processed_data):
        data = []
        if data_type == 'test':
            header = ['video_section', 'observed_pose', 'observed_mask', 'obs_frames']
        else:
            header = ['video_section', 'observed_pose', 'future_pose', 'observed_mask', 'future_mask', 'obs_frames']
        with open(os.path.join(self.output_dir, 'SOMOF_PoseTrack_wo_{}.csv'.format(data_type)), 'w') as f_object:
            writer = csv.writer(f_object)
            writer.writerow(header)
        if data_type == 'test':
            if self.is_interactive:
                for vid_id in range(len(processed_data['obs_pose'])):
                    data.append(['%d-%d' % (vid_id, 0),
                                 processed_data['obs_pose'][vid_id],
                                 processed_data['obs_mask'][vid_id],
                                 processed_data['obs_frames_path'][vid_id]
                                 ])
            else:
                for vid_id in range(len(processed_data['obs_pose'])):
                    for p_id in range(len(processed_data['future_pose'][vid_id])):
                        data.append([
                            '%d-%d' % (vid_id, 0), processed_data['obs_pose'][vid_id][p_id],
                            processed_data['obs_mask'][vid_id][p_id],
                            processed_data['obs_frames_path'][vid_id]
                        ])
        else:
            if self.is_interactive:
                for vid_id in range(len(processed_data['obs_pose'])):
                    data.append([
                        '%d-%d' % (vid_id, 0), processed_data['obs_pose'][vid_id],
                        processed_data['future_pose'][vid_id],
                        processed_data['obs_mask'][vid_id],
                        processed_data['future_mask'][vid_id], processed_data['obs_frames_path'][vid_id]
                    ])
            else:
                for vid_id in range(len(processed_data['obs_pose'])):
                    for p_id in range(len(processed_data['future_pose'][vid_id])):
                        data.append([
                            '%d-%d' % (vid_id, 0),
                            processed_data['obs_pose'][vid_id][p_id], processed_data['future_pose'][vid_id][p_id],
                            processed_data['obs_mask'][vid_id][p_id], processed_data['future_pose'][vid_id][p_id],
                            processed_data['obs_frames_path'][vid_id]
                        ])
        with open(os.path.join(self.output_dir, 'SOMOF_PoseTrack_wo_{}.csv'.format(data_type)), 'a') as f_object:
            writer = csv.writer(f_object)
            writer.writerows(data)
